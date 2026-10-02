"""Volume mesh deformation with IDWarp-JAX, as a CSDL operation.

`IDWarp-JAX <https://github.com/LSDOlab/idwarp-jax>`_ is LSDO Lab's JAX implementation of MDO
Lab's IDWarp (inverse-distance-weighted deformation). Unlike the Fortran IDWarp that DAFoam
normally uses, it works on the *whole* mesh in one process and has no compiled dependency:

* the moving surface is **global** (all wall points of the design patches, in ascending
  OpenFOAM point-index order, which is the order :func:`csdl_dafoam.foam_io.read_patch_points`
  returns), so no surface gathering across MPI ranks is needed;
* the deformed volume is global too, and each rank takes its own (decomposed) points from it.

:class:`JaxMeshWarper` is the drop-in alternative to
:class:`csdl_dafoam.mesh_warp.DAFoamMeshWarper` (which wraps the Fortran IDWarp attached to a
DAFoam instance).
"""
import logging

import csdl_alpha as csdl
import numpy as np

from csdl_dafoam._optional import require

logger = logging.getLogger(__name__)

__all__ = ["JaxIDWarp", "match_local_points", "JaxMeshWarper"]


class JaxIDWarp:
    """Prepared IDWarp-JAX deformation of a case's volume mesh.

    Parameters
    ----------
    case_dir : path-like
        Case with a reconstructed ``constant/polyMesh`` (decomposed cases keep it).
    surface_patches : sequence of str
        Patches that move (DAFoam's ``designSurfaces``).
    device : jax.Device, optional
        Defaults to the first CPU device.
    LdefFact : float
        Deformation-length factor. 100 is IDWarp-JAX's own default and gave the best cell
        quality on the bundled airfoil mesh (see ``docs/how-it-works.md``).
    **options
        Passed to ``idwarp_jax.build_volume_pts_func`` (``err_tol``, ``bucket_size``,
        ``useRotations``, ``cornerAngle``, ...).

    Attributes
    ----------
    wall_points : ndarray (n_wall, 3)
        Undeformed moving-surface coordinates, in the order :meth:`warp` expects.
    points : ndarray (n_points, 3)
        Undeformed global volume mesh.
    mesh : dict
        :func:`csdl_dafoam.foam_io.read_mesh` output (for quality checks).
    """

    def __init__(self, case_dir, surface_patches, device=None, LdefFact=100.0, **options):
        jax = require("jax")
        jax.config.update("jax_enable_x64", True)  # the mesh and the warp are double precision
        idwarp_jax = require("idwarp_jax")
        from csdl_dafoam.foam_io import idwarp_jax_mesh_dict

        mesh_dict, self.mesh = idwarp_jax_mesh_dict(case_dir, surface_patches)
        self.points = self.mesh["points"]
        device = device if device is not None else jax.devices("cpu")[0]

        options.setdefault("print_timings", False)
        options.setdefault("progress", False)
        self._forward, self._reverse, aux = idwarp_jax.build_volume_pts_func(
            mesh_dict,
            self.points,
            device,
            surface_face_type=1,
            LdefFact=LdefFact,
            return_aux=True,
            **options,
        )
        self.wall_points = np.asarray(aux["wall_points"])
        self.wall_ids = np.asarray(aux["wall_ids"])
        self._flow = {"pitch": 0.0}

    def warp(self, surface_points):
        """Deformed global volume points for the given ``(n_wall, 3)`` surface points."""
        return np.asarray(self._forward(np.asarray(surface_points, dtype=np.float64).reshape(-1, 3), self._flow))

    def vjp(self, surface_points, volume_seed):
        """``(d volume / d surface)^T`` applied to an ``(n_points, 3)`` seed; returns ``(n_wall, 3)``."""
        d_surface, _ = self._reverse(
            np.asarray(surface_points, dtype=np.float64).reshape(-1, 3),
            self._flow,
            np.asarray(volume_seed, dtype=np.float64).reshape(-1, 3),
        )
        return np.asarray(d_surface)


def match_local_points(global_points, local_points, tolerance=1e-9):
    """Index of each local (per-rank) mesh point in the global mesh.

    A decomposed mesh copies point coordinates bit for bit, so an exact nearest-neighbor
    match recovers the global index; points on processor boundaries appear on several ranks
    and map to the same global point. Raises if any point has no match within ``tolerance``.
    """
    from scipy.spatial import cKDTree

    local_points = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    distance, index = cKDTree(global_points).query(local_points)
    worst = distance.max() if distance.size else 0.0
    if worst > tolerance:
        raise ValueError(
            f"{int((distance > tolerance).sum())} local mesh points have no match in the global "
            f"mesh (worst distance {worst:.3e}); is constant/polyMesh the mesh the case was "
            "decomposed from?"
        )
    return index


class JaxMeshWarper(csdl.CustomExplicitOperation):
    """Global surface coordinates ``x_surf`` -> this rank's volume coordinates ``x_vol``.

    ``x_surf`` is the flattened ``(n_wall * 3,)`` **global** moving surface (the same on every
    rank). ``x_vol`` is flattened ``(n_local_points * 3,)`` in DAFoam's local point order.

    Every rank warps the whole mesh (cheap next to the CFD solve) and keeps its points. In the
    reverse pass each rank scatters its volume seed into a global seed (zeros elsewhere) and
    applies the global VJP; the per-rank surface contributions are summed by the surrounding
    ``csdl`` MPI region, because ``x_surf`` enters it as a global input.
    """

    def __init__(self, warper: JaxIDWarp, local_to_global):
        super().__init__()
        self.warper = warper
        self.local_to_global = np.asarray(local_to_global, dtype=np.int64)

    def evaluate(self, x_surf):
        logger.debug("JaxMeshWarper.evaluate")
        self.declare_input("x_surf", x_surf)
        return self.create_output("x_vol", (3 * self.local_to_global.size,))

    def compute(self, inputs, outputs):
        logger.debug("JaxMeshWarper.compute")
        volume = self.warper.warp(inputs["x_surf"])
        outputs["x_vol"] = volume[self.local_to_global].flatten()

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        logger.debug("JaxMeshWarper.compute_jacvec_product")
        if mode == "fwd":
            raise NotImplementedError("forward mode has not been implemented for JaxMeshWarper")
        if "x_vol" in d_outputs and "x_surf" in d_inputs:
            seed = np.zeros_like(self.warper.points)
            # np.add.at: a point shared by several ranks (or duplicated locally) accumulates
            np.add.at(seed, self.local_to_global, np.asarray(d_outputs["x_vol"]).reshape(-1, 3))
            d_inputs["x_surf"] += self.warper.vjp(inputs["x_surf"], seed).flatten()
