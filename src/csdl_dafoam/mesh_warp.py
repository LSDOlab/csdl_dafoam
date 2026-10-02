"""CSDL operation wrapping IDWarp's volume-mesh deformation."""
import logging

import csdl_alpha as csdl

from csdl_dafoam._run_directory import in_run_directory

logger = logging.getLogger(__name__)

__all__ = ["DAFoamMeshWarper"]


class DAFoamMeshWarper(csdl.CustomExplicitOperation):
    """Map design-surface coordinates ``x_surf`` to volume coordinates ``x_vol`` (IDWarp).

    Modeled on ``DAFoamWarper`` in DAFoam's ``mphys_dafoam.py``. ``x_surf`` is the
    flattened ``(n_surf_points_on_this_rank * 3,)`` array of the design surfaces.
    """

    def __init__(self, dafoam_instance):
        super().__init__()
        self.dafoam_instance = dafoam_instance

    def evaluate(self, x_surf):
        logger.debug("DAFoamMeshWarper.evaluate")
        self.declare_input("x_surf", x_surf)
        solver_grid_size = self.dafoam_instance.mesh.getSolverGrid().shape
        return self.create_output("x_vol", solver_grid_size)

    def compute(self, inputs, outputs):
        """Forward evaluation: mesh warp, ``x_surf -> x_vol``."""
        logger.debug("DAFoamMeshWarper.compute")
        dafoam_instance = self.dafoam_instance

        with in_run_directory(dafoam_instance):
            x_surf = inputs["x_surf"]
            dafoam_instance.setSurfaceCoordinates(x_surf.reshape((-1, 3)), dafoam_instance.designSurfacesGroup)
            dafoam_instance.mesh.warpMesh()
            outputs["x_vol"] = dafoam_instance.mesh.getSolverGrid()

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        """Reverse mode: ``d_x_surf += (dx_vol/dx_surf)^T d_x_vol``."""
        logger.debug("DAFoamMeshWarper.compute_jacvec_product")
        if mode == "fwd":
            raise NotImplementedError("forward mode has not been implemented for DAFoamMeshWarper")

        dafoam_instance = self.dafoam_instance

        if "x_vol" in d_outputs and "x_surf" in d_inputs:
            with in_run_directory(dafoam_instance):
                dxV = d_outputs["x_vol"]
                dafoam_instance.mesh.warpDeriv(dxV)
                dxS = dafoam_instance.mesh.getdXs()
                dxS = dafoam_instance.mapVector(
                    dxS, dafoam_instance.allWallsGroup, dafoam_instance.designSurfacesGroup
                )
            d_inputs["x_surf"] += dxS.flatten()
