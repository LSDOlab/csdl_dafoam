"""Geometry handling: STEP import with caching, and the thickness/camber FFD parameterization."""
import hashlib
import pickle
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from csdl_dafoam._run_directory import working_directory

__all__ = [
    "write_geometry_pickle",
    "read_geometry_pickle",
    "read_simple_pickle",
    "write_simple_pickle",
    "load_geometry",
    "setup_airfoil_geometry",
    "setup_geometry",
]


# --------------------------------------------------------------------------------------
# Pickle helpers
# --------------------------------------------------------------------------------------
def write_geometry_pickle(geometry, geometry_file_path):
    """Pickle ``geometry`` with its coefficients stored as plain arrays."""
    with open(geometry_file_path, "wb+") as handle:
        geometry_copy = geometry.copy()
        for i, function in geometry.functions.items():
            function_copy = function.copy()
            function_copy.coefficients = function.coefficients.value.copy()
            geometry_copy.functions[i] = function_copy

        pickle.dump(geometry_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_geometry_pickle(geometry_file_path):
    """Inverse of :func:`write_geometry_pickle`. Needs an active ``csdl.Recorder``."""
    import lsdo_geo as lg

    with open(geometry_file_path, "rb") as handle:
        function_set = pickle.load(handle)
    for function in function_set.functions.values():
        function.coefficients = csdl.Variable(value=function.coefficients)

    return lg.Geometry(
        functions=function_set.functions,
        function_names=function_set.function_names,
        name=function_set.name,
        space=function_set.space,
    )


def read_simple_pickle(file_path):
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def write_simple_pickle(var_to_write, file_path):
    with open(file_path, "wb+") as handle:
        pickle.dump(var_to_write.copy(), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _file_digest(path, length=12):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]


# --------------------------------------------------------------------------------------
# STEP import with an MPI-aware cache
# --------------------------------------------------------------------------------------
def load_geometry(stp_file_path, cache_dir, comm=None):
    """Import a STEP file as an ``lsdo_geo.Geometry``, caching the import as a pickle.

    The cache file name includes a digest of the STEP file, so editing the geometry
    invalidates the cache. Rank 0 imports and writes; other ranks wait and read. A
    cache that cannot be unpickled (for example after an ``lsdo_geo`` upgrade) is
    regenerated. A ``csdl.Recorder`` must be active.
    """
    import lsdo_geo

    stp_file_path = Path(stp_file_path)
    cache_dir = Path(cache_dir)
    rank = comm.Get_rank() if comm is not None else 0
    cache_file = cache_dir / f"{stp_file_path.stem}_{_file_digest(stp_file_path)}_import.pickle"

    def barrier():
        if comm is not None:
            comm.Barrier()

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        geometry = None
        if cache_file.is_file():
            try:
                geometry = read_geometry_pickle(cache_file)
            except Exception:  # stale/incompatible cache -> regenerate
                geometry = None
        if geometry is None:
            with working_directory(cache_dir):
                geometry = lsdo_geo.import_geometry(str(stp_file_path), parallelize=False)
            write_geometry_pickle(geometry, cache_file)
    barrier()
    if rank != 0:
        geometry = read_geometry_pickle(cache_file)
    return geometry


# --------------------------------------------------------------------------------------
# Thickness/camber FFD parameterization of an airfoil of unit span
# --------------------------------------------------------------------------------------
def setup_airfoil_geometry(
    geometry,
    num_ffd_coefficients_chordwise,
    percent_change_in_thickness_dof,
    normalized_percent_camber_change_dof,
    pitch_angle_deg=0.0,
):
    """Parameterize ``geometry`` (an airfoil in the x-z plane, span along y) with an FFD block.

    The FFD block has ``num_ffd_coefficients_chordwise`` control points along the chord,
    2 across the span (the two symmetry planes) and 2 through the thickness. The interior
    chordwise control points (all but the first and last) are the design variables:

    ``percent_change_in_thickness_dof``
        shape ``(num_ffd_coefficients_chordwise - 2,)``; percent change of the local
        block thickness.
    ``normalized_percent_camber_change_dof``
        same shape; vertical shift of the control points as a percent of the chord.

    The geometry's coefficients are updated in place and the geometry is returned.
    ``pitch_angle_deg`` rotates the geometry about the span axis through the leading edge
    (fixed, not a design variable).
    """
    import lsdo_geo as lg

    num_ffd_sections = 2  # symmetry boundaries (left, right)
    num_chordwise = num_ffd_coefficients_chordwise

    ffd_block = lg.construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(num_chordwise, num_ffd_sections, 2),
        degree=(3, 1, 1),
    )

    # lsdo_geo >= 1.0 renamed these; the old names are deprecated aliases (and the only names
    # in older versions).
    sectional_parameterization = getattr(lg, "SectionalParameterization", None) or lg.VolumeSectionalParameterization
    sectional_parameters_class = getattr(lg, "SectionalParameters", None) or lg.VolumeSectionalParameterizationInputs

    ffd_sectional_parameterization = sectional_parameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,  # shape (num_chordwise, 2, 2, 3)
        principal_parametric_dimension=1,
    )
    sectional_parameters = sectional_parameters_class()
    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

    # (1) thickness
    original_block_thickness = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]

    percent_change_in_thickness = csdl.Variable(shape=(num_chordwise, num_ffd_sections), value=0.0)
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 0], percent_change_in_thickness_dof)
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 1], percent_change_in_thickness_dof)

    delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
    thickness_upper_translation = 1 / 2 * delta_block_thickness
    thickness_lower_translation = -thickness_upper_translation

    ffd_coefficients = ffd_coefficients.set(
        csdl.slice[:, :, 1, 2], ffd_coefficients[:, :, 1, 2] + thickness_upper_translation
    )
    ffd_coefficients = ffd_coefficients.set(
        csdl.slice[:, :, 0, 2], ffd_coefficients[:, :, 0, 2] + thickness_lower_translation
    )

    # (2) camber, normalized by the original block (about the chord) length
    normalized_percent_camber_change = csdl.Variable(shape=(num_chordwise, num_ffd_sections), value=0.0)
    normalized_percent_camber_change = normalized_percent_camber_change.set(
        csdl.slice[1:-1, 0], normalized_percent_camber_change_dof
    )
    normalized_percent_camber_change = normalized_percent_camber_change.set(
        csdl.slice[1:-1, 1], normalized_percent_camber_change_dof
    )

    block_length = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
    # NOTE: block_length is the spacing of the first two chordwise control points, i.e.
    # chord / (num_chordwise - 1) for this uniformly spaced block, not the chord. A camber dof
    # of 5 therefore moves the control point by 5% of that spacing (0.0125 chord for 5 points).
    camber_change = (normalized_percent_camber_change / 100) * block_length
    ffd_coefficients = ffd_coefficients.set(
        csdl.slice[:, :, :, 2],
        ffd_coefficients[:, :, :, 2]
        + csdl.expand(camber_change, (num_chordwise, num_ffd_sections, 2), "ij->ijk"),
    )

    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients)

    if pitch_angle_deg != 0:
        rotation_origin = geometry.evaluate(geometry.project(np.array([0.0, 0.0, 0.0])))
        geometry.rotate(rotation_origin, np.array([0.0, 1.0, 0.0]), pitch_angle_deg, units="degrees")

    return geometry


def setup_geometry(geometry, geometry_values_dict):
    """Dict-style wrapper around :func:`setup_airfoil_geometry` (name used by the original scripts)."""
    return setup_airfoil_geometry(
        geometry,
        geometry_values_dict["num_ffd_coefficients_chordwise"],
        geometry_values_dict["percent_change_in_thickness_dof"],
        geometry_values_dict["normalized_percent_camber_change_dof"],
        geometry_values_dict.get("pitch_angle_deg", 0.0),
    )
