"""csdl_dafoam: CSDL interface to DAFoam for aerodynamic shape optimization.

Importing this package is cheap and does not require DAFoam, PETSc, MPI or CSDL. The
public names below are loaded on first access.
"""
from importlib import import_module

__version__ = "0.1.0"

_PUBLIC = {
    # name: submodule
    "instantiate_dafoam": "solver",
    "instantiateDAFoam": "solver",
    "DAFoamSolver": "solver",
    "DAFoamFunctions": "solver",
    "DAFoamError": "solver",
    "DAFoamMeshWarper": "mesh_warp",
    "JaxIDWarp": "jax_warp",
    "JaxMeshWarper": "jax_warp",
    "compute_dafoam_input_variables": "inputs",
    "compute_ambient_conditions_group": "atmosphere",
    "build_airfoil_model": "airfoil",
    "AirfoilModel": "airfoil",
    "FlightConditions": "airfoil",
    "copy_case": "cases",
    "prepare_case": "cases",
    "geometry_path": "cases",
    "list_cases": "cases",
}

__all__ = ["__version__", *_PUBLIC]


def __getattr__(name):
    if name in _PUBLIC:
        return getattr(import_module(f"{__package__}.{_PUBLIC[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
