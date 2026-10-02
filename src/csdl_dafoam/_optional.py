"""Helpers for importing the heavy, non-pip-installable runtime dependencies lazily.

DAFoam, IDWarp, petsc4py and mpi4py cannot be installed from PyPI wheels (DAFoam links
against a source-built OpenFOAM), so ``import csdl_dafoam`` must work without them.
They are imported only at the point a DAFoam run is actually set up, and a missing one
produces an error that says how to install it instead of a bare ``ModuleNotFoundError``.
"""
import importlib

_INSTALL_HINT = (
    "csdl_dafoam needs the DAFoam stack (DAFoam, IDWarp, petsc4py, mpi4py), which is "
    "installed through conda rather than pip. See the installation guide: create the "
    "environment with `conda env create -f environment.yml` (docs/installation.md)."
)


def require(module_name):
    """Import ``module_name`` or raise an ImportError that points at the install guide."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import '{module_name}'. {_INSTALL_HINT}") from exc
