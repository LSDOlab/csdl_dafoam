"""Run DAFoam calls from inside the OpenFOAM case directory."""
import os
from contextlib import contextmanager


@contextmanager
def working_directory(path):
    """Temporarily ``chdir`` into ``path`` (no-op when ``path`` is None)."""
    if path is None:
        yield
        return
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def in_run_directory(dafoam_instance):
    """Context manager that enters the case directory ``instantiate_dafoam`` recorded.

    DAFoam's Python layer resolves several paths against the current working directory
    (solution renaming, adjoint field output, failed-mesh dumps), while the surrounding
    CSDL script usually has a different working directory (geometry caches etc.). Every
    DAFoam call made by the CSDL operations goes through this context manager.
    """
    return working_directory(getattr(dafoam_instance, "run_directory", None))
