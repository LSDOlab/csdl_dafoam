import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))  # so tests can `import fakes`

from fakes import FakePETSc  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Tests marked ``dafoam`` need a real DAFoam install; skip them when it is absent."""
    try:
        import dafoam  # noqa: F401  (the default warper, IDWarp-JAX, needs no compiled IDWarp)
        import petsc4py  # noqa: F401
        from mpi4py import MPI  # noqa: F401

        have_dafoam = True
    except Exception:
        have_dafoam = False
    if have_dafoam:
        return
    skip = pytest.mark.skip(reason="DAFoam/petsc4py/mpi4py are not installed")
    for item in items:
        if "dafoam" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _run_in_temp_directory(tmp_path, monkeypatch):
    """lsdo_geo and modOpt write caches/outputs into the working directory; keep them out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_petsc(monkeypatch):
    """Route ``csdl_dafoam.solver`` to the numpy-backed PETSc fake."""
    import csdl_dafoam.solver as solver

    monkeypatch.setattr(solver, "_petsc", lambda: FakePETSc)
    return FakePETSc


@pytest.fixture
def recorder():
    """A fresh, active CSDL recorder."""
    import csdl_alpha as csdl

    rec = csdl.Recorder(inline=False)
    rec.start()
    yield rec
    try:
        rec.stop()  # tests normally stop it themselves before building a simulator
    except Exception:
        pass
