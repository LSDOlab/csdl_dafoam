"""`import csdl_dafoam` must work without DAFoam, PETSc, MPI or CSDL installed."""
import subprocess
import sys
import textwrap


def test_import_does_not_pull_in_heavy_dependencies():
    code = textwrap.dedent(
        """
        import sys
        import csdl_dafoam
        heavy = {"dafoam", "idwarp", "petsc4py", "mpi4py", "csdl_alpha", "lsdo_geo", "jax"}
        loaded = heavy & set(sys.modules)
        assert not loaded, f"imported eagerly: {sorted(loaded)}"
        assert csdl_dafoam.__version__
        assert "DAFoamSolver" in dir(csdl_dafoam)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_dafoam_gives_actionable_error(monkeypatch, tmp_path):
    import pytest

    import csdl_dafoam.solver as solver

    monkeypatch.setitem(sys.modules, "dafoam", None)  # makes `import dafoam` raise ImportError
    with pytest.raises(ImportError, match="environment.yml"):
        solver.instantiate_dafoam({}, comm=None, run_directory=tmp_path)


def test_unknown_attribute_raises():
    import pytest

    import csdl_dafoam

    with pytest.raises(AttributeError):
        csdl_dafoam.does_not_exist


def test_every_advertised_public_name_resolves():
    import csdl_dafoam

    for name in csdl_dafoam.__all__:
        assert getattr(csdl_dafoam, name) is not None, name
