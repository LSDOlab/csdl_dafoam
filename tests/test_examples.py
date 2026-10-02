"""Run the example scripts end to end against the linear fake CFD solver.

This checks the scripts themselves (argument parsing, case preparation, model build,
analysis, optimization, output files), not the CFD.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeAirfoilDAFoam, FakeComm, FakeJaxAirfoilDAFoam, FakePETSc

pytest.importorskip("lsdo_geo")
pytest.importorskip("lsdo_function_spaces")
pytest.importorskip("jax")
pytest.importorskip("idwarp_jax")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_env(monkeypatch, tmp_path):
    import csdl_dafoam.airfoil as airfoil_module
    import csdl_dafoam.solver as solver_module
    from csdl_dafoam import foam_io

    fake_mpi4py = types.ModuleType("mpi4py")
    fake_mpi4py.MPI = types.SimpleNamespace(SUM=None, COMM_WORLD=FakeComm())
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py)
    def fake_instantiate(options, comm, run_directory, mesh_options):
        if mesh_options is None:  # warper="jax": DAFoam owns only its local volume points
            return FakeJaxAirfoilDAFoam(foam_io.read_mesh(run_directory)["points"], run_directory=str(run_directory))
        return FakeAirfoilDAFoam(foam_io.read_patch_points(run_directory, "wing"), run_directory=str(run_directory))

    monkeypatch.setattr(airfoil_module, "instantiate_dafoam", fake_instantiate)
    monkeypatch.setattr(solver_module, "_petsc", lambda: FakePETSc)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.chdir(tmp_path)
    # the examples build their own communicator via mpi4py
    return tmp_path


def run_example(name, argv, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *argv])
    module.main()


@pytest.mark.parametrize("case", ["naca0012_euler", "naca0012"])
@pytest.mark.slow
def test_analysis_example(example_env, monkeypatch, capsys, case):
    run_example("airfoil_analysis", ["--case", case, "--workdir", str(example_env / "run")], monkeypatch)
    out = capsys.readouterr().out
    assert f"case={case}" in out and "CL =" in out and "CD =" in out
    assert (example_env / "run" / "case" / "constant" / "polyMesh" / "boundary").is_file()


@pytest.mark.slow
def test_analysis_example_check_totals(example_env, monkeypatch, capsys):
    run_example("airfoil_analysis", ["--workdir", str(example_env / "run"), "--check-totals"], monkeypatch)
    out = capsys.readouterr().out
    assert "adjoint vs finite-difference" in out
    # every printed column must agree to far better than the sensitivity itself. 1%, not tighter:
    # IDWarp-JAX prunes its kd-tree interactions to err_tol = 5e-4, which makes forward and reverse
    # differ by a few tenths of a percent; wiring errors are O(1).
    lines = [l for l in out.splitlines() if "max|adjoint - FD|" in l]
    assert lines
    for line in lines:
        diff = float(line.split("= ")[1].split()[0])
        scale = float(line.split("max|FD| = ")[1].rstrip(")"))
        assert diff <= 1e-2 * scale + 1e-12, line


@pytest.mark.parametrize(
    "extra",
    [
        [],  # defaults: IDWarp-JAX + OpenSQP
        ["--optimizer", "PySLSQP"],
        ["--warper", "idwarp"],
    ],
    ids=["jax+OpenSQP", "jax+PySLSQP", "idwarp+OpenSQP"],
)
@pytest.mark.slow
def test_optimization_example_writes_results(example_env, monkeypatch, capsys, extra):
    pytest.importorskip("modopt")
    pytest.importorskip("pyslsqp")
    run_example(
        "airfoil_optimization",
        ["--workdir", str(example_env / "run"), "--maxiter", "3", *extra],
        monkeypatch,
    )
    out = capsys.readouterr().out
    assert "initial:" in out and "final:" in out and "CD change" in out
    expected_optimizer = extra[extra.index("--optimizer") + 1] if "--optimizer" in extra else "OpenSQP"
    expected_warper = extra[extra.index("--warper") + 1] if "--warper" in extra else "jax"
    assert f"warper={expected_warper}" in out and f"optimizer={expected_optimizer}" in out
    saved = np.load(example_env / "run" / "optimized_airfoil.npz")
    assert saved["x_initial"].shape == saved["x_optimized"].shape
    assert saved["thickness"].shape == (3,) and saved["camber"].shape == (3,)
    assert not np.allclose(saved["x_initial"], saved["x_optimized"]), "the optimizer never moved the design"
    assert (example_env / "run" / "ASO_2DAF_rank0_outputs").is_dir()
