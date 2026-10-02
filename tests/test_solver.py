"""DAFoamSolver / DAFoamFunctions against a linear fake DAFoam with known derivatives."""
import os

import csdl_alpha as csdl
import numpy as np
import pytest
from fakes import FakeLinearDAFoam

from csdl_dafoam.solver import DAFoamError, DAFoamFunctions, DAFoamSolver


def build_model(fake, **solver_kwargs):
    """x_vol, (airspeed, aoa), p, T  ->  states  ->  CD, CL."""
    rng = np.random.default_rng(7)
    inputs = csdl.VariableGroup()
    inputs.aero_vol_coords = csdl.Variable(value=rng.standard_normal(fake.n_vol), name="x")
    inputs.patch_velocity = csdl.Variable(value=np.array([238.0, 3.0]), name="v")
    inputs.pressure = csdl.Variable(value=np.array([101325.0]), name="p")
    inputs.temperature = csdl.Variable(value=np.array([300.0]), name="T")

    solver = DAFoamSolver(fake, **solver_kwargs)
    states = solver.evaluate(inputs)
    outputs = DAFoamFunctions(fake).evaluate(states, inputs)
    return inputs, states, outputs


@pytest.mark.parametrize("method", ["fixedPoint", "Krylov"])
def test_primal_and_total_derivatives_match_analytic(fake_petsc, recorder, method):
    fake = FakeLinearDAFoam(adjoint_method=method)
    inputs, states, out = build_model(fake)
    recorder.stop()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()

    x = inputs.aero_vol_coords.value
    v = inputs.patch_velocity.value
    p, T = inputs.pressure.value, inputs.temperature.value
    rhs = fake.B["aero_vol_coords"] @ x + fake.B["patch_velocity"] @ v + fake.B["pressure"] @ p + fake.B["temperature"] @ T
    w = np.linalg.solve(fake.A, rhs)

    np.testing.assert_allclose(sim[states], w, rtol=1e-12)
    for name in ("CD", "CL"):
        expected = fake.c[name] @ w + fake.d[name] @ x + fake.e[name] @ v
        np.testing.assert_allclose(sim[getattr(out, name)], expected, rtol=1e-12)

    ofs = [out.CD, out.CL]
    wrts = [inputs.aero_vol_coords, inputs.patch_velocity, inputs.pressure, inputs.temperature]
    totals = sim.compute_totals(ofs, wrts)

    Ainv_T_c = {k: np.linalg.solve(fake.A.T, fake.c[k]) for k in ("CD", "CL")}
    for name, of in (("CD", out.CD), ("CL", out.CL)):
        np.testing.assert_allclose(
            totals[of, inputs.aero_vol_coords], (Ainv_T_c[name] @ fake.B["aero_vol_coords"] + fake.d[name])[None, :], rtol=1e-10
        )
        np.testing.assert_allclose(
            totals[of, inputs.patch_velocity], (Ainv_T_c[name] @ fake.B["patch_velocity"] + fake.e[name])[None, :], rtol=1e-10
        )
        np.testing.assert_allclose(
            totals[of, inputs.pressure], (Ainv_T_c[name] @ fake.B["pressure"])[None, :], rtol=1e-10
        )
        np.testing.assert_allclose(
            totals[of, inputs.temperature], (Ainv_T_c[name] @ fake.B["temperature"])[None, :], rtol=1e-10
        )


def test_dafoam_calls_run_inside_case_directory(fake_petsc, recorder, tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    fake = FakeLinearDAFoam(run_directory=str(case))
    inputs, states, out = build_model(fake)
    recorder.stop()
    before = os.getcwd()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()
    sim.compute_totals([out.CD], [inputs.patch_velocity])

    assert fake.calls, "fake DAFoam was never called"
    assert {os.path.realpath(cwd) for _, cwd in fake.calls} == {os.path.realpath(case)}
    assert os.getcwd() == before


def test_failed_mesh_check_raises(fake_petsc, recorder):
    fake = FakeLinearDAFoam()
    fake.mesh_ok = 0
    fake.writeFailedMesh = lambda: None
    build_model(fake)
    recorder.stop()
    with pytest.raises(DAFoamError, match="mesh check"):
        csdl.experimental.PySimulator(recorder).run()


def test_mesh_check_can_be_disabled(fake_petsc, recorder):
    fake = FakeLinearDAFoam()
    fake.mesh_ok = 0
    build_model(fake, check_mesh=False)
    recorder.stop()
    csdl.experimental.PySimulator(recorder).run()  # does not raise


def test_primal_failure_raises(fake_petsc, recorder):
    fake = FakeLinearDAFoam()
    fake.primalFail = 1
    build_model(fake)
    recorder.stop()
    with pytest.raises(DAFoamError, match="primal"):
        csdl.experimental.PySimulator(recorder).run()


@pytest.mark.parametrize("policy, expectation", [("warn", "warns"), ("raise", "raises")])
def test_adjoint_failure_policy(fake_petsc, recorder, policy, expectation):
    fake = FakeLinearDAFoam()
    fake.adjoint_fail = 1
    inputs, _, out = build_model(fake, on_adjoint_failure=policy)
    recorder.stop()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()
    if expectation == "warns":
        with pytest.warns(RuntimeWarning, match="adjoint"):
            sim.compute_totals([out.CD], [inputs.patch_velocity])
    else:
        with pytest.raises(DAFoamError, match="adjoint"):
            sim.compute_totals([out.CD], [inputs.patch_velocity])


def test_invalid_adjoint_policy_rejected(fake_petsc, recorder):
    with pytest.raises(ValueError):
        DAFoamSolver(FakeLinearDAFoam(), on_adjoint_failure="ignore")
