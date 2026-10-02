"""End-to-end ``build_airfoil_model`` with a fake CFD solver.

Real here: the bundled STEP geometry and CFD surface mesh, B-spline projection, the FFD
thickness/camber parameterization, the CSDL graph (including its MPI region and the JAX
backend), the standard atmosphere and the input mapping. Fake: the flow solver and mesh
warp (linear, see ``fakes.py``). A real DAFoam run is covered by ``test_dafoam_integration``.
"""
import sys
import types

import numpy as np
import pytest
from fakes import FakeAirfoilDAFoam, FakeComm, FakeJaxAirfoilDAFoam

pytest.importorskip("lsdo_geo")
pytest.importorskip("lsdo_function_spaces")
pytest.importorskip("jax")

from csdl_dafoam import cases, foam_io  # noqa: E402
from csdl_dafoam import airfoil as airfoil_module  # noqa: E402
from csdl_dafoam.airfoil import FlightConditions, build_airfoil_model  # noqa: E402
from csdl_dafoam.options import naca0012_options  # noqa: E402


@pytest.fixture(scope="module", params=["jax", "idwarp"])
def built(tmp_path_factory, request):
    warper = request.param
    import csdl_dafoam.solver as solver_module
    from fakes import FakePETSc

    root = tmp_path_factory.mktemp("airfoil")
    case = cases.copy_case("naca0012", root / "case")
    surface = foam_io.read_patch_points(case, "wing")
    if warper == "jax":
        pytest.importorskip("idwarp_jax")
        fake = FakeJaxAirfoilDAFoam(foam_io.read_mesh(case)["points"], run_directory=str(case))
    else:
        fake = FakeAirfoilDAFoam(surface, run_directory=str(case))

    # csdl's MPI region imports mpi4py lazily for an all-reduce that is the identity on one rank
    fake_mpi4py = types.ModuleType("mpi4py")
    fake_mpi4py.MPI = types.SimpleNamespace(SUM=None, COMM_WORLD=FakeComm())
    sys.modules["mpi4py"] = fake_mpi4py

    original = airfoil_module.instantiate_dafoam
    original_petsc = solver_module._petsc
    airfoil_module.instantiate_dafoam = lambda *args, **kwargs: fake
    solver_module._petsc = lambda: FakePETSc
    try:
        model = build_airfoil_model(
            case_dir=case,
            da_options=naca0012_options(),
            geometry_file=cases.geometry_path(),
            cache_dir=root / "cache",
            flight=FlightConditions(),
            comm=FakeComm(),
            check_mesh=False,
            timing=False,
            warper=warper,
        )
        yield model, fake, surface, root
    finally:
        airfoil_module.instantiate_dafoam = original
        solver_module._petsc = original_petsc
        sys.modules.pop("mpi4py", None)


def test_design_variables_and_problem_definition(built):
    model, *_ = built
    dvs = model.design_variables
    assert dvs["thickness"].shape == (3,)
    assert dvs["camber"].shape == (3,)
    assert dvs["angle_of_attack"].value == pytest.approx(3.0)
    assert model.recorder.active_graph is not None


def test_geometry_caches_were_written_outside_the_case(built):
    _, _, _, root = built
    names = {p.name for p in (root / "cache").iterdir()}
    assert any(n.endswith("_import.pickle") for n in names)
    assert any(n.startswith("projected_surface_mesh_") for n in names)
    assert not any((root / "case").glob("projected_surface_mesh_*")), "case directory must stay clean"
    assert not (root / "case" / "stored_files").exists()


def test_analysis_matches_the_linear_fake(built):
    model, fake, surface, _ = built
    sim = model.make_simulator(additional_outputs=[model.x_surf])
    result = model.analyze(sim)

    # at the undeformed design the FFD reproduces the CFD surface (projection error ~1e-6)
    x = np.asarray(sim[model.x_surf]).flatten()
    np.testing.assert_allclose(x, surface.flatten(), atol=1e-4)

    if model.jax_warp is not None:
        # what DAFoam receives: IDWarp-JAX's deformed global mesh, reordered to the local points
        x = model.jax_warp.warp(x.reshape(-1, 3))[model.local_to_global].flatten()
    v = np.array([238.0, 3.0])
    p, T = 101325.0, 300.0
    rhs = (
        fake.B["aero_vol_coords"] @ x + fake.B["patch_velocity"] @ v
        + fake.B["pressure"] @ [p] + fake.B["temperature"] @ [T]
    )
    w = np.linalg.solve(fake.A, rhs)
    for name in ("CD", "CL"):
        expected = fake.c[name] @ w + fake.d[name] @ x + fake.e[name] @ v
        assert result[name] == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_derivatives_through_geometry_warp_and_flow_agree_with_finite_differences(built):
    """The reverse-mode chain d(CD, CL)/d(thickness, camber, alpha) matches central differences."""
    model, *_ = built
    sim = model.make_simulator()
    sim.run()
    dvs = list(model.design_variables.values())
    # move off the (possibly symmetric) base point so no derivative is trivially zero
    sim[model.design_variables["thickness"]] = np.array([5.0, -3.0, 2.0])
    sim[model.design_variables["camber"]] = np.array([1.0, 2.0, -1.0])
    sim.run()

    totals = sim.compute_totals()  # objective + constraint w.r.t. the declared design variables
    model.recorder.start()  # csdl's finite-difference helper builds graph nodes
    try:
        fd = sim.compute_totals(use_finite_difference=True, finite_difference_step_size=1e-4)
    finally:
        model.recorder.stop()
    for of in (model.CD, model.CL):
        scale = max(np.abs(fd[of, dv]).max() for dv in dvs)  # dominant sensitivity of this output
        for dv in dvs:
            # central-difference noise is ~1e-4 of the dominant sensitivity at this step size
            np.testing.assert_allclose(totals[of, dv], fd[of, dv], rtol=2e-3, atol=2e-3 * scale)
    # the derivatives must not all be zero (that would also "agree")
    assert max(np.abs(totals[model.CD, dv]).max() for dv in dvs) > 1e-6


def test_thickness_and_camber_change_the_geometry_the_flow_sees(built):
    model, fake, surface, _ = built
    sim = model.make_simulator(additional_outputs=[model.x_surf])
    sim[model.design_variables["thickness"]] = np.array([20.0, 20.0, 20.0])
    sim.run()
    thick = np.asarray(sim[model.x_surf])
    assert np.abs(thick[:, 2]).max() > np.abs(surface[:, 2]).max() + 1e-3

    sim[model.design_variables["thickness"]] = np.zeros(3)
    sim[model.design_variables["camber"]] = np.array([5.0, 5.0, 5.0])
    sim.run()
    cambered = np.asarray(sim[model.x_surf])
    assert (cambered[:, 2] - surface[:, 2]).mean() > 1e-3


@pytest.mark.parametrize("algorithm", ["OpenSQP", "PySLSQP"])
@pytest.mark.slow
def test_optimize_runs_modopt_algorithms(built, tmp_path, monkeypatch, algorithm):
    pytest.importorskip("modopt")
    pytest.importorskip("pyslsqp")
    monkeypatch.chdir(tmp_path)  # modOpt writes its output directory into the cwd
    model, *_ = built
    sim = model.make_simulator()
    optimizer = model.optimize(sim, algorithm=algorithm, maxiter=3, problem_name="smoke")
    assert optimizer is not None
    assert (tmp_path / "smoke_outputs").exists()
    assert np.isfinite(np.ravel(sim[model.CD])[0])

    # one history entry per gradient evaluation, with CD, CL and the design variables
    assert len(model.history) >= 1
    assert {"CD", "CL", "time", "thickness", "camber", "angle_of_attack"} <= set(model.history[0])
    assert all(np.isfinite(e["CD"]) and np.isfinite(e["CL"]) for e in model.history)

    # the simulator was left at the optimizer's final design
    scaled = np.asarray(optimizer.results["x"])
    descaled = scaled / optimizer.problem.x_scaler - optimizer.problem.x_adder
    held = np.concatenate([np.ravel(sim[v]) for v in (
        model.design_variables["thickness"], model.design_variables["camber"], model.design_variables["angle_of_attack"])])
    assert sorted(np.round(held, 9)) == sorted(np.round(descaled, 9))


def test_unknown_optimizer_is_rejected(built):
    model, *_ = built
    with pytest.raises(ValueError, match="algorithm"):
        model.optimize(algorithm="SNOPT")


def test_jax_warper_local_point_map_recovers_the_decomposed_ordering(built):
    model, fake, *_ = built
    if model.jax_warp is None:
        pytest.skip("only for warper='jax'")
    np.testing.assert_array_equal(model.local_to_global, fake.local_to_global_truth)
