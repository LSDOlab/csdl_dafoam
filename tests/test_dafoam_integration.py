"""Runs against a real DAFoam/OpenFOAM install (skipped when it is not importable).

    pytest -m dafoam                                   # one rank, the 336-cell test mesh: seconds
    CSDL_DAFOAM_TEST_CASE=naca0012_euler pytest -m dafoam   # the 4,032-cell mesh of the examples: about a minute
    mpirun -np 4 python -m pytest -m dafoam -p no:cacheprovider   # four ranks

Run against DAFoam v5.1.1 / OpenFOAM v2506 built from source on ARM64 Linux (scripts/vm/). Bounds are
loose on purpose: they catch a broken interface (NaNs, wrong signs, wrong units), not solver accuracy.
"""
import os

import numpy as np
import pytest

#: ``naca0012_coarse`` (336 cells) keeps the test fast; its solution is crude (see docs/status.md), which is fine for
#: checking the interface, so the bounds below are loose
CASE = os.environ.get("CSDL_DAFOAM_TEST_CASE", "naca0012_coarse")

pytestmark = pytest.mark.dafoam

pytest.importorskip("dafoam")
pytest.importorskip("lsdo_geo")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from mpi4py import MPI

    from csdl_dafoam import cases
    from csdl_dafoam.airfoil import FlightConditions, build_airfoil_model
    from csdl_dafoam.options import options_for_case

    comm = MPI.COMM_WORLD
    root = tmp_path_factory.mktemp("dafoam", numbered=False) if comm.Get_rank() == 0 else None
    root = comm.bcast(root, root=0)
    case = root / "case"
    if comm.Get_rank() == 0:
        cases.copy_case(CASE, case, nprocs=comm.Get_size())
    comm.Barrier()

    return build_airfoil_model(
        case_dir=case,
        da_options=options_for_case(CASE),
        geometry_file=cases.geometry_path(),
        cache_dir=root / "cache",
        flight=FlightConditions(),
        comm=comm,
    )


def test_analysis_gives_sane_coefficients(model):
    result = model.analyze()
    # NACA 0012 at 3 deg, Mach ~0.69
    assert 0.1 < result["CL"] < 0.9
    assert 0.0 < result["CD"] < 0.1


def test_adjoint_derivatives_match_finite_differences(model):
    """Step 0.1 (percent thickness/camber, degrees of alpha): a step-size study (scripts/vm/fd_study.py) showed the
    finite differences are noise at 1e-3 or below (the flow solve is converged to 1e-8, the response to a 1e-3 step is
    smaller than that) and drift at 1 (nonlinearity); between 0.01 and 0.1 they agree with the adjoint to 0.2 to 4%."""
    sim = model.make_simulator()
    sim.run()
    totals = sim.compute_totals()
    model.recorder.start()
    try:
        fd = sim.compute_totals(use_finite_difference=True, finite_difference_step_size=0.1)
    finally:
        model.recorder.stop()
    for key in totals:
        a, b = np.ravel(totals[key]), np.ravel(fd[key])
        scale = np.abs(b).max()
        # a wiring error is O(1) of the dominant sensitivity; measured disagreement here is <= 4.3%
        np.testing.assert_allclose(a, b, atol=0.06 * scale)
