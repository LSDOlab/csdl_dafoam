"""Adjoint vs finite-difference step-size study on the real DAFoam solver.

Run inside an environment where DAFoam works (the VM; ``. ~/dafoam/loadDAFoam.sh``):

    python scripts/vm/fd_study.py [--case naca0012_euler] [--steps 1e-3,1e-2,1e-1,1]

Why a study and not one comparison: a CFD solve is converged only to a tolerance, so its outputs carry noise of
roughly that size. A finite-difference step so small that the function changes by less than that noise
gives garbage, and one so large that the function is nonlinear gives truncation error. The adjoint is
trustworthy if the finite differences converge towards it between those two failure modes.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from csdl_dafoam import cases
from csdl_dafoam.airfoil import FlightConditions, build_airfoil_model
from csdl_dafoam.options import options_for_case


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="naca0012_euler", choices=sorted(cases.list_cases()))
    parser.add_argument("--steps", default="1e-3,1e-2,1e-1,1.0", help="comma-separated finite-difference steps")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    root = Path(args.workdir or tempfile.mkdtemp(prefix="fdstudy_"))
    case = cases.prepare_case(args.case, root / "case", comm=comm)

    t0 = time.time()
    model = build_airfoil_model(
        case_dir=case,
        da_options=options_for_case(args.case),
        geometry_file=cases.geometry_path(),
        cache_dir=root / "cache",
        flight=FlightConditions(),
        cl_target=0.5,
        comm=comm,
    )
    print(f"model built in {time.time() - t0:.1f} s", flush=True)

    sim = model.make_simulator()
    t0 = time.time()
    result = model.analyze(sim)
    print(f"analysis in {time.time() - t0:.1f} s:  CL = {result['CL']:.6f}   CD = {result['CD']:.6f}", flush=True)

    t0 = time.time()
    totals = sim.compute_totals()
    print(f"adjoint totals in {time.time() - t0:.1f} s", flush=True)

    dvs = list(model.design_variables.items())
    outputs = {"CD": model.CD, "CL": model.CL}
    adjoint = {(o, n): np.ravel(totals[var_o, var_d]) for o, var_o in outputs.items() for n, var_d in dvs}

    fd_by_step = {}
    for step in [float(s) for s in args.steps.split(",")]:
        t0 = time.time()
        model.recorder.start()  # csdl's finite-difference helper builds graph nodes
        try:
            fd = sim.compute_totals(use_finite_difference=True, finite_difference_step_size=step)
        finally:
            model.recorder.stop()
        fd_by_step[step] = {(o, n): np.ravel(fd[var_o, var_d]) for o, var_o in outputs.items() for n, var_d in dvs}
        print(f"finite differences, step {step:g}: {time.time() - t0:.1f} s", flush=True)

    np.set_printoptions(precision=4, linewidth=160)
    for (o, n), adj in adjoint.items():
        print(f"\nd{o}/d{n}   adjoint: {adj}")
        for step, fd in fd_by_step.items():
            scale = np.abs(adj).max() or 1.0
            err = np.abs(fd[o, n] - adj).max() / scale
            print(f"    FD step {step:<8g}: {fd[o, n]}   max|FD-adj|/max|adj| = {err:.3f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
