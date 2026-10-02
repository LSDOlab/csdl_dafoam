"""Where does the time in ``compute_totals`` go? (real DAFoam, rank 0 profiled)

    mpirun -np 4 python scripts/vm/profile_totals.py

Profiles the first call (derivative-graph compilation + one-time DAFoam setup + adjoints) and the second call
(adjoints only) separately and prints, for each, the functions with the largest *own* time, then the time spent
in the DAFoam methods the CSDL operations call, so JAX/CSDL overhead can be told apart from DAFoam work.
"""
import cProfile
import pstats
import tempfile
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from csdl_dafoam import cases
from csdl_dafoam.airfoil import FlightConditions, build_airfoil_model
from csdl_dafoam.options import options_for_case

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os

case_name = os.environ.get("CASE", "naca0012_euler")
root = Path(comm.bcast(tempfile.mkdtemp(prefix="prof_") if rank == 0 else None, root=0))
case = cases.prepare_case(case_name, root / "case", comm=comm)

model = build_airfoil_model(
    case_dir=case,
    da_options=options_for_case(case_name),
    geometry_file=cases.geometry_path(),
    cache_dir=root / "cache",
    flight=FlightConditions(),
    cl_target=0.5,
    comm=comm,
)
sim = model.make_simulator()
sim.run()

# DAFoam methods the CSDL operations call (names as they appear on the PYDAFOAM / solver objects)
DAFOAM_METHODS = (
    "calcdRdWT", "createMLRKSPMatrixFree", "solveLinearEqn", "runFPAdj", "runColoring", "calcJacTVecProduct",
    "initializedRdWTMatrixFree", "setStates", "getStates", "evalFunctions", "set_solver_input",
    "writeAdjointFields", "renameSolution", "calcPrimalResidualStatistics", "solvePrimal", "__call__",
)


def report(profile, title):
    if rank != 0:
        return
    stats = pstats.Stats(profile)
    print(f"\n=========== {title}: {stats.total_tt:.1f} s total", flush=True)
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:14]
    print("  largest own time:")
    for (filename, line, name), (cc, nc, tt, ct, callers) in rows:
        short = filename.split("/")[-1] if filename.startswith("/") else filename
        print(f"    {tt:8.2f} s own  {ct:8.2f} s cum  {nc:6d} calls  {name} ({short}:{line})")
    print("  DAFoam methods called by the CSDL operations (cumulative time):")
    found = []
    for (filename, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        if any(m in name for m in DAFOAM_METHODS) and ("PYDAFOAM" in name or "pyDAFoam" in filename or "DASolvers" in name or "dafoam" in filename or "method" in name):
            found.append((ct, nc, name, filename.split("/")[-1], line))
    for ct, nc, name, filename, line in sorted(found, reverse=True)[:12]:
        print(f"    {ct:8.2f} s cum  {nc:6d} calls  {name} ({filename}:{line})")


for label in ("1st compute_totals (compile + one-time setup + adjoints)", "2nd compute_totals (adjoints only)"):
    comm.Barrier()
    profile = cProfile.Profile()
    start = time.time()
    profile.enable()
    sim.compute_totals()
    profile.disable()
    comm.Barrier()
    if rank == 0:
        print(f"\n{label}: wall {time.time() - start:.1f} s", flush=True)
    report(profile, label)
    sim[model.design_variables["thickness"]] = sim[model.design_variables["thickness"]] + 1.0
    sim.run()
