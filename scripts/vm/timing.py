"""Wall-clock timing of the pieces of one optimization iteration on the real DAFoam solver.

    mpirun -np 1 python scripts/vm/timing.py
    mpirun -np 4 python scripts/vm/timing.py

Reports (rank 0): model build, a forward solve after a design change of optimizer-step size (warm-started
from the previous solution, which is what every optimization iteration does), and the adjoint total
derivatives, first call (includes JAX compilation of the derivative graph) and second call (pure adjoint).
"""
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
root = Path(comm.bcast(tempfile.mkdtemp(prefix="timing_") if rank == 0 else None, root=0))
case = cases.prepare_case(case_name, root / "case", comm=comm)


def say(text):
    if rank == 0:
        print(text, flush=True)


def timed(label, function):
    comm.Barrier()
    start = time.time()
    value = function()
    comm.Barrier()
    say(f"  {label:<58s} {time.time() - start:8.2f} s")
    return value


# Optional experiments, set through the environment (documented in docs/examples.md, "Speeding it up"):
#   ADJ_TOL      GMRES relative tolerance of the adjoint (DAFoam default here: 1e-6)
#   ADJ_PC_LAG   recompute the adjoint preconditioner every N derivative evaluations (default 1)
#   PRIMAL_TOL   primalMinResTol (default 1e-8)
#   TAG          label for the saved derivatives, totals_<TAG>.npz, to compare settings against each other
da_options = options_for_case(case_name)
if os.environ.get("ADJ_TOL"):
    da_options["adjEqnOption"]["gmresRelTol"] = float(os.environ["ADJ_TOL"])
if os.environ.get("ADJ_PC_LAG"):
    da_options["adjPCLag"] = int(os.environ["ADJ_PC_LAG"])
if os.environ.get("PRIMAL_TOL"):
    da_options["primalMinResTol"] = float(os.environ["PRIMAL_TOL"])
tag = os.environ.get("TAG", "default")

say(f"\n=== {comm.Get_size()} MPI rank(s), {case_name}, tag {tag}: adjoint tol {da_options['adjEqnOption']['gmresRelTol']:g}, "
    f"PC lag {da_options.get('adjPCLag', 'default')}, primal tol {da_options['primalMinResTol']:g} ===")
model = timed(
    "build model (geometry, projection, IDWarp-JAX, DAFoam init, 1st solve)",
    lambda: build_airfoil_model(
        case_dir=case,
        da_options=da_options,
        geometry_file=cases.geometry_path(),
        cache_dir=root / "cache",
        flight=FlightConditions(),
        cl_target=0.5,
        comm=comm,
    ),
)
sim = model.make_simulator()
timed("sim.run() at the base design (state already converged)", sim.run)

# a design change of roughly optimizer-step size
sim[model.design_variables["thickness"]] = np.array([5.0, 5.0, 5.0])
sim[model.design_variables["camber"]] = np.array([2.0, 2.0, 2.0])
timed("sim.run() after a design change (warm-started solve)", sim.run)
timed("sim.run() again, unchanged (no re-solve needed?)", sim.run)

timed("compute_totals(), 1st call (derivative-graph compile + adjoints)", sim.compute_totals)
sim[model.design_variables["thickness"]] = np.array([6.0, 6.0, 6.0])
timed("sim.run() after a second design change", sim.run)
totals = timed("compute_totals(), 2nd call (adjoints, no compile)", sim.compute_totals)
if rank == 0:
    np.savez(f"totals_{tag}_np{comm.Get_size()}.npz", **{f"{of.name}__{dv.name}": np.ravel(value) for (of, dv), value in totals.items()})

say(f"\nCL = {float(np.ravel(sim[model.CL])[0]):.5f}  CD = {float(np.ravel(sim[model.CD])[0]):.6f}")
