"""Shared set-up for the examples (not part of the installed package)."""
import argparse
import logging
import os
from pathlib import Path

from csdl_dafoam import cases
from csdl_dafoam.airfoil import FlightConditions, build_airfoil_model
from csdl_dafoam.options import options_for_case


def make_parser(description, default_case):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--case",
        default=default_case,
        choices=sorted(cases.list_cases()),
        help="bundled OpenFOAM case (default: %(default)s)",
    )
    parser.add_argument("--workdir", default="run", help="directory for the case copy, caches and outputs")
    parser.add_argument("--aoa", type=float, default=3.0, help="angle of attack [deg]")
    parser.add_argument(
        "--adjoint-tol",
        type=float,
        default=1e-6,
        help="relative tolerance of the adjoint linear solve; 1e-3 is about 30%% faster per adjoint at a derivative error of ~7e-4 "
        "(see docs/examples.md) (default: %(default)s)",
    )
    parser.add_argument(
        "--warper",
        default="jax",
        choices=["jax", "idwarp"],
        help="volume-mesh deformation: LSDO Lab's IDWarp-JAX, or the Fortran IDWarp (default: %(default)s)",
    )
    return parser


def comm_world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def build(args, comm, optimize_aoa, cl_target):
    """Copy the case, enter the work directory and assemble the CSDL model."""
    logging.basicConfig(level=logging.INFO if comm.Get_rank() == 0 else logging.WARNING, format="%(message)s")

    workdir = Path(args.workdir).resolve()
    case_dir = cases.prepare_case(args.case, workdir / "case", comm=comm)
    os.chdir(workdir)  # modOpt and the CSDL simulator write their output next to the case

    return build_airfoil_model(
        case_dir=case_dir,
        da_options=options_for_case(args.case, adjoint_rel_tol=args.adjoint_tol),
        geometry_file=cases.geometry_path(),
        cache_dir=workdir / "cache",
        flight=FlightConditions(angle_of_attack_deg=args.aoa, optimize_angle_of_attack=optimize_aoa),
        cl_target=cl_target,
        comm=comm,
        warper=args.warper,
    )


def read_solution_if_available(case_dir, comm):
    """Rank 0 reads the latest flow solution of the case (others get ``None``); ``None`` too if there is none to read."""
    from csdl_dafoam import viz

    comm.Barrier()  # all ranks have finished writing
    if comm.Get_rank() != 0:
        return None
    try:
        return viz.read_solution(case_dir)
    except FileNotFoundError:
        print("no flow solution found in the case directory; skipping plots")
        return None
