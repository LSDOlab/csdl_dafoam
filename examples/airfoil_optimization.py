"""2D airfoil optimization: minimize CD at fixed CL by changing thickness, camber and alpha.

Default is the Euler-mode case (coarse mesh, Re_c ~ 1.5e11, so the boundary layer is
negligible and drag is essentially wave drag plus numerical dissipation):

    mpirun -np 4 python examples/airfoil_optimization.py

The original RANS tutorial setup (Re_c ~ 1.5e7, wall function) is the same script on the
other bundled case:

    mpirun -np 4 python examples/airfoil_optimization.py --case naca0012

The stack: lsdo_geo (FFD thickness/camber parameterization of the STEP geometry), IDWarp-JAX
(volume mesh deformation), DAFoam (flow solve and adjoint), CSDL (derivative assembly) and
modOpt's OpenSQP (or `--optimizer PySLSQP`).

    mpirun -np 4 python examples/airfoil_optimization.py --optimizer PySLSQP
    mpirun -np 4 python examples/airfoil_optimization.py --warper idwarp    # Fortran IDWarp instead

Outputs (rank 0, working directory): optimization.png (C_p contours and surface C_p, initial vs optimized), history.png
(CD, CL and convergence measures per iteration), history.npz, and movie.gif / movie.mp4 (the C_p field at every iteration,
drawn from the solutions DAFoam keeps; the .mp4 needs ffmpeg).

Design variables: 3 thickness + 3 camber FFD degrees of freedom and the angle of attack.
Constraint: CL = 0.5. Gradients: DAFoam adjoint.
"""
import numpy as np
from _common import build, comm_world, make_parser, read_solution_if_available


def main():
    parser = make_parser(__doc__.splitlines()[0], default_case="naca0012_euler")
    parser.add_argument(
        "--optimizer",
        default="OpenSQP",
        choices=["OpenSQP", "PySLSQP"],
        help="modOpt algorithm (default: %(default)s)",
    )
    parser.add_argument("--maxiter", type=int, default=20, help="major iterations")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="optimality and feasibility tolerance")
    parser.add_argument("--cl-target", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true", help="skip the figures and the movie")
    parser.add_argument("--no-movie", action="store_true", help="skip the movie of the C_p field over the iterations")
    args = parser.parse_args()

    comm = comm_world()
    rank = comm.Get_rank()
    model = build(args, comm, optimize_aoa=True, cl_target=args.cl_target)

    sim = model.make_simulator(additional_outputs=[model.x_surf])
    before_alpha = float(np.ravel(sim[model.design_variables['angle_of_attack']])[0])
    before = model.analyze(sim)
    x_before = np.array(sim[model.x_surf])
    case_dir = model.dafoam_instance.run_directory
    solution_before = None if args.no_plot else read_solution_if_available(case_dir, comm)

    model.optimize(sim, algorithm=args.optimizer, maxiter=args.maxiter, tolerance=args.tolerance)
    after = {"CL": float(np.ravel(sim[model.CL])[0]), "CD": float(np.ravel(sim[model.CD])[0])}
    x_after = np.array(sim[model.x_surf])
    dvs = {name: np.ravel(sim[var]) for name, var in model.design_variables.items()}

    if rank == 0:
        print(f"\ncase={args.case}  warper={args.warper}  optimizer={args.optimizer}  CL target={args.cl_target}")
        print(f"  initial: CL = {before['CL']:.5f}  CD = {before['CD'] * 1e4:.1f} counts")
        print(f"  final:   CL = {after['CL']:.5f}  CD = {after['CD'] * 1e4:.1f} counts")
        print(f"  CD change: {100 * (after['CD'] / before['CD'] - 1):+.1f} %")
        for name, value in dvs.items():
            print(f"  {name}: {np.round(value, 4)}")
        np.savez("optimized_airfoil.npz", x_initial=x_before, x_optimized=x_after, **dvs)
        print("saved optimized_airfoil.npz; modOpt output is in ASO_2DAF_rank0_outputs/")

    solution_after = None if args.no_plot else read_solution_if_available(case_dir, comm)
    if rank == 0 and solution_before is not None and solution_after is not None:
        import matplotlib

        matplotlib.use("Agg")
        from csdl_dafoam import viz

        fig = viz.plot_airfoil_comparison(
            [solution_before, solution_after],
            labels=(f"initial (alpha {before_alpha:.2f} deg)", f"optimized (alpha {dvs['angle_of_attack'][0]:.2f} deg)"),
            title=f"{args.case}: minimize CD at CL = {args.cl_target}",
        )
        fig.savefig("optimization.png", dpi=150)
        print("saved optimization.png (C_p contours, surface Cp, shapes: initial vs optimized)")

    if rank == 0 and not args.no_plot and model.history:
        import matplotlib

        matplotlib.use("Agg")
        from csdl_dafoam import viz

        history = model.history
        np.savez("history.npz", CD=[e["CD"] for e in history], CL=[e["CL"] for e in history], time=[e["time"] for e in history],
                 **{name: np.array([e[name] for e in history]) for name in model.design_variables})
        table = None
        try:
            table = viz.read_optimization_history(f"ASO_2DAF_rank{rank}_outputs")
            viz.plot_optimization_history(table, iterations=history, cl_target=args.cl_target).savefig("history.png", dpi=150)
            print("saved history.png (CD, CL, optimality and feasibility per iteration) and history.npz")
        except (FileNotFoundError, ValueError) as error:
            print(f"history.png not written: {error}")
        if not args.no_movie and all(e["time"] is not None for e in history):
            import shutil

            outputs = ["movie.gif"] + (["movie.mp4"] if shutil.which("ffmpeg") else [])
            viz.make_movie(case_dir, outputs, history, table=table, cl_target=args.cl_target)
            print(f"saved {' and '.join(outputs)} (C_p field at every iteration)")


if __name__ == "__main__":
    main()
