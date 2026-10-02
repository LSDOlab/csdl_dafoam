"""2D airfoil analysis: one DAFoam flow solve of the bundled NACA 0012.

Run on 4 ranks (the bundled case is decomposed to match the number of ranks):

    mpirun -np 4 python examples/airfoil_analysis.py
    mpirun -np 4 python examples/airfoil_analysis.py --case naca0012 --aoa 2.0
    mpirun -np 4 python examples/airfoil_analysis.py --check-totals

``--case naca0012_euler`` (default) is the coarse-mesh, very-high-Reynolds-number "Euler
mode"; ``--case naca0012`` is the same mesh at Re_c ~ 1.5e7 with a turbulent boundary layer
(wall function). See docs/examples.md for what the numbers mean.
"""
import numpy as np
from _common import build, comm_world, make_parser, read_solution_if_available


def main():
    parser = make_parser(__doc__.splitlines()[0], default_case="naca0012_euler")
    parser.add_argument(
        "--check-totals",
        action="store_true",
        help="also compare adjoint derivatives with finite differences (one extra flow solve per design variable per step)",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the Mach-number and surface-Cp figure")
    args = parser.parse_args()

    comm = comm_world()
    # No lift constraint and fixed angle of attack: we only want to evaluate the flow.
    model = build(args, comm, optimize_aoa=False, cl_target=None)

    sim = model.make_simulator()
    result = model.analyze(sim)
    if comm.Get_rank() == 0:
        print(f"\ncase={args.case}  warper={args.warper}  alpha={args.aoa} deg")
        print(f"  CL = {result['CL']:.5f}")
        print(f"  CD = {result['CD']:.6f}  ({result['CD'] * 1e4:.1f} counts)")

    if not args.no_plot:
        solution = read_solution_if_available(model.dafoam_instance.run_directory, comm)
        if solution is not None:
            import matplotlib

            matplotlib.use("Agg")
            from csdl_dafoam import viz

            fig = viz.plot_airfoil_comparison([solution], labels=[f"alpha = {args.aoa} deg"], title=args.case)
            fig.savefig("analysis.png", dpi=150)
            print("saved analysis.png (C_p contours and surface C_p)")

    if args.check_totals:
        totals = sim.compute_totals()
        model.recorder.start()  # csdl's finite-difference helper builds graph nodes
        try:
            fd = sim.compute_totals(use_finite_difference=True, finite_difference_step_size=0.01)
        finally:
            model.recorder.stop()
        if comm.Get_rank() == 0:
            print("\nadjoint vs finite-difference total derivatives")
            for (of, wrt), adjoint in totals.items():
                diff = np.abs(np.ravel(adjoint) - np.ravel(fd[of, wrt])).max()
                scale = np.abs(np.ravel(fd[of, wrt])).max()
                print(f"  d{of.name}/d{wrt.name}: max|adjoint - FD| = {diff:.3e}  (max|FD| = {scale:.3e})")


if __name__ == "__main__":  # required: lsdo_geo may spawn worker processes that re-import this file
    main()
