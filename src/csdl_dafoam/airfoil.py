"""High-level assembly of a 2D airfoil CSDL model around DAFoam.

:func:`build_airfoil_model` wires together the pieces of this package, in this order:

1. DAFoam instance for an OpenFOAM case (:func:`~csdl_dafoam.solver.instantiate_dafoam`), and the
   mesh warper: IDWarp-JAX (default) or the Fortran IDWarp attached to the DAFoam instance
2. ``lsdo_geo`` geometry from a STEP file, projected onto the CFD surface mesh (cached)
3. thickness/camber FFD design variables (:func:`~csdl_dafoam.geometry.setup_airfoil_geometry`)
4. freestream conditions from the standard-atmosphere model
5. MPI region: mesh warp -> DAFoam flow solve -> CL and CD

It returns an :class:`AirfoilModel` holding the recorded CSDL graph, ready for a CSDL
simulator and an optimizer. Run it under ``mpirun -np N`` with ``numberOfSubdomains = N`` in
the case's ``decomposeParDict`` (see :func:`csdl_dafoam.cases.copy_case`).
"""
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from csdl_dafoam._optional import require
from csdl_dafoam._run_directory import working_directory
from csdl_dafoam.atmosphere import compute_ambient_conditions_group
from csdl_dafoam.geometry import (
    load_geometry,
    read_simple_pickle,
    setup_airfoil_geometry,
    write_simple_pickle,
)
from csdl_dafoam.inputs import compute_dafoam_input_variables
from csdl_dafoam.jax_warp import JaxMeshWarper
from csdl_dafoam.mesh_warp import DAFoamMeshWarper
from csdl_dafoam.mpi_utils import gather_array_to_rank0, hash_array_tol
from csdl_dafoam.solver import DAFoamFunctions, DAFoamSolver, instantiate_dafoam

logger = logging.getLogger(__name__)

__all__ = ["FlightConditions", "AirfoilModel", "build_airfoil_model"]


@dataclass
class FlightConditions:
    """Freestream conditions. ``airspeed_m_s`` should match ``U0`` in the DAFoam options."""

    airspeed_m_s: float = 238.0
    angle_of_attack_deg: float = 3.0
    altitude_m: float = 0.0
    #: make the angle of attack a design variable
    optimize_angle_of_attack: bool = True
    angle_of_attack_bounds_deg: tuple = (0.0, 10.0)


@dataclass
class AirfoilModel:
    """The recorded CSDL model returned by :func:`build_airfoil_model`."""

    recorder: csdl.Recorder
    CL: csdl.Variable
    CD: csdl.Variable
    design_variables: dict
    dafoam_instance: object
    geometry: object
    comm: object
    rank: int
    flight_conditions: csdl.VariableGroup
    ambient_conditions: csdl.VariableGroup
    x_surf: csdl.Variable
    timings: dict = field(default_factory=dict)
    #: ``JaxIDWarp`` and the local-to-global point map (``warper="jax"`` only, else ``None``)
    jax_warp: object = None
    local_to_global: object = None
    #: per optimizer iteration (one per gradient evaluation) from :meth:`optimize`: dicts with ``CD``, ``CL``, the design variables
    history: list = field(default_factory=list)

    def make_simulator(self, **kwargs):
        """Create the (JAX) CSDL simulator for this model.

        CL and CD are always readable from it (``sim[model.CL]``), even when CL is not a
        constraint. Pass ``additional_outputs=[...]`` to expose other variables too.
        """
        options = dict(gpu=False, save_on_update=False, filename="ASO_2DAF_sim", output_saved=False)
        options.update(kwargs)
        extra = list(options.pop("additional_outputs", None) or [])
        extra += [v for v in (self.CL, self.CD) if not any(v is e for e in extra)]
        return csdl.experimental.JaxSimulator(self.recorder, additional_outputs=extra, **options)

    def analyze(self, simulator=None):
        """Run the model once at the current design values and return ``{"CL":..., "CD":...}``."""
        simulator = simulator or self.make_simulator()
        simulator.run()
        return {"CL": float(np.ravel(simulator[self.CL])[0]), "CD": float(np.ravel(simulator[self.CD])[0])}

    def _record_history(self, simulator):
        """Make ``simulator.compute_totals`` also append the current CD, CL and design variables to a list (returned).

        One entry per gradient evaluation (modOpt's table counts them in its ``ngev`` column): ``CD``, ``CL``, the design
        variables, and ``time``, the name of the solution directory DAFoam kept for that design (``0.0003``, ...).
        """
        history = []
        busy = []  # one of the wrapped methods may call the other: record once per outermost call

        def wrap(method):
            def recording(*args, **kwargs):
                if busy:
                    return method(*args, **kwargs)
                entry = {"CD": float(np.ravel(simulator[self.CD])[0]), "CL": float(np.ravel(simulator[self.CL])[0])}
                for name, variable in self.design_variables.items():
                    entry[name] = np.array(np.ravel(simulator[variable]), dtype=float)
                busy.append(True)
                try:
                    result = method(*args, **kwargs)
                finally:
                    busy.pop()
                # the adjoint has just renamed the converged solution to counter * 1e-4: the newest such directory is this design's
                from csdl_dafoam import viz

                try:
                    kept = [t for t in viz.solution_times(self.dafoam_instance.run_directory) if float(t) < 1.0]
                except OSError:
                    kept = []
                entry["time"] = kept[-1] if kept else None
                history.append(entry)
                return result

            return recording

        # modOpt asks for ``compute_optimization_derivatives``; ``compute_totals`` is the general entry point
        for name in ("compute_optimization_derivatives", "compute_totals"):
            if hasattr(simulator, name):
                setattr(simulator, name, wrap(getattr(simulator, name)))
        return history

    def optimize(self, simulator=None, algorithm="OpenSQP", maxiter=20, tolerance=1e-5, problem_name=None, **solver_options):
        """Minimize CD (subject to the CL constraint, if any) with a modOpt algorithm.

        Parameters
        ----------
        algorithm : {"OpenSQP", "PySLSQP"}
            modOpt optimizer. ``OpenSQP`` is modOpt's SQP with BFGS Hessian approximation and an
            elastic-mode merit function; ``PySLSQP`` wraps the classic SLSQP Fortran.
        maxiter : int
            Major iterations.
        tolerance : float
            Optimality and feasibility tolerance (``opt_tol``/``feas_tol`` for OpenSQP, ``acc`` for
            PySLSQP).
        **solver_options
            Passed through to the algorithm and overriding the above (for example
            ``verbosity=1`` for OpenSQP, or ``ls_maxiter``).

        Afterwards the simulator holds the optimizer's final design and has been re-run there, so
        ``simulator[model.CD]`` etc. are the optimized values. Returns the modOpt optimizer object
        (``optimizer.results`` has the iteration/convergence summary).
        """
        import modopt

        algorithms = {
            "OpenSQP": ("OpenSQP", {"maxiter": maxiter, "opt_tol": tolerance, "feas_tol": tolerance}),
            "PySLSQP": ("PySLSQP", {"maxiter": maxiter, "acc": tolerance, "visualize": False}),
        }
        if algorithm not in algorithms:
            raise ValueError(f"algorithm must be one of {sorted(algorithms)}; got {algorithm!r}")
        class_name, defaults = algorithms[algorithm]
        if not hasattr(modopt, class_name):
            raise ImportError(
                f"this modOpt has no {class_name}; install LSDOlab's modOpt from git "
                "(requirements-lab.txt), not PyPI's unrelated 'modopt'"
            )

        simulator = simulator or self.make_simulator()
        self.history = self._record_history(simulator)
        problem = modopt.CSDLAlphaProblem(problem_name=problem_name or f"ASO_2DAF_rank{self.rank}", simulator=simulator)
        options = {**defaults, **solver_options}
        if algorithm == "PySLSQP":  # an external-library wrapper: options go in a dictionary
            optimizer = modopt.PySLSQP(problem, solver_options=options)
        else:  # OpenSQP declares its options directly
            optimizer = modopt.OpenSQP(problem, **options)
        optimizer.solve()

        # modOpt works with scaled design variables; put the final design into the simulator
        x_optimal = np.asarray(optimizer.results["x"])
        simulator.update_design_variables(x_optimal / problem.x_scaler - problem.x_adder)
        simulator.run()
        return optimizer


@contextmanager
def _timed(name, rank, timings, enabled=True):
    if not enabled:
        yield
        return
    logger.info("Rank %d: %s...", rank, name)
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info("Rank %d: %s elapsed time: %.3f s", rank, name, elapsed)
    timings[name] = elapsed


def _rank0_then_share(comm, rank, path, compute, read):
    """Rank 0 reads ``path`` if present, else computes and writes it; the others then read it."""
    if rank == 0:
        value = compute(path)
    if comm is not None:
        comm.Barrier()
    if rank != 0:
        value = read(path)
    return value


def build_airfoil_model(
    case_dir,
    da_options,
    geometry_file,
    mesh_options=None,
    cache_dir=None,
    flight=None,
    num_ffd_coefficients_chordwise=5,
    thickness_bounds_percent=(-100.0, 100.0),
    camber_bounds_percent=(-50.0, 50.0),
    cl_target=0.5,
    comm=None,
    check_mesh=True,
    on_adjoint_failure="warn",
    projection_options=None,
    timing=True,
    warper="jax",
    warper_options=None,
):
    """Assemble the 2D airfoil CSDL model. See the module docstring for the pipeline.

    Parameters
    ----------
    case_dir : path-like
        OpenFOAM case directory (copy one with :func:`csdl_dafoam.cases.copy_case`).
    da_options : dict
        DAFoam options, e.g. from :func:`csdl_dafoam.options.naca0012_options`.
    geometry_file : path-like
        STEP file of the airfoil (e.g. :func:`csdl_dafoam.cases.geometry_path`).
    mesh_options : dict, optional
        Fortran-IDWarp options (``warper="idwarp"`` only); defaults to
        :func:`csdl_dafoam.options.idwarp_options` for ``case_dir``.
    cache_dir : path-like, optional
        Where geometry/projection caches are written (``lsdo_geo`` also drops its own
        ``stored_files`` here). Defaults to ``<case_dir>/../csdl_dafoam_cache``.
    flight : FlightConditions, optional
    num_ffd_coefficients_chordwise : int
        FFD control points along the chord; ``num - 2`` thickness and ``num - 2`` camber
        design variables are created.
    thickness_bounds_percent, camber_bounds_percent : (float, float)
        Design-variable bounds (percent).
    cl_target : float or None
        Equality constraint ``CL == cl_target``; ``None`` for an unconstrained CD objective.
    comm : mpi4py communicator, optional
        Defaults to ``MPI.COMM_WORLD``.
    check_mesh, on_adjoint_failure
        Passed to :class:`~csdl_dafoam.solver.DAFoamSolver`.
    projection_options : dict, optional
        Overrides for ``Geometry.project`` (tolerances and grid-search settings).
    warper : {"jax", "idwarp"}
        Volume-mesh deformation. ``"jax"`` (default) uses LSDO Lab's IDWarp-JAX through
        :class:`~csdl_dafoam.jax_warp.JaxMeshWarper`: no compiled IDWarp needed, the whole
        moving surface is global, so no surface gathering. ``"idwarp"`` uses the Fortran IDWarp
        attached to the DAFoam instance.
    warper_options : dict, optional
        Options of :class:`~csdl_dafoam.jax_warp.JaxIDWarp` (``device``, ``LdefFact``, ``err_tol``, ...).
    """
    from csdl_dafoam.options import idwarp_options

    if comm is None:
        comm = require("mpi4py.MPI").COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    flight = flight or FlightConditions()
    case_dir = Path(case_dir).resolve()
    cache_dir = Path(cache_dir) if cache_dir else case_dir.parent / "csdl_dafoam_cache"
    mesh_options = mesh_options if mesh_options is not None else idwarp_options(case_dir)
    timings = {}

    if warper not in ("jax", "idwarp"):
        raise ValueError(f"warper must be 'jax' or 'idwarp'; got {warper!r}")

    # ---- mesh warper, DAFoam instance and the design-surface mesh ----------------------------
    jax_warp = local_to_global = None
    if warper == "jax":
        from csdl_dafoam.jax_warp import JaxIDWarp, match_local_points

        with _timed("preparing IDWarp-JAX", rank, timings, timing):
            jax_warp = JaxIDWarp(case_dir, da_options["designSurfaces"], **(warper_options or {}))
        dafoam_instance = instantiate_dafoam(da_options, comm, case_dir, mesh_options=None)
        local_to_global = match_local_points(jax_warp.points, dafoam_instance.xv0)
        # the moving surface is global: every rank has all of it, in IDWarp-JAX's point order
        x_surf_initial, x_surf_indices = jax_warp.wall_points, None
    else:
        dafoam_instance = instantiate_dafoam(da_options, comm, case_dir, mesh_options)
        x_surf_mpi = dafoam_instance.getSurfaceCoordinates(dafoam_instance.designSurfacesGroup)

        # Gather the surface mesh on rank 0: the projection and geometry evaluation need all
        # points at once, and ranks with no surface elements would otherwise break them.
        x_surf_initial, _, x_surf_indices = gather_array_to_rank0(x_surf_mpi, comm)

    x_surf_hash = hash_array_tol(x_surf_initial) if (rank == 0 or warper == "jax") else None
    if warper != "jax":
        x_surf_hash = comm.bcast(x_surf_hash, root=0)
    projection_file = cache_dir / f"projected_surface_mesh_{x_surf_hash}.pickle"
    cache_dir.mkdir(parents=True, exist_ok=True)

    recorder = csdl.Recorder(inline=True, debug=True)
    recorder.start()

    # ---- Geometry import and projection of the CFD surface mesh -------------------------
    with _timed("loading geometry", rank, timings, timing):
        geometry = load_geometry(geometry_file, cache_dir, comm)

    projection_kwargs = dict(
        grid_search_density_parameter=1,
        projection_tolerance=1e-3,
        grid_search_density_cutoff=50,
        force_reprojection=False,
        plot=False,
        # Serial by default: a 2D surface mesh projects in about a second, and lsdo_function_spaces'
        # process pool relies on inherited globals, which fails where the start method is
        # "spawn" (macOS, Windows). Pass num_workers=N through projection_options for big meshes.
        num_workers=1,
    )
    projection_kwargs.update(projection_options or {})

    def compute_projection(path):
        if path.is_file():
            logger.info("Found surface mesh projection pickle: %s", path)
            return read_simple_pickle(path)
        with _timed("projecting on surface mesh", rank, timings, timing), working_directory(cache_dir):
            projected = geometry.project(x_surf_initial, **projection_kwargs)
        write_simple_pickle(projected, path)
        return projected

    projected_surf_mesh = _rank0_then_share(comm, rank, projection_file, compute_projection, read_simple_pickle)

    # ---- Design variables --------------------------------------------------------------
    num_dof = num_ffd_coefficients_chordwise - 2
    percent_change_in_thickness_dof = csdl.Variable(shape=(num_dof,), value=np.zeros(num_dof), name="thickness")
    normalized_percent_camber_change_dof = csdl.Variable(shape=(num_dof,), value=np.zeros(num_dof), name="camber")
    percent_change_in_thickness_dof.set_as_design_variable(
        lower=thickness_bounds_percent[0], upper=thickness_bounds_percent[1], scaler=0.01
    )
    normalized_percent_camber_change_dof.set_as_design_variable(
        lower=camber_bounds_percent[0], upper=camber_bounds_percent[1], scaler=0.01
    )

    # lsdo_geo caches to the working directory; do the (cheap) setup one rank at a time to
    # avoid races on that cache.
    with _timed("setting up geometry", rank, timings, timing):
        for r in range(comm_size):
            comm.Barrier()
            if rank == r:
                with working_directory(cache_dir):
                    geometry = setup_airfoil_geometry(
                        geometry,
                        num_ffd_coefficients_chordwise,
                        percent_change_in_thickness_dof,
                        normalized_percent_camber_change_dof,
                    )
            comm.Barrier()

    with _timed("evaluating geometry", rank, timings, timing), working_directory(cache_dir):
        x_surf_full = geometry.evaluate(projected_surf_mesh, plot=False)
    comm.Barrier()

    # ---- Flight and atmospheric conditions ------------------------------------------------
    flight_conditions_group = csdl.VariableGroup()
    flight_conditions_group.airspeed_m_s = csdl.Variable(value=flight.airspeed_m_s, name="air speed")
    flight_conditions_group.angle_of_attack = csdl.Variable(value=flight.angle_of_attack_deg, name="angle_of_attack")
    flight_conditions_group.altitude_m = csdl.Variable(value=flight.altitude_m, name="altitude (m)")

    ambient_conditions_group = compute_ambient_conditions_group(flight_conditions_group.altitude_m)

    angle_of_attack = flight_conditions_group.angle_of_attack  # the design variable itself
    if flight.optimize_angle_of_attack:
        angle_of_attack.set_as_design_variable(
            lower=flight.angle_of_attack_bounds_deg[0], upper=flight.angle_of_attack_bounds_deg[1]
        )

    # ---- Mesh warp -> flow solve -> functions (parallel region) ------------------------------
    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:
        if warper == "jax":
            x_vol = JaxMeshWarper(jax_warp, local_to_global).evaluate(x_surf_full.flatten())
        else:
            i0, i1 = x_surf_indices[rank]
            x_vol = DAFoamMeshWarper(dafoam_instance).evaluate(x_surf_full[i0:i1, :].flatten())

        flight_conditions_group.angle_of_attack = mpi_region.split_custom(
            flight_conditions_group.angle_of_attack, split_func=lambda x: x
        )

        dafoam_input_variables_group = compute_dafoam_input_variables(
            dafoam_instance, ambient_conditions_group, flight_conditions_group, x_vol
        )

        dafoam_solver = DAFoamSolver(dafoam_instance, check_mesh=check_mesh, on_adjoint_failure=on_adjoint_failure)
        dafoam_solver_states = dafoam_solver.evaluate(dafoam_input_variables_group)

        dafoam_function_outputs = DAFoamFunctions(dafoam_instance).evaluate(
            dafoam_solver_states, dafoam_input_variables_group
        )

        CL = dafoam_function_outputs.CL
        CD = dafoam_function_outputs.CD
        mpi_region.set_as_global_output(CL)
        mpi_region.set_as_global_output(CD)

    # ---- Objective and constraint --------------------------------------------------------
    CD.set_as_objective()
    CD.add_name("CD")
    if cl_target is not None:
        CL.set_as_constraint(lower=cl_target, upper=cl_target)  # an equality constraint
    CL.add_name("CL")

    recorder.stop()
    # Inline (eager) evaluation was needed while building (the geometry code reads values), but the simulator
    # re-activates this recorder to build the derivative graph, and with inline on every derivative operation would
    # then execute as it is created: on the real solver that ran DAFoam's adjoint 14 times in the first
    # compute_totals() instead of 2.
    recorder.inline = False

    return AirfoilModel(
        recorder=recorder,
        CL=CL,
        CD=CD,
        design_variables={
            "thickness": percent_change_in_thickness_dof,
            "camber": normalized_percent_camber_change_dof,
            "angle_of_attack": angle_of_attack,
        },
        dafoam_instance=dafoam_instance,
        geometry=geometry,
        comm=comm,
        rank=rank,
        flight_conditions=flight_conditions_group,
        ambient_conditions=ambient_conditions_group,
        x_surf=x_surf_full,
        timings=timings,
        jax_warp=jax_warp,
        local_to_global=local_to_global,
    )
