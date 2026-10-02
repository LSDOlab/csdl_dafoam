# Quickstart

With the environment from {doc}`installation` active, evaluate the bundled NACA 0012 once:

```bash
mpirun -np 4 python examples/airfoil_analysis.py
```

This copies the bundled OpenFOAM case into `run/case/` (decomposed for 4 ranks), builds the CSDL model, runs
the DAFoam flow solve and prints CL and CD.

The same thing in a few lines of Python, which is all the example does:

```python
from mpi4py import MPI
import csdl_dafoam as cd
from csdl_dafoam.options import options_for_case

comm = MPI.COMM_WORLD
case = cd.prepare_case("naca0012_euler", "run/case", comm=comm)   # decomposed for comm.size ranks

model = cd.build_airfoil_model(
    case_dir=case,
    da_options=options_for_case("naca0012_euler"),
    geometry_file=cd.geometry_path(),
    cache_dir="run/cache",
    flight=cd.FlightConditions(angle_of_attack_deg=3.0, optimize_angle_of_attack=False),
    cl_target=None,            # no lift constraint: just analyze
    comm=comm,
)
print(model.analyze())         # {'CL': ..., 'CD': ...}
```

Add an optimizer and you have the optimization example:

```python
model = cd.build_airfoil_model(..., cl_target=0.5)   # optimize_angle_of_attack defaults to True
sim = model.make_simulator()
model.optimize(sim, maxiter=20)                      # SLSQP; the simulator now holds the optimum
```

Next: {doc}`examples` for what each script does, {doc}`how-it-works` to build your own model from the pieces.
