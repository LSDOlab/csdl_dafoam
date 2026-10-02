# Migrating from the original scripts

The original repository was a folder of scripts you ran from inside `folder_cfd/` (`run.sh` -> `mpi_tot.py`).
The code is the same code, reorganized; the old names still work.

| Original | Now |
| --- | --- |
| `folder_cfd/csdl_dafoam.py`: `instantiateDAFoam` | `csdl_dafoam.solver.instantiate_dafoam` (`instantiateDAFoam` kept as an alias) |
| `csdl_dafoam.py`: `DAFoamSolver`, `DAFoamFunctions` | `csdl_dafoam.solver` |
| `csdl_dafoam.py`: `compute_dafoam_input_variables` | `csdl_dafoam.inputs` |
| `folder_cfd/csdl_idwarp.py`: `DAFoamMeshWarper` | `csdl_dafoam.mesh_warp` |
| `folder_cfd/standard_atmosphere_model.py` | `csdl_dafoam.atmosphere` |
| `folder_cfd/bwb_helper_functions.py` (`setup_geometry`, pickle helpers, `gather_array_to_rank0`) | `csdl_dafoam.geometry` (`setup_geometry` kept as an alias), `csdl_dafoam.mpi_utils` |
| `folder_cfd/mpi_tot.py` | `csdl_dafoam.airfoil.build_airfoil_model` plus `examples/airfoil_optimization.py --case naca0012` |
| `run.sh` | `mpirun -np 4 python examples/airfoil_optimization.py --case naca0012` |
| `folder_cfd/openfoam_naca0012/`, `folder_geo/.../airfoil_transonic_unitspan_2.stp` | bundled data, `csdl_dafoam.cases` |
| Docker image `cfdkang/csdl_dafoam` | conda environment ({doc}`installation`) |

## Behavior changes

- Volume-mesh deformation defaults to IDWarp-JAX (`warper="jax"`) instead of the Fortran IDWarp; the optimizer defaults
  to modOpt's OpenSQP instead of PySLSQP. `warper="idwarp"` and `algorithm="PySLSQP"` reproduce the original choices.
- Failed mesh check or primal solve raises `DAFoamError` (originally: skipped / ignored). See {doc}`how-it-works`.
- DAFoam calls run in the case directory (originally only the adjoint solve did).
- Geometry and projection caches go to `cache_dir`, keyed by a digest of the STEP file and the surface mesh; the
  committed `stored_files/*.pickle` and `__pycache__` files are gone, and a cache that no longer unpickles is
  regenerated instead of crashing.
- `lsdo_geo` imports use the current top-level names (`lg.SectionalParameterization`, ...), falling back to the
  deprecated `VolumeSectionalParameterization` names on older versions.
- Projection is serial by default (`projection_options={"num_workers": N}` to parallelize).
- The OpenFOAM case no longer ships a stale `blockMeshDict` (it described a different geometry) or `controlDict~`.
- `gather_array_to_rank0` uses pickle-based `comm.gather` instead of `Gatherv` with explicit MPI datatypes: same
  result, no `mpi4py` import in the module, testable without MPI.
- The debugging switches of `mpi_tot.py` (`faulthandler`, `PETSC_OPTIONS=-malloc_debug`, interactive `vedo`
  plots) are not carried over.

## Version note

The original ran on a Python 3.9 Docker image (`dafoam/opt-packages`, OpenFOAM v1812-era DAFoam and `csdl_alpha`'s
`dev` branch). The packaged stack targets DAFoam v5.1.1 on OpenFOAM v2506 and current `csdl_alpha` main. The
DAFoam methods used are identical in v4.0.4 and v5.1.1 and `csdl_alpha` main now contains the MPI region
operations, but the bundled case files were written for v1812 and have not yet been run on v2506 ({doc}`status`).
