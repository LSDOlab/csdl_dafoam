# Verification status

The package was first overhauled on a machine with no Linux, conda or OpenFOAM, and tested against fake solvers. It has
since been run against a **real DAFoam v5.1.1 / OpenFOAM v2506 built from source on ARM64 Linux** (a Lima VM on an
Apple-silicon Mac, `scripts/vm/`). This page states exactly what each claim rests on.

## Verified on the real solver (ARM64 Linux, DAFoam 5.1.1, 4032-cell NACA 0012, Euler-mode case, 1 and 4 MPI ranks)

| Claim | Evidence |
| --- | --- |
| DAFoam, OpenFOAM v2506 (with DAFoam's MPI patch) and the reverse-mode-AD OpenFOAM build and import on ARM64 | `scripts/vm/install_dafoam_ubuntu.sh`; `import dafoam`, `PYDAFOAM` |
| The interface runs a real flow solve: alpha = 3 deg, 238 m/s gives **CL = 0.4706, CD = 0.01174 (117 counts)**; the SIMPLE solve converges to the 1e-8 tolerance in 164 iterations from a cold start | `tests/test_dafoam_integration.py`, `scripts/vm/fd_study.py` |
| **Adjoint derivatives are correct.** Finite differences converge onto the adjoint as the step grows past the noise floor: best-step disagreement 0.2 to 0.7% (alpha, camber), 1 to 2% (thickness), 0.1 to 0.5% for CL. At a step of 1e-3 or below the finite differences are noise (8% to 34% off), which is why a naive single-step check fails | `scripts/vm/fd_study.py` |
| **A real optimization works**: `examples/airfoil_optimization.py` (IDWarp-JAX, OpenSQP, 4 ranks). **100 iterations (36.7 min): CD 117.4 to 75.4 counts (-35.8%), CL 0.50000** (target 0.5), alpha 0.00 deg (its lower bound), thickness changes of +19.6%, +35.7%, +33.9% at the three interior control points and camber changes of +4.2%, +8.2%, +11.2%. CD stops changing at about iteration 65; optimality ends near 6e-5 (tolerance 1e-5), so modOpt reports the iteration cap, not convergence. Earlier, shorter runs ended at different, less optimized local designs: 15 iterations CD 85.6 (alpha 2.81 deg), 40 iterations CD 76.6 (alpha 0.84 deg, still improving). The recorded CD of every iteration matches modOpt's objective to 0.0014 counts | run logs, `modopt_summary.out`, `history.npz`; the movie and figures in the docs |
| The initial design's drag includes a **real shock**: the surface C_p drops sharply at about 30% chord (from -1.45 to -0.55, upper surface), and the optimized design has none. So the drag reduction is wave-drag reduction, not an artifact of the optimizer | `viz.plot_airfoil_comparison` of the real initial and final flow solutions |
| `csdl_dafoam.viz` reads real DAFoam output: an MPI-decomposed (4 processor) solution is reassembled on the global mesh, deformed meshes included, and plotted (pressure-coefficient contours, surface C_p, shapes) | `tests/test_viz.py` on a real 4-rank snapshot (coarse mesh); the figures in the README and docs |
| The 336-cell test mesh (`naca0012_coarse`) runs on the real solver: model build 2.6 s, warm solve 0.2 s, adjoint 0.74 s, adjoint agrees with finite differences to 0 to 2.3%, 1 and 4 ranks identical; OpenFOAM v2506's `checkMesh` accepts it (non-orthogonality 22.5 deg, skewness 0.7) | `scripts/vm/fd_study.py`, `timing.py`, `tests/test_coarse_mesh.py` |
| The install is reproducible: every download is checksum-verified (the 8 hashes were computed independently in the VM), the two branch-only sources are pinned to commits, the Python and lab-package versions are recorded | `scripts/vm/install_dafoam_ubuntu.sh`, `pip-freeze-ubuntu2404-aarch64.txt`, `requirements-verified.txt`, `tests/test_vm_scripts.py` |
| Cost on this machine, 4032 cells: forward solve about 3.9 s (1 rank) and 1.4 s (4 ranks); adjoint (two outputs) about 7 s at the base design on 4 ranks, up to about 31 s (4 ranks) or 101 s (1 rank) at a design with CL 0.58, so it depends on the flow state; first `compute_totals` about 17 s on 4 ranks before the recorder fix was re-timed (JAX compilation is about 5 s of it; not re-timed since) | `scripts/vm/timing.py`, `profile_totals.py` |
| The CSDL graph itself is not the cost: a 2,400-node graph compiles in about 5 to 9 s. The slow first call was our bug (inline recorder re-executing the adjoint 14 times instead of 2), now fixed | `scripts/vm/profile_totals.py` before and after |
| No GPU: DAFoam and OpenFOAM are CPU-only MPI codes | DAFoam docs and source; the VM has no GPU passthrough |

## Verified without DAFoam (fake solvers, native checks)

| Claim | How |
| --- | --- |
| The package builds a wheel, installs into a clean environment pulling only NumPy and SciPy, ships its data (geometry, mesh, cases), and its console script works | built and installed the wheel in a fresh venv |
| `import csdl_dafoam` needs no DAFoam, PETSc, MPI, CSDL or JAX | `test_import.py` (subprocess check of `sys.modules`) |
| The `lsdo_geo` / `lsdo_function_spaces` code paths work with the current releases (1.0.0 / 1.0.0) and CSDL main | real STEP import, caching, projection of the real wall mesh (max error 2.7e-6), FFD thickness and camber response |
| `DAFoamSolver`, `DAFoamFunctions`, `DAFoamMeshWarper` produce correct states, functions and **total derivatives** given a solver obeying DAFoam's API | exact comparison against closed-form linear fakes, both adjoint methods; deliberate sign/hand-back bugs are caught |
| `build_airfoil_model` and both example scripts run end to end, with IDWarp-JAX or the Fortran-IDWarp interface, and with OpenSQP or PySLSQP; reverse-mode derivatives through geometry, warp and flow agree with finite differences | `test_airfoil_model.py`, `test_examples.py` with a fake flow solver |
| **IDWarp-JAX on the real bundled airfoil mesh**: wall points land on the prescribed surface (2e-5), the span coordinate is untouched, no cell inverts and none falls below 69% of its volume even for a 50%-thickness plus 5%-camber change (86% with the default `LdefFact=100`), and its VJP matches finite differences (about 1e-7 relative) | `test_jax_warp.py` (real `idwarp-jax` 0.1.0, CPU) |
| **Warped meshes pass OpenFOAM v2506's own `checkMesh`** under DAFoam's criteria, natively on an Apple-silicon Mac (OpenFOAM.app v2506 arm64): max non-orthogonality 22.8 to 23.7 deg (limit 70), skewness 1.4 (limit 4), aspect ratio about 98 (limit 1000), no negative volumes, for the undeformed mesh and three IDWarp-JAX perturbations up to +50% thickness and 5% camber; OpenFOAM's minimum cell volume matches `foam_io.cell_volumes` to 1e-6; a deliberately tangled mesh is rejected | `test_openfoam_mesh.py` (`pytest -m openfoam`), `csdl_dafoam.foam_check` |
| The bundled mesh and case files are accepted by OpenFOAM v2506 | `checkMesh`, `potentialFoam` and stock `rhoSimpleFoam` all read the case (turbulence, thermophysics, boundary conditions) |
| **OpenSQP** (modOpt 0.3.1) solves the CSDL problem, hands its final design back to the simulator, and needs `qpsolvers`, `quadprog` and `highspy` | `test_airfoil_model.py`; found by running it |
| The interface calls only DAFoam methods that exist in v4.0.4 and v5.1.1 | compared against both releases' `pyDAFoam.py` / `mphys_dafoam.py` (names present; signatures read, not executed) |
| Python 3.9 and NumPy 2.0 / SciPy 1.13 / JAX 0.4.30 | the test run |

## Also verified on the real solver: parallel runs

| Claim | Evidence |
| --- | --- |
| 1 and 4 MPI ranks give the same answer (CL 0.57947, CD 0.012846 to the printed digits at the same design), so IDWarp-JAX's mapping of each rank's points into the global mesh works with a real `decomposePar` decomposition (shared processor-boundary points included), and the CSDL MPI region's reverse-pass summation is right | `scripts/vm/timing.py` at 1 and 4 ranks |
| The optimization example runs on 4 ranks | `examples/airfoil_optimization.py` run above |

## Not verified

| Item | Why it matters | What to do |
| --- | --- | --- |
| **modOpt does not report convergence on this problem.** The 100-iteration run ends at the iteration cap: drag has been flat since about iteration 65 (75.42 counts) and the lift constraint is met to 1e-7, but optimality reaches only about 6e-5 against a tolerance of 1e-5 | the design is a good local result for this mesh, but the optimizer's own criterion was not met | look at scaling of the design variables, and try `--tolerance 1e-4` or a larger `--maxiter` |
| **Whether the optimum is physically meaningful.** The mesh has only 4,032 cells and the Euler-mode case has no resolved boundary layer, so the 117-count baseline drag is likely dominated by numerical dissipation and a smeared shock, not wave drag alone, and an optimizer can exploit mesh-resolution artifacts | do not read these drag numbers as physical; the design changes (thinner, cambered, alpha 2.8) are plausible but unvalidated | refine the mesh and re-evaluate the optimized shape; compare with RANS |
| **The RANS case** (`--case naca0012`) on the real solver | only the Euler-mode case has been run | `airfoil_analysis.py --case naca0012` |
| OpenSQP against PySLSQP on the real problem | only OpenSQP was run | `--optimizer PySLSQP` |
| **The conda recipes** (`recipes/`, `recipe/recipe.yaml`, `environment.yml`) | still never built. The source build revealed that they are **incomplete**: DAFoam's C++ also needs **Hisa4DAFoam** built first (its headers are on the include path), OpenFOAM's FFTW-dependent utilities need a second `Allwmake` pass after the environment is re-sourced, and DAFoam's own test suite needs pyOFM, mphys and pygeo. The verified route is `scripts/vm/install_dafoam_ubuntu.sh`, which has all of this | port those three findings into `recipes/openfoam-dafoam` and `recipes/dafoam`, then build on Linux |
| x86-64 Linux, and the build script on anything but Ubuntu 24.04 ARM64 | verified only on one platform | run `scripts/vm/install_dafoam_ubuntu.sh` there |
| DAFoam's own regression suite (`tests/Allrun`) | not run: the test for our solver (`runRegTests_DARhoSimpleCFoam.py`) needs mphys and pygeo, which this build omits | install the MACH components or accept our own `pytest -m dafoam` as the check |
| IDWarp-JAX: the second symmetry plane (y = 0.1) it does not model, far-field motion, and scale | the solve and the adjoint were fine with them on this mesh; their effect on accuracy is unquantified, and every rank warps the whole mesh, untested for large 3D meshes | refine, compare with the Fortran IDWarp |
| The GitHub workflows (`tests.yml`, `pages.yml`, conda) | written, never run | they run on the pull request; Pages also needs Settings, Pages, Source: "GitHub Actions" |
| 3D visualization (`write_patch_vtk` on a real 3D wing, `plot_surface_3d`) | tested only on the one-cell-thick 2D mesh | try on a 3D case |
| `--adjoint-tol 1e-3` in a full optimization | measured on one gradient only (29% faster, derivative error 7e-4) | run an optimization with it and compare the optimum |

## Open questions for the maintainers

- Is publishing a *source-built OpenFOAM* package to the `lsdolab` channel acceptable (size, GPL licensing of
  DAFoam/OpenFOAM, rattler-build runner time)? The alternative is to keep the DAFoam stack in a single
  user-built environment and publish only `csdl_dafoam`.
- Should `lsdo-geo` and `lsdo-function-spaces` (PyPI) also be published on the channel so the environment needs
  no pip section?
