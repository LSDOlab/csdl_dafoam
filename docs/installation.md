# Installation

```{tip}
The two procedures that have actually been run are the source build in {doc}`install-mac` (a Linux VM on an Apple-silicon Mac,
verified end to end) and its Linux equivalent in {doc}`install-linux` (same script; verified on ARM64 only). A conda package of the
DAFoam stack is planned but its recipes have never been built.
```

DAFoam links against a source-built OpenFOAM, so there are no PyPI wheels for it. This package therefore
has two layers:

| Layer | Contains | Installed with |
| --- | --- | --- |
| compiled stack | OpenFOAM v2506 + AD libraries, DAFoam, PETSc, MPI | **built from source** with `scripts/vm/install_dafoam_ubuntu.sh` (about 2 hours, unattended); conda packages are planned |
| Python | `csdl_dafoam` itself, `csdl_alpha`, `lsdo_geo`, `lsdo_function_spaces`, `modopt`, `idwarp-jax` | pip (git for the lab packages) |

Volume-mesh deformation uses [IDWarp-JAX](https://github.com/LSDOlab/idwarp-jax), which is pure Python/JAX, so
the Fortran IDWarp (and the CGNS build behind it) is **not needed**. It remains available as `warper="idwarp"`;
its optional recipes are in `recipes/`.

Anything that runs DAFoam needs Linux. The pure-Python parts, and the whole test suite, also work on macOS.

## 1. Full install: build DAFoam from source

* Apple-silicon Mac: {doc}`install-mac` (a Lima Linux VM, then the Linux procedure inside it).
* Linux (Ubuntu 24.04): {doc}`install-linux`.

Both end with `python -m csdl_dafoam.doctor` (also installed as `csdl-dafoam-doctor`), which prints which components import and whether the
OpenFOAM environment is loaded. A fully working install shows `[ok]` for every row.

## Conda (experimental, not yet available)

`environment.yml` creates the *Python* side of the environment (MPI, PETSc, JAX, the lab packages, this repository) with
`conda env create -f environment.yml`; it does not contain DAFoam or OpenFOAM. The recipes for those in
[`recipes/`](https://github.com/LSDOlab/csdl_dafoam/tree/main/recipes) were written from DAFoam's installation guide and have **not been
built**; the source build revealed they are incomplete (see {doc}`status`). Publishing a source-built OpenFOAM package to the `lsdolab`
channel also needs a decision on size and licensing (DAFoam and OpenFOAM are GPL-3.0).

## 2. Python-only install (no DAFoam)

Enough to use the geometry parameterization, the atmosphere model, to read the code, and to run the test suite:

```bash
pip install -r requirements-lab.txt        # csdl_alpha, modopt, idwarp-jax from LSDOlab's git
pip install -e ".[geometry,opt,test]"      # incl. OpenSQP's qpsolvers, quadprog, highspy
pytest
```

`requirements-lab.txt` exists because `csdl_alpha`, the lab's `modopt` and `idwarp-jax` are not on PyPI. Do **not**
`pip install modopt` from PyPI; that is an unrelated package.

Importing `csdl_dafoam` never requires DAFoam, PETSc or MPI. Setting up a DAFoam run without them raises an
`ImportError` that points back here.

## 3. Building the DAFoam packages yourself

Needs [rattler-build](https://github.com/prefix-dev/rattler-build) on Linux x86-64 and a few hours:

```bash
rattler-build build -r recipes/openfoam-dafoam    -c lsdolab -c conda-forge
rattler-build build -r recipes/dafoam             -c ./output -c lsdolab -c conda-forge
rattler-build build -r recipe                     -c ./output -c lsdolab -c conda-forge
```

Then point `environment.yml` at the local channel by adding `- file:///path/to/csdl_dafoam/output` ahead of
`lsdolab` in its `channels:` list. The OpenFOAM build compiles OpenFOAM twice (original and reverse-mode AD),
needs about 10 GB of scratch space, and three of its source hashes are placeholders to fill in with
`scripts/recipe_hashes.sh`. See `recipes/README.md`.

Publishing to the channel is a manual GitHub Actions dispatch (`.github/workflows/conda-stack.yml`) and should
only follow a local build that passed its recipe tests.

## macOS

DAFoam itself (the flow and adjoint solver) is Linux-only; nobody has published a macOS or ARM-Linux build, and the
conda recipes here target `linux-64`. What does run on an Apple-silicon Mac:

- everything in section 2 (geometry, IDWarp-JAX, OpenSQP, the whole test suite, the examples against the fake solver);
- **native OpenFOAM v2506** from [OpenFOAM.app](https://github.com/gerlero/openfoam-app), for validating meshes and
  cases with OpenFOAM's own tools:

  ```bash
  brew tap gerlero/openfoam
  brew install --cask gerlero/openfoam/openfoam@2506     # arm64, macOS 14+; installs to /Applications
  pip install -e ".[test]" && pytest -m openfoam         # warped meshes vs OpenFOAM's checkMesh
  ```

  `csdl_dafoam.foam_check.check_mesh` finds it automatically (the `openfoam2506` launcher) and applies DAFoam's
  mesh-check criteria. OpenFOAM.app installs OpenFOAM inside a read-only, case-sensitive disk image mounted on demand;
  add your own solvers to `$FOAM_USER_APPBIN`.

For a real DAFoam solve from a Mac, the route that has been run end to end is an **ARM64 Linux VM** (Lima, Apple's hypervisor, native
speed, no emulation) with DAFoam built from source by `scripts/vm/install_dafoam_ubuntu.sh`: see `scripts/vm/README.md`. On an
M5 Max the build took a few hours, and the airfoil examples then run in the VM. Alternatives are a Linux machine or DAFoam's
official Docker image `dafoam/opt-packages` (amd64 only, so emulated on Apple silicon).

## 4. Running

DAFoam decomposes the mesh to match the MPI rank count, so the case must be prepared for the number of ranks
you run. The examples do this for you (`csdl_dafoam.cases.prepare_case`); for your own cases, set
`numberOfSubdomains` in `system/decomposeParDict` to the `-np` value.

```bash
mpirun -np 4 python examples/airfoil_analysis.py
```

## Versions

The build script, recipes and `environment.yml` follow DAFoam v5.1.1's tested set: OpenFOAM v2506, PETSc 3.21, mpi4py >= 4,
NumPy >= 2, Python >= 3.11 (3.12 used), OpenMPI 4.1. The interface itself calls only DAFoam
methods that exist unchanged in v4.0.4 and v5.1.1.

## Troubleshooting

`ImportError: Could not import 'dafoam'`
: The compiled stack is not installed or the environment is not activated. Run `python -m csdl_dafoam.doctor`.

`ImportError: qpsolvers cannot be imported` (or `quadprog`, `highspy`)
: OpenSQP, the default optimizer, solves QP subproblems with these. `pip install -e ".[opt]"`, or pick
  `--optimizer PySLSQP`. (modOpt's own `opensqp` extra lists only `qpsolvers`.)

`DAFoam mesh check failed`
: DAFoam's `checkMesh` rejected the (warped) mesh and wrote the failed mesh into the case directory. Usually the
  optimizer asked for too large a design step; tighten the design-variable bounds or scaling. Pass
  `check_mesh=False` to `build_airfoil_model` only to debug.

`DAFoam primal solution failed`
: The flow solve did not complete. Inspect the solver log and residuals in the case directory.

Numbers of ranks differ from `numberOfSubdomains`
: DAFoam aborts at start-up. Re-run `prepare_case` (or fix `decomposeParDict`).

macOS or Windows: `NameError: name 'global_functions' is not defined`
: `lsdo_function_spaces` parallelizes surface projection with a process pool that needs the `fork` start method.
  `csdl_dafoam` projects serially by default, so this only appears if you pass `num_workers > 1` through
  `projection_options`.
