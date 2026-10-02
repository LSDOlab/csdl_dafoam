# Conda recipes for the DAFoam stack

> **EXPERIMENTAL: never built.** A from-source build of DAFoam on ARM64 Linux (`scripts/vm/install_dafoam_ubuntu.sh`) showed these recipes
> are incomplete: DAFoam's C++ also needs **Hisa4DAFoam** built first, OpenFOAM's FFTW-dependent utilities need a second `Allwmake` pass after the
> environment is re-sourced, and DAFoam's test suite needs pyOFM, mphys and pygeo. Use the build script until these are ported and a recipe has
> been built. See `docs/status.md`.

`recipe/recipe.yaml` (one level up) packages `csdl_dafoam` itself. This directory packages what it
needs at run time and that conda-forge does not provide, so that
`conda env create -f environment.yml` works without Docker.

| Recipe | Package | What | Build |
| --- | --- | --- | --- |
| `mdolab-baseclasses/` | `mdolab-baseclasses` | **optional**: pure Python, needed only by the Fortran IDWarp | seconds |
| `openfoam-dafoam/` | `openfoam-dafoam` | OpenFOAM v2506 + DAFoam's UPstream patch + the reverse-mode-AD fork, in DAFoam's `$DAFOAM_ROOT_PATH` layout, with an activation script | hours, ~10 GB scratch |
| `idwarp/` | `idwarp` | **optional**: the Fortran/f2py IDWarp, only for `warper="idwarp"`; builds its own static CGNS | minutes |
| `dafoam/` | `dafoam` | DAFoam's adjoint library + solvers + Python layer (runs DAFoam's own `Allmake`) | an hour or more |

## Status: written from DAFoam's installation guide, never built, and known to be incomplete

These were authored on a machine without conda, Linux or OpenFOAM, translating the steps of
https://dafoam.github.io/installation-source.html (v5.1.1) into rattler-build recipes. Nothing here has
been built yet. Expect the first build to need fixes, most likely in: relocating OpenFOAM's absolute
paths (`openfoam-dafoam`), the compiler names wmake expects (the shims in `build.sh`), and how
conda-forge's PETSc is found (`PETSC_ARCH` is empty for an installed PETSc, not a build tree).
Three source hashes are `TODO`; `scripts/recipe_hashes.sh` prints them.

**Known gaps, found by doing the same build from source on ARM64 Linux (`scripts/vm/install_dafoam_ubuntu.sh`, which works):**

1. DAFoam's C++ includes headers from **Hisa4DAFoam** (`git clone https://github.com/DAFoam/Hisa4DAFoam && ./Allmake`, after OpenFOAM and before DAFoam). Neither recipe builds it yet.
2. OpenFOAM's `Allwmake` must run **twice**, re-sourcing `etc/bashrc` in between, or `boxTurb` and `noise` fail to link: ThirdParty FFTW is built during the first pass and OpenFOAM only adds it to the library path if it already exists when the environment is sourced.
3. DAFoam's own regression tests need pyOFM, mphys and pygeo (not needed to run csdl_dafoam).

Versions (DAFoam v5.1.1's tested set, from its installation guide): OpenFOAM v2506, PETSc 3.21.x,
mpi4py >= 4, NumPy >= 2, Python >= 3.11, IDWarp 2.6.4, mdolab-baseclasses 1.9.0. OpenMPI 4.1 matches what DAFoam's
Docker image (Ubuntu 24.04) uses.

## Build order and publishing

Needs [rattler-build](https://github.com/prefix-dev/rattler-build), on Linux x86-64:

```bash
rattler-build build -r recipes/openfoam-dafoam    -c lsdolab -c conda-forge
rattler-build build -r recipes/dafoam             -c ./output -c lsdolab -c conda-forge
rattler-build build -r recipe                     -c ./output -c lsdolab -c conda-forge

# only if you want the Fortran IDWarp (warper="idwarp"):
rattler-build build -r recipes/mdolab-baseclasses -c lsdolab -c conda-forge
rattler-build build -r recipes/idwarp             -c ./output -c lsdolab -c conda-forge
```

(`-c ./output` makes later builds find the packages built earlier, before anything is published.)
Publishing to the `lsdolab` channel is what `.github/workflows/conda-stack.yml` does on a manual
dispatch with `publish: true`; do it only after a local build has passed its tests.

## IDWarp-JAX replaces most of this

The default warper, [IDWarp-JAX](https://github.com/LSDOlab/idwarp-jax), is pure Python/JAX and installed with pip
(git), so the compiled stack shrinks to OpenFOAM + DAFoam; the Fortran IDWarp, CGNS and baseclasses recipes are
kept only for users who want `warper="idwarp"`.

## Why not conda-forge's `openfoam`?

conda-forge ships ESI OpenFOAM v2412. DAFoam v5 is built and tested against v2506 plus its own
patches and an AD fork of the libraries, so it needs its own OpenFOAM.
