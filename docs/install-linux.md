# Installing on a Linux machine

The same script that built DAFoam inside the Mac's VM ({doc}`install-mac`) builds it on a Linux machine, with no VM.

```{admonition} What has and has not been tested
:class: warning
The script was run end to end on **Ubuntu 24.04.5, ARM64 (aarch64)** only (inside a VM on an Apple-silicon Mac). It has
**not** been run on x86-64, on other Ubuntu releases, on other distributions, or on an HPC system. Nothing in it is
architecture-specific (OpenFOAM v2506 has x86-64 and ARM64 build rules and the script picks the platform up from the
OpenFOAM environment), and DAFoam's own guide targets Ubuntu 24.04 on x86-64, so it is **expected to work on x86-64
Ubuntu 24.04**; treat it as a starting point and expect to fix small things the way the aarch64 run needed.
```

## Requirements

* Ubuntu 24.04 (the version DAFoam's guide targets), with `sudo` for the package install step;
* about **11 GB** of disk and, on 12 cores, about **2 hours** (PETSc 20 min, OpenFOAM 40, OpenFOAM-AD 40, Hisa4DAFoam + DAFoam
  7). Fewer cores take proportionally longer; the script uses all of them (`NPROC=8 bash install_dafoam_ubuntu.sh` to limit);
* about 16 GiB of RAM (the C++ compiles are the peak);
* network access to GitHub, SourceForge and the PETSc download site (about 1 GB).

## 1. System packages

```bash
git clone https://github.com/LSDOlab/csdl_dafoam && cd csdl_dafoam
bash scripts/vm/install_prereqs_ubuntu.sh        # apt: compilers, cmake, flex, bison, OpenMPI, CGAL, scotch, ...
```

## 2. Build DAFoam

```bash
bash scripts/vm/install_dafoam_ubuntu.sh         # installs into ~/dafoam; set DAFOAM_ROOT_PATH to change
```

Run it under `tmux`/`screen`, or as `nohup bash scripts/vm/install_dafoam_ubuntu.sh > install.out 2>&1 &`, so that closing the
terminal does not kill it. The log is `~/dafoam/build.log`; `~/dafoam/DONE` appears on success and `~/dafoam/FAILED` on failure.
The script is resumable (finished stages are skipped): fix the problem and run it again. Every download is checked against a
SHA-256 and the two sources that DAFoam only offers as branches are pinned to commits.

When it finishes, load the environment in every shell that will run DAFoam:

```bash
. ~/dafoam/loadDAFoam.sh
python -c "import dafoam; print(dafoam.__file__)" && which simpleFoam simpleFoamADR
```

## 3. Install csdl_dafoam

```bash
. ~/dafoam/loadDAFoam.sh
pip install -r requirements-verified.txt                      # csdl_alpha, modopt, idwarp-jax at the verified commits
pip install pyslsqp qpsolvers quadprog highspy jax pytest matplotlib
pip install -e . --no-deps
python -m csdl_dafoam.doctor                                  # all rows [ok] except the optional Fortran IDWarp
```

## 4. Verify and run

```bash
pytest -m dafoam                                              # real-solver tests, about 1 min on 1 rank
mpirun -np 4 python examples/airfoil_analysis.py
mpirun -np 4 python examples/airfoil_optimization.py
```

For the verified numbers to compare with (Euler-mode case, alpha = 3 deg): **CL = 0.4706, CD = 0.011743 (117.4 counts)**. If your
analysis prints values that differ in the third digit, something in the stack differs; those that match to 4 to 5 digits show the
installation is equivalent.

## Other systems

* **No `sudo` (HPC):** skip `install_prereqs_ubuntu.sh`, load the cluster's compiler, MPI, CMake, flex and bison modules instead,
  and point the script at them. DAFoam's guide has an Intel-compiler variant (<https://dafoam.github.io/installation-source-intel.html>).
  The script's PETSc configure flags and `mpi4py` build use the MPI wrappers on `PATH`.
* **Another distribution:** install the packages in `install_prereqs_ubuntu.sh` under their local names; the build steps are
  distribution-independent.
* **OpenFOAM's `Allwmake` fails at a link step with `fftw`:** see "Problems met" in {doc}`install-mac`; the script already handles it.
* **A conda install** (`environment.yml`, `recipes/`) is described in {doc}`installation`; those recipes are unbuilt and known to be
  incomplete, so prefer this script until they are fixed.
