# Installing on an Apple-silicon Mac

DAFoam runs only on Linux. On a Mac the route that has been run end to end is a **Linux VM**: [Lima](https://lima-vm.io)
runs Ubuntu 24.04 for ARM64 under Apple's hypervisor, so the guest executes ARM64 instructions directly on the Mac's cores.
There is no emulation and CPU-bound work runs at close to native speed. DAFoam is then built from source inside the VM.

This page is the exact procedure that produced the verified results in {doc}`status`. Where a step was run by hand rather than
by a script, it says so.

## What you need

| | |
| --- | --- |
| Mac | Apple silicon (M-series), macOS 13 or later (verified on macOS 26) |
| Homebrew | to install Lima |
| RAM | the VM reserves 32 GiB in the shipped config; edit `memory:` in `scripts/vm/lima-dafoam.yaml` for a smaller Mac (the build itself is happy with 16 GiB; fewer cores means a longer build) |
| Disk | 160 GiB virtual disk, grows on demand; the finished install is **about 11 GB** |
| Time | **about 2 hours** of unattended build on 12 cores (measured on an M5 Max: Python 1 min, PETSc 20, OpenFOAM v2506 40, OpenFOAM-AD 40, Hisa4DAFoam + DAFoam 7) |
| Network | the build downloads about 1 GB of sources |

## 1. Create the VM

```bash
brew install lima
git clone https://github.com/LSDOlab/csdl_dafoam && cd csdl_dafoam
limactl start --name=dafoam scripts/vm/lima-dafoam.yaml     # answer "Proceed"; first start downloads the Ubuntu image
limactl shell dafoam uname -m                               # should print aarch64
```

The config installs DAFoam's documented Ubuntu prerequisites on first boot (the same list as
`scripts/vm/install_prereqs_ubuntu.sh`). **No host folder is mounted into the VM**: files go in with `limactl copy`.

```{warning}
Never add a Lima mount written as `{{.Dir}}/../..`. In Lima `{{.Dir}}` is the *instance* directory, so that path is your home
directory. An earlier version of this config did exactly that and exposed the whole home folder (read-only) to the VM before it
was caught. `tests/test_vm_scripts.py` now fails if the config has any mount.
```

Start the VM from **one** terminal only. A second `limactl start` (or `limactl shell` offering to start it) races the first and
one of them fails with "another hostagent may already be running".

## 2. Build DAFoam (about 2 hours)

```bash
limactl copy scripts/vm/install_dafoam_ubuntu.sh dafoam:~/
limactl shell dafoam -- sudo systemd-run --unit=dafoam-build --uid=$(id -u) --gid=$(id -g) \
    --setenv=HOME=$HOME --working-directory=$HOME --collect bash -c 'bash install_dafoam_ubuntu.sh > install.out 2>&1'
```

`systemd-run` makes the build a detached service inside the VM, so it survives closing your terminal (a plain
`limactl shell ... &` can be killed with the session). Follow it, and check it is still alive:

```bash
limactl shell dafoam -- tail -f dafoam/build.log                          # Ctrl-C leaves the build running
limactl shell dafoam -- systemctl is-active dafoam-build                  # "active" while building
limactl shell dafoam -- ls dafoam/.stages                                 # finished stages
limactl shell dafoam -- ls dafoam/DONE dafoam/FAILED                      # one of these exists at the end
```

Stages, in order: `1-python`, `2-petsc`, `2b-petsc4py`, `3-openfoam`, `4-openfoam-ad`, `5a-pyofm`, `5b-hisa4dafoam`,
`5-dafoam`. The script is **resumable**: if a stage fails, fix the cause and launch it again; finished stages are skipped.

What the script pins, so the build is repeatable: OpenFOAM v2506, PETSc 3.21.6, mpi4py 4.1.1, DAFoam 5.1.1, pyOFM 1.2.3,
OpenFOAM-AD at commit `2e519c1`, Hisa4DAFoam at commit `73ced10`, and the Python packages numpy 2.3.5, scipy 1.18.1, Cython
3.0.5, OpenMDAO 3.45.1. What it does not pin: the Ubuntu packages (the 24.04 release as of build; compilers were gcc 13.3,
OpenMPI 4.1.6). `scripts/vm/pip-freeze-ubuntu2404-aarch64.txt` records the full verified Python environment.

## 3. Check DAFoam

```bash
limactl shell dafoam bash -c '. ~/dafoam/loadDAFoam.sh && python -c "import dafoam; print(dafoam.__file__)" && which simpleFoam simpleFoamADR'
```

Every shell that runs DAFoam must first `. ~/dafoam/loadDAFoam.sh` (it sets the OpenFOAM environment).

## 4. Install csdl_dafoam and its Python stack in the VM

```bash
tar --exclude=.git -czf /tmp/csdl_dafoam.tgz . && limactl copy /tmp/csdl_dafoam.tgz dafoam:~/
limactl shell dafoam bash -c '
  mkdir -p csdl_dafoam && tar xzf csdl_dafoam.tgz -C csdl_dafoam
  . ~/dafoam/loadDAFoam.sh
  pip install -r csdl_dafoam/requirements-verified.txt
  pip install pyslsqp qpsolvers quadprog highspy jax pytest matplotlib
  pip install -e csdl_dafoam --no-deps
  python -m csdl_dafoam.doctor'
```

`requirements-verified.txt` pins `csdl_alpha`, `modopt` and `idwarp-jax` to the commits that were verified together. The doctor
prints `[ok]` for every row except the optional Fortran IDWarp.

## 5. Run

```bash
limactl shell dafoam bash -c '. ~/dafoam/loadDAFoam.sh && cd csdl_dafoam && pytest -m dafoam'              # real-solver tests, about 1 min
limactl shell dafoam bash -c '. ~/dafoam/loadDAFoam.sh && cd csdl_dafoam && mpirun -np 4 python examples/airfoil_analysis.py'
limactl shell dafoam bash -c '. ~/dafoam/loadDAFoam.sh && cd csdl_dafoam && mpirun -np 4 python examples/airfoil_optimization.py --maxiter 40'
```

Results land in `run/` inside the VM (set with `--workdir`). Copy figures back to the Mac to look at them:

```bash
limactl copy dafoam:csdl_dafoam/run/optimization.png .      # path as printed by the example
```

## Managing the VM

```bash
limactl stop dafoam                    # free the RAM; the install stays on its disk
limactl start dafoam                   # resume (no config file needed after the first start)
limactl delete dafoam                  # remove the VM and the 11 GB
```

## Problems met while building this, and their fixes

| Symptom | Cause | Fix (already in the script) |
| --- | --- | --- |
| `petsc4py` wheel fails: `execute() got an unexpected keyword argument 'dry_run'` | petsc4py 3.21's build script needs an old setuptools | `pip install "setuptools<70"` |
| `boxTurb` / `noise` fail to link: `undefined reference to fftw_...` | OpenFOAM only adds ThirdParty FFTW to the library path if it exists when `etc/bashrc` is sourced, and it is built during the first `Allwmake` | run `Allwmake`, re-source the environment, run it again (quick) |
| DAFoam fails: `fatal error: writeFields.H` / `characteristicBase.H: No such file` | the guide builds **Hisa4DAFoam** between OpenFOAM and DAFoam, and DAFoam's include path reads it | stage `5b-hisa4dafoam` before `5-dafoam` |
| a resumed run fails `OpenFOAM environment not found` | the script regenerates `loadDAFoam.sh` on each start and dropped the OpenFOAM lines when that stage was skipped | the loader now always includes them when OpenFOAM is installed |
| a failure marker appears but the build is fine | an `ERR` trap fires on commands inside OpenFOAM's `etc/bashrc` that return non-zero by design | the trap is now `EXIT`-only |
| `limactl shell` seems to hang | it waits for the VM's processes | run long work under `systemd-run`, as above |
| `pkill -f` kills your own command | the pattern appears in your shell's command line | use `systemctl stop <unit>` |
