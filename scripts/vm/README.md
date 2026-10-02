# Running DAFoam on an Apple-silicon Mac (Linux VM)

DAFoam is Linux-only. On a Mac, the route that has been verified end to end here is an **ARM64 Linux VM** under Apple's
hypervisor ([Lima](https://lima-vm.io)). The guest runs ARM64 instructions directly on the Mac's cores (no emulation), so
CPU-bound work runs at close to native speed.

```bash
brew install lima
cd <repo>
limactl start --name=dafoam scripts/vm/lima-dafoam.yaml         # Ubuntu 24.04 aarch64, 12 CPUs, 32 GiB, 160 GiB disk
limactl copy scripts/vm/install_dafoam_ubuntu.sh dafoam:~/
limactl shell dafoam bash install_dafoam_ubuntu.sh               # a few hours (two OpenFOAM builds); resumable
```

Run long jobs detached so they survive the shell, and read their logs:

```bash
limactl shell dafoam -- sudo systemd-run --unit=build --uid=$(id -u) --gid=$(id -g) --setenv=HOME=$HOME \
    --working-directory=$HOME --collect bash -c 'bash install_dafoam_ubuntu.sh > install.out 2>&1'
limactl shell dafoam -- tail -f dafoam/build.log
```

Then install this package's Python stack in the VM's environment and use it:

```bash
tar --exclude=.git -czf /tmp/repo.tgz . && limactl copy /tmp/repo.tgz dafoam:~/
limactl shell dafoam bash -c 'mkdir -p csdl_dafoam && tar xzf repo.tgz -C csdl_dafoam && . dafoam/loadDAFoam.sh &&
  pip install "csdl_alpha @ git+https://github.com/LSDOlab/CSDL_alpha.git@main" \
    "modopt @ git+https://github.com/LSDOlab/modopt.git" "idwarp-jax @ git+https://github.com/LSDOlab/idwarp-jax.git" \
    lsdo-geo lsdo-function-spaces pyslsqp qpsolvers quadprog highspy jax pytest && pip install -e csdl_dafoam --no-deps'
limactl shell dafoam bash -c '. dafoam/loadDAFoam.sh && cd csdl_dafoam && pytest -m dafoam -s'
limactl shell dafoam bash -c '. dafoam/loadDAFoam.sh && cd csdl_dafoam && mpirun -np 4 python examples/airfoil_optimization.py'
```

## Things that went wrong, so they need not go wrong again

* **Do not mount host folders using `{{.Dir}}`.** In Lima that is the *instance* directory; `{{.Dir}}/../..` is your home
  directory. The config therefore has no mounts; copy files in with `limactl copy`.
* **Do not run `limactl start` twice** (for example from two terminals): the second one loses the race and errors.
* **petsc4py 3.21 needs `setuptools<70`** (its build script calls a removed `distutils` argument); the script pins it.
* **FFTW and OpenFOAM's environment.** OpenFOAM only puts a ThirdParty library on the path if it exists when `etc/bashrc` is
  sourced; the script builds OpenFOAM, re-sources, and builds again so `boxTurb`/`noise` link.
* **DAFoam needs two things the guide lists after the OpenFOAM section:** pyOFM and Hisa4DAFoam (its C++ includes Hisa4DAFoam's
  headers). Both are stages in the script.
* **`pkill -f` can kill your own shell** if the pattern appears in its command line.

## Measurement scripts

* `fd_study.py`: adjoint against finite differences over a range of step sizes (the right way to validate adjoint gradients).
* `timing.py`, `profile_totals.py`: wall-clock and profile of one optimization iteration for a given number of ranks.
