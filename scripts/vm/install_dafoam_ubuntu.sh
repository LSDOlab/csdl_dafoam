#!/usr/bin/env bash
# Build DAFoam v5.1.1 (with OpenFOAM v2506 and the reverse-mode AD libraries) on Ubuntu 24.04, x86-64
# or ARM64, following https://dafoam.github.io/installation-source.html, in one resumable script.
#
# Differences from the guide, all because csdl_dafoam does not need them:
#   * (the guide's order: OpenFOAM, OpenFOAM-AD, pyOFM, Hisa4DAFoam, then DAFoam: Hisa4DAFoam is a build
#     dependency of DAFoam's C++, not an optional extra)
#   * no Fortran IDWarp / CGNS / pyHyp / pyOptSparse / MACH framework (IDWarp-JAX replaces IDWarp);
#   * a plain Python venv instead of Miniconda (Ubuntu 24.04 ships Python 3.12).
#
# Run inside the VM or on the Linux machine (after install_prereqs_ubuntu.sh):
#   bash install_dafoam_ubuntu.sh      about 2 hours on 12 cores, 11 GB; safe to re-run, finished stages are skipped
# Log:                 $DAFOAM_ROOT_PATH/build.log
set -eo pipefail

export DAFOAM_ROOT_PATH="${DAFOAM_ROOT_PATH:-$HOME/dafoam}"
NPROC="${NPROC:-$(nproc)}"
STAGES="$DAFOAM_ROOT_PATH/.stages"
mkdir -p "$DAFOAM_ROOT_PATH"/{packages,repos,OpenFOAM/sharedBins,OpenFOAM/sharedLibs} "$STAGES"
exec > >(tee -a "$DAFOAM_ROOT_PATH/build.log") 2>&1

LOAD="$DAFOAM_ROOT_PATH/loadDAFoam.sh"
CURRENT=""
rm -f "$DAFOAM_ROOT_PATH/FAILED" "$DAFOAM_ROOT_PATH/DONE"
# An EXIT trap, not ERR: OpenFOAM's etc/bashrc runs commands that return non-zero as a matter of course,
# which would fire an ERR trap spuriously. Here only the script's final status counts.
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "failed in stage ${CURRENT:-setup} (exit $rc) at $(date)" > "$DAFOAM_ROOT_PATH/FAILED"; echo "!! FAILED in stage ${CURRENT:-setup} (exit $rc)"; fi' EXIT
stage() { # stage <name> <function>: run once, remember success
  CURRENT="$1"
  if [ -f "$STAGES/$1" ]; then echo "== [$1] already done"; return; fi
  echo "== [$1] $(date)"; "$2"; touch "$STAGES/$1"; echo "== [$1] done $(date)"
}
load() { set +eu; . "$LOAD"; set -e; }   # OpenFOAM's bashrc is not safe under set -u
fetch() {  # fetch <url> <output file> <sha256>: download and verify, so a changed or damaged file stops the build
  wget -q "$1" -O "$2"
  echo "$3  $2" | sha256sum -c --quiet - || { echo "!! checksum mismatch for $2 (from $1)"; exit 1; }
}

# ---------------------------------------------------------------------------------------------
write_loader() {
  cat > "$LOAD" <<EOF
#!/bin/bash
export DAFOAM_ROOT_PATH=$DAFOAM_ROOT_PATH
export VIRTUAL_ENV=\$DAFOAM_ROOT_PATH/venv
export PATH=\$VIRTUAL_ENV/bin:\$PATH
export PETSC_DIR=\$DAFOAM_ROOT_PATH/packages/petsc-3.21.6
export PETSC_ARCH=real-opt
export PETSC_LIB=\$PETSC_DIR/\$PETSC_ARCH/lib
export LD_LIBRARY_PATH=\$PETSC_LIB:\${LD_LIBRARY_PATH:-}
EOF
  # this function runs at the start of every run (resumes included), so OpenFOAM's lines must be part of it
  # whenever OpenFOAM is already installed, not only on the run that builds it
  if [ -d "$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc" ]; then
    cat >> "$LOAD" <<EOF
source \$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc/bashrc
export LD_LIBRARY_PATH=\$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:\$LD_LIBRARY_PATH
export PATH=\$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:\$PATH
EOF
  fi
  chmod 755 "$LOAD"
}

python_env() {
  python3 -m venv "$DAFOAM_ROOT_PATH/venv"
  load
  # petsc4py 3.21's build script calls distutils' execute(dry_run=...), removed from newer setuptools
  # (the guide gets an old one from Miniconda), so pin it
  pip install --upgrade pip wheel "setuptools<70"
  pip install numpy==2.3.5 scipy==1.18.1 cython==3.0.5 openmdao==3.45.1
  # mpi4py from source against the system OpenMPI (more robust than a wheel, as the guide says)
  cd "$DAFOAM_ROOT_PATH/packages"
  fetch https://github.com/mpi4py/mpi4py/releases/download/4.1.1/mpi4py-4.1.1.tar.gz mpi4py-4.1.1.tar.gz eb2c8489bdbc47fdc6b26ca7576e927a11b070b6de196a443132766b3d0a2a22
  tar xf mpi4py-4.1.1.tar.gz && cd mpi4py-4.1.1 && pip install .
}

petsc() {
  load; cd "$DAFOAM_ROOT_PATH/packages"
  if [ ! -f "$PETSC_DIR/real-opt/lib/libpetsc.so" ]; then
    [ -d petsc-3.21.6 ] || { fetch https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.6.tar.gz petsc-3.21.6.tar.gz cb2dc00742a89cf8acf9ff8aae189e6864e8b90f4997f087be6e54ff39c30d74
                             tar xf petsc-3.21.6.tar.gz; }
    cd petsc-3.21.6
    ./configure --PETSC_ARCH=real-opt --with-scalar-type=real --with-debugging=0 \
      --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes \
      --download-fblaslapack=yes --download-f2cblaslapack=yes --with-shared-libraries=yes \
      --with-fortran-bindings=1 --with-cxx-dialect=C++11 --with-make-np="$NPROC"
    make PETSC_DIR="$PETSC_DIR" PETSC_ARCH=real-opt all
  fi
}

petsc4py() {
  load
  cd "$PETSC_DIR/src/binding/petsc4py" && pip install . --no-build-isolation
  python -c "from petsc4py import PETSc; print('petsc4py', PETSc.Sys.getVersion())"
}

openfoam_original() {
  cd "$DAFOAM_ROOT_PATH/OpenFOAM"
  if [ ! -d OpenFOAM-v2506 ]; then   # skipped when resuming, so built objects are kept
    fetch "https://sourceforge.net/projects/openfoam/files/v2506/OpenFOAM-v2506.tgz/download" OpenFOAM-v2506.tgz 63d26f48ae7ee9a7806a0ceb339ef8a0ba485a4714d54fbfb31e78e1a4849965
    fetch "https://sourceforge.net/projects/openfoam/files/v2506/ThirdParty-v2506.tgz/download" ThirdParty-v2506.tgz f36ab81ba9e94b4771e27ad1c439bfd377d2a16b2919e7d969bc755ddf9eb0cc
    tar xf OpenFOAM-v2506.tgz && tar xf ThirdParty-v2506.tgz && rm -f OpenFOAM-v2506.tgz ThirdParty-v2506.tgz
    cd OpenFOAM-v2506
    fetch https://github.com/DAFoam/files/releases/download/v1.0.0/UPstream_OF2506_Patch.C UPstream_OF2506_Patch.C d4f221d793ed250ae02b9e4d3a0f8c5b4768e0d8dd65b7993e6ba1f20e7086e8
    mv UPstream_OF2506_Patch.C src/Pstream/mpi/UPstream.C
  fi
  # the guide appends these to loadDAFoam.sh
  grep -q "OpenFOAM-v2506/etc/bashrc" "$LOAD" || cat >> "$LOAD" <<EOF
source \$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc/bashrc
export LD_LIBRARY_PATH=\$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:\$LD_LIBRARY_PATH
export PATH=\$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:\$PATH
EOF
  # OpenFOAM only puts a ThirdParty library (FFTW, ...) on the library path if it already exists when
  # etc/bashrc is sourced. A first run builds FFTW *during* Allwmake, after the environment was set, so
  # boxTurb/noise then fail to link. Hence: build, re-source, build again (the second pass is quick).
  export WM_QUIET=true WM_NCOMPPROCS="$NPROC"
  load; ( cd "$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506" && ./Allwmake ) || echo "first Allwmake pass incomplete; re-sourcing the environment and retrying"
  load; ( cd "$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506" && ./Allwmake )
  simpleFoam -help | head -3
}

openfoam_ad() {
  load
  cd "$DAFOAM_ROOT_PATH/OpenFOAM"
  # pinned to the tip of branch v2506-ad as built and verified (2026-01-28), not to the moving branch
  fetch https://github.com/DAFoam/OpenFOAM-AD/archive/2e519c1d4d5ad32cb2f42b7298f9087ca243b9c3.tar.gz OpenFOAM-AD.tgz 062bc1fd80419fac0cc3c96a37704ace3c3161141f6ab1e49ea505fda5b441de
  tar xf OpenFOAM-AD.tgz && mv OpenFOAM-AD-* OpenFOAM-AD && rm OpenFOAM-AD.tgz
  cd OpenFOAM-AD
  sed -i 's/export WM_AD_MODE=.*/export WM_AD_MODE=ADR/g' etc/bashrc
  (
    set +eu; . etc/bashrc; set -e   # a fresh OpenFOAM environment for the AD tree
    export WM_QUIET=true WM_NCOMPPROCS="$NPROC"
    ./Allwmake
    simpleFoamADR -help | head -3
  )
  # link the AD libraries into the original tree with relative links (portable)
  ./renameAD.sh platforms/linux*ADR --ADR --commit
  load
  cd "$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506"/platforms/*/lib
  ln -sf ../../../../OpenFOAM-AD/platforms/linux*ADR/lib/*.so .
  ( cd dummy && ln -sf ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/dummy/*.so . )
  ( cd "$FOAM_MPI" && ln -sf ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/"$FOAM_MPI"/*.so . )
}

pyofm() {
  load
  cd "$DAFOAM_ROOT_PATH/repos"
  [ -d pyofm-1.2.3 ] || { fetch https://github.com/mdolab/pyofm/archive/refs/tags/v1.2.3.tar.gz pyofm.tar.gz 414146cf88fbf07e43276df15d64db46484601ade49cb595c399d132a4d96dfc; tar xf pyofm.tar.gz; }
  cd pyofm-1.2.3 && make && pip install .
}

hisa4dafoam() {  # DAFoam's include path reads OpenFOAM/Hisa4DAFoam, so this must exist before DAFoam builds
  load
  cd "$DAFOAM_ROOT_PATH/OpenFOAM"
  if [ ! -d Hisa4DAFoam ]; then   # pinned to the commit that was built and verified (2026-05-11)
    git init -q Hisa4DAFoam && git -C Hisa4DAFoam remote add origin https://github.com/DAFoam/Hisa4DAFoam
    git -C Hisa4DAFoam fetch -q --depth 1 origin 73ced10b6dffc8aa2fb5433a787c74dc21f313da
    git -C Hisa4DAFoam checkout -q FETCH_HEAD
  fi
  cd Hisa4DAFoam && ./Allmake
}

dafoam() {
  load
  cd "$DAFOAM_ROOT_PATH/repos"
  [ -d dafoam-5.1.1 ] || { fetch https://github.com/mdolab/dafoam/archive/refs/tags/v5.1.1.tar.gz dafoam.tgz 235fca28541754b8775b60575f4cb9e4192480b776b9662f38c11bba85c35639; tar xf dafoam.tgz; }
  cd dafoam-5.1.1
  export WM_QUIET=true WM_NCOMPPROCS="$NPROC"
  ./Allmake
  python -c "import dafoam; print('dafoam imported from', dafoam.__file__)"
}

write_loader
stage 1-python python_env
stage 2-petsc petsc
stage 2b-petsc4py petsc4py
stage 3-openfoam openfoam_original
stage 4-openfoam-ad openfoam_ad
stage 5a-pyofm pyofm
stage 5b-hisa4dafoam hisa4dafoam
stage 5-dafoam dafoam
echo "ALL DONE $(date). Use:  . $LOAD"
touch "$DAFOAM_ROOT_PATH/DONE"
