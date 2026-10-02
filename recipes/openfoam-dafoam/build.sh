#!/usr/bin/env bash
# Translation of DAFoam's "Compile from source" OpenFOAM steps (original build, then the
# reverse-mode AD build linked back into the original tree) to a conda build.
# Reference: https://dafoam.github.io/installation-source.html
set -eo pipefail

# OpenFOAM's bashrc is not safe under `set -e` / `set -u`; source it with both relaxed, then
# verify it actually loaded.
load_foam_env() {
  set +e +u
  # shellcheck disable=SC1090
  . "$1"
  set -e
  if [ -z "${WM_PROJECT_DIR:-}" ]; then
    echo "ERROR: OpenFOAM environment did not load from $1" >&2
    exit 1
  fi
}

export DAFOAM_ROOT_PATH="$PREFIX/opt/dafoam"
FOAM_DIR="$DAFOAM_ROOT_PATH/OpenFOAM"
mkdir -p "$FOAM_DIR/sharedBins" "$FOAM_DIR/sharedLibs" "$DAFOAM_ROOT_PATH/packages" "$DAFOAM_ROOT_PATH/repos"

# ---- unprefixed compiler names: wmake's rules call plain gcc / g++ -------------------------
mkdir -p "$SRC_DIR/_shims"
ln -sf "$CC"  "$SRC_DIR/_shims/gcc"
ln -sf "$CXX" "$SRC_DIR/_shims/g++"
export PATH="$SRC_DIR/_shims:$PATH"

# ---- place the sources ---------------------------------------------------------------------
mv "$SRC_DIR/openfoam"    "$FOAM_DIR/OpenFOAM-v2506"
mv "$SRC_DIR/thirdparty"  "$FOAM_DIR/ThirdParty-v2506"
mv "$SRC_DIR/openfoam-ad" "$FOAM_DIR/OpenFOAM-AD"
cp "$SRC_DIR/patch/UPstream_OF2506_Patch.C" "$FOAM_DIR/OpenFOAM-v2506/src/Pstream/mpi/UPstream.C"

# ---- the loader DAFoam's Allmake sources (conda-flavoured loadDAFoam.sh) ------------------------
# Paths are written relative to $DAFOAM_ROOT_PATH, which activate.sh derives from $CONDA_PREFIX, so
# the file survives conda's prefix relocation.
cat > "$DAFOAM_ROOT_PATH/loadDAFoam.sh" <<'EOF'
#!/bin/bash
# DAFoam environment for the conda package set. Source from bash (OpenFOAM's bashrc is bash/zsh only).
export DAFOAM_ROOT_PATH="${DAFOAM_ROOT_PATH:-${CONDA_PREFIX}/opt/dafoam}"

# PETSc comes from conda (an install, not a build tree: no PETSC_ARCH)
export PETSC_DIR="${CONDA_PREFIX}"
export PETSC_ARCH=""
export PETSC_LIB="${CONDA_PREFIX}/lib"

# OpenFOAM
source "$DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc/bashrc"
export LD_LIBRARY_PATH="$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:$LD_LIBRARY_PATH"
export PATH="$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:$PATH"
EOF
chmod 755 "$DAFOAM_ROOT_PATH/loadDAFoam.sh"

# The ThirdParty/OpenFOAM builds below must see PETSc paths only through the loader above.
export CONDA_PREFIX="$PREFIX"
export WM_QUIET=true
export WM_NCOMPPROCS="${CPU_COUNT:-2}"

# ---- 1) original OpenFOAM ------------------------------------------------------------------
(
  load_foam_env "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
  cd "$FOAM_DIR/OpenFOAM-v2506"
  ./Allwmake
)

# ---- 2) reverse-mode AD OpenFOAM (CoDiPack) -----------------------------------------------------
(
  load_foam_env "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
  cd "$FOAM_DIR/OpenFOAM-AD"
  sed -i 's/export WM_AD_MODE=.*/export WM_AD_MODE=ADR/g' etc/bashrc
  load_foam_env etc/bashrc
  ./Allwmake
  ./renameAD.sh platforms/linux*ADR --ADR --commit
)

# ---- 3) link the AD libraries into the original tree with relative links (portable) ---------------
(
  load_foam_env "$DAFOAM_ROOT_PATH/loadDAFoam.sh"
  cd "$FOAM_DIR/OpenFOAM-v2506"/platforms/*/lib
  ln -sf ../../../../OpenFOAM-AD/platforms/linux*ADR/lib/*.so .
  ( cd dummy && ln -sf ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/dummy/*.so . )
  ( cd "$FOAM_MPI" && ln -sf ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/"$FOAM_MPI"/*.so . )
)

# ---- 4) shrink the package: object files and build trees are not needed at run time --------------
( cd "$FOAM_DIR/OpenFOAM-v2506" && find . -name '*.o' -delete -o -name '*.dep' -delete && rm -rf build ../ThirdParty-v2506/build )
( cd "$FOAM_DIR/OpenFOAM-AD"    && find . -name '*.o' -delete -o -name '*.dep' -delete && rm -rf build )

# ---- activation ----------------------------------------------------------------------------------
mkdir -p "$PREFIX/etc/conda/activate.d"
cp "$SRC_DIR/scripts/activate.sh" "$PREFIX/etc/conda/activate.d/openfoam-dafoam.sh"
