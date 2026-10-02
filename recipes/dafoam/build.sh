#!/usr/bin/env bash
# DAFoam's own Allmake does the work (original + ADR compilation of the adjoint library and
# solvers, then `pip install .`). We only provide the environment it expects.
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
# loadDAFoam.sh (from openfoam-dafoam) sources OpenFOAM and sets PETSC_*; it is re-sourced by Allmake.
load_foam_env "$DAFOAM_ROOT_PATH/loadDAFoam.sh"

# Unprefixed compiler names for wmake (see openfoam-dafoam/build.sh)
mkdir -p "$SRC_DIR/_shims"
ln -sf "$CC"  "$SRC_DIR/_shims/gcc"
ln -sf "$CXX" "$SRC_DIR/_shims/g++"
export PATH="$SRC_DIR/_shims:$PATH"

export WM_QUIET=true
export WM_NCOMPPROCS="${CPU_COUNT:-2}"

# Allmake finishes with a bare `pip install .`, which would resolve dependencies from the network
# and use build isolation; do it offline against the host environment instead.
sed -i 's/^pip install \.$/"$PYTHON" -m pip install . --no-deps --no-build-isolation -vv/' Allmake

./Allmake
