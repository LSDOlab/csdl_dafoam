#!/usr/bin/env bash
# Mirrors DAFoam's "Compile from source" guide: CGNS 4.5.0 (static, Fortran, no HDF5), then
# IDWarp against conda's PETSc.
set -euo pipefail

# ---- CGNS: static library linked into IDWarp's extension, so no run-time CGNS dependency ----
export CGNS_HOME="$SRC_DIR/cgns-install"
mkdir -p "$SRC_DIR/cgns-src/build"
(
  cd "$SRC_DIR/cgns-src/build"
  cmake .. \
    -DCGNS_ENABLE_FORTRAN=1 \
    -DCMAKE_INSTALL_PREFIX="$CGNS_HOME" \
    -DCGNS_BUILD_CGNSTOOLS=0 \
    -DCGNS_ENABLE_HDF5=OFF \
    -DCGNS_ENABLE_64BIT=OFF \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_Fortran_FLAGS="-fPIC"
  make -j"${CPU_COUNT:-2}" all install
)

# ---- IDWarp ----
export PETSC_DIR="$PREFIX"
export PETSC_ARCH=""

cd "$SRC_DIR"
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
sed -i 's/mpifort/mpif90/g' config/config.mk
# conda's Fortran/C compilers are picked up through the MPI wrappers
export OMPI_FC="$FC" OMPI_CC="$CC" OMPI_CXX="$CXX"

make

"$PYTHON" -m pip install . --no-deps --no-build-isolation -vv
