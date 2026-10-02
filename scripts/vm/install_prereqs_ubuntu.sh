#!/usr/bin/env bash
# DAFoam's documented Ubuntu prerequisites plus python3-venv. Needs sudo. (The Lima VM runs the same package list from
# scripts/vm/lima-dafoam.yaml; keep the two in step.) Tested: Ubuntu 24.04 aarch64. Not tested: x86-64, other releases.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates cmake flex bison libfl-dev libcgal-dev libopenmpi-dev openmpi-bin \
  libscotch-dev libreadline-dev libncurses-dev wget vim git lcov patchelf pkg-config swig gfortran \
  libxrender1 libxml2-dev libegl1 curl python3-venv python3-dev python3-pip rsync
