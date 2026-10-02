#!/usr/bin/env bash
# Print the sha256 of every source archive used by recipes/*/recipe.yaml so the TODO placeholders
# can be filled in. Downloads the OpenFOAM tarballs (~0.2 GB), so it is not run automatically.
set -euo pipefail

hash_of() {
  printf '%-34s ' "$1"
  curl -fsSL "$2" | sha256sum | cut -d' ' -f1
}

hash_of "OpenFOAM-v2506.tgz"     "https://sourceforge.net/projects/openfoam/files/v2506/OpenFOAM-v2506.tgz/download"
hash_of "ThirdParty-v2506.tgz"   "https://sourceforge.net/projects/openfoam/files/v2506/ThirdParty-v2506.tgz/download"
hash_of "OpenFOAM-AD (pinned)"   "https://github.com/DAFoam/OpenFOAM-AD/archive/2e519c1d4d5ad32cb2f42b7298f9087ca243b9c3.tar.gz"
