# Loads the DAFoam/OpenFOAM environment on `conda activate`. OpenFOAM's bashrc supports bash and
# zsh; other shells only get the root path.
export DAFOAM_ROOT_PATH="${CONDA_PREFIX}/opt/dafoam"
if [ -n "${BASH_VERSION:-}" ] || [ -n "${ZSH_VERSION:-}" ]; then
    # shellcheck disable=SC1091
    . "$DAFOAM_ROOT_PATH/loadDAFoam.sh" > /dev/null 2>&1 || \
        echo "openfoam-dafoam: could not load $DAFOAM_ROOT_PATH/loadDAFoam.sh" >&2
fi
