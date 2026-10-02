"""Access to the bundled geometry and OpenFOAM cases."""
import re
import shutil
from importlib import resources
from pathlib import Path

__all__ = ["list_cases", "geometry_path", "copy_case", "prepare_case"]

#: Written into every case copied by :func:`copy_case`; :func:`prepare_case` only ever deletes
#: directories that carry it.
MARKER = ".csdl_dafoam_case"

# name -> (base case directory, overlay directories applied in order, description)
_CASES = {
    "naca0012": (
        "naca0012",
        (),
        "NACA 0012, compressible RANS (SA + wall function), Re_c ~ 1.5e7, 4032-cell mesh",
    ),
    "naca0012_euler": (
        "naca0012",
        ("naca0012_euler_overlay",),
        "NACA 0012, 'Euler mode': same mesh, mu reduced 1e4x (Re_c ~ 1.5e11)",
    ),
    "naca0012_coarse": (
        "naca0012",
        ("naca0012_euler_overlay", "naca0012_coarse_overlay"),
        "NACA 0012, Euler mode on a 336-cell mesh: for fast tests of the interface, NOT for accurate results",
    ),
}


def _data_dir():
    return Path(str(resources.files("csdl_dafoam") / "data"))


def list_cases():
    """Return ``{case_name: description}`` of the bundled OpenFOAM cases."""
    return {name: entry[2] for name, entry in _CASES.items()}


def geometry_path(name="naca0012"):
    """Path to a bundled STEP geometry (currently only ``"naca0012"``)."""
    path = _data_dir() / "geometry" / f"{name}_unit_span.stp"
    if not path.is_file():
        raise KeyError(f"no bundled geometry named {name!r}")
    return path


def copy_case(name, destination, nprocs=None, overwrite=False):
    """Copy a bundled OpenFOAM case to ``destination`` and return the destination path.

    Parameters
    ----------
    name : str
        One of :func:`list_cases`.
    destination : path-like
        Directory to create. Must not exist unless ``overwrite`` is true (existing files
        are then overwritten, others left alone).
    nprocs : int, optional
        Number of MPI ranks you will run on. DAFoam requires it to equal
        ``numberOfSubdomains`` in ``system/decomposeParDict``, which is rewritten here.
    """
    if name not in _CASES:
        raise KeyError(f"unknown case {name!r}; available: {sorted(_CASES)}")
    base, overlays, _ = _CASES[name]

    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists (pass overwrite=True to reuse it)")

    cases_dir = _data_dir() / "cases"
    shutil.copytree(cases_dir / base, destination, dirs_exist_ok=True)
    for overlay_name in overlays:
        shutil.copytree(cases_dir / overlay_name, destination, dirs_exist_ok=True)

    if nprocs is not None:
        if int(nprocs) < 1:
            raise ValueError("nprocs must be >= 1")
        decompose = destination / "system" / "decomposeParDict"
        text = decompose.read_text()
        new_text, count = re.subn(r"(numberOfSubdomains\s+)\d+(\s*;)", rf"\g<1>{int(nprocs)}\g<2>", text)
        if count != 1:
            raise RuntimeError(f"could not find numberOfSubdomains in {decompose}")
        decompose.write_text(new_text)

    (destination / MARKER).write_text(f"{name}\n")
    return destination


def prepare_case(name, destination, comm=None, clean=True):
    """MPI-safe :func:`copy_case`: rank 0 copies, everyone gets the path.

    ``nprocs`` is taken from the communicator size. With ``clean=True`` a previous copy at
    ``destination`` is deleted first, discarding stale ``processor*`` directories, solution
    time directories and coloring files from earlier runs. Only a directory created by
    :func:`copy_case` (it carries a marker file) is ever deleted; anything else raises.
    """
    destination = Path(destination)
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)

    error = None
    if rank == 0:
        try:
            if destination.exists():
                if not clean:
                    raise FileExistsError(f"{destination} already exists")
                if not (destination / MARKER).is_file():
                    raise FileExistsError(
                        f"{destination} exists and was not created by csdl_dafoam; refusing to delete it"
                    )
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_case(name, destination, nprocs=size)
        except Exception as exc:  # re-raised on every rank below so no rank is left waiting
            error = exc
    if comm is not None:
        error = comm.bcast(error, root=0)
        comm.Barrier()
    if error is not None:
        raise error
    return destination.resolve()
