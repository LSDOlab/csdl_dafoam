"""Check that the installation can run a DAFoam case: ``python -m csdl_dafoam.doctor``.

Reports which components import, versions, and whether the OpenFOAM environment is loaded.
Exit status is non-zero if a required component is missing (``--no-fail`` always exits 0).
"""
import argparse
import importlib
import importlib.util
import os
import shutil
import sys

# (import name, what it is, needed for)
_COMPONENTS = [
    ("numpy", "NumPy", "everything"),
    ("scipy", "SciPy", "everything"),
    ("csdl_alpha", "CSDL (csdl_alpha)", "everything"),
    ("lsdo_geo", "lsdo_geo", "geometry parameterization"),
    ("lsdo_function_spaces", "lsdo_function_spaces", "geometry parameterization"),
    ("modopt", "modOpt (LSDOlab's, from git)", "examples: optimization"),
    ("qpsolvers", "qpsolvers", "OpenSQP optimizer"),
    ("quadprog", "quadprog", "OpenSQP optimizer"),
    ("highspy", "highspy (HiGHS)", "OpenSQP optimizer"),
    ("jax", "JAX", "CSDL simulator"),
    ("idwarp_jax", "IDWarp-JAX (LSDOlab's, from git)", "mesh warping (default)"),
    ("mpi4py", "mpi4py", "DAFoam runs"),
    ("petsc4py", "petsc4py", "DAFoam runs"),
    ("idwarp", "IDWarp (Fortran)", "optional: warper='idwarp'"),
    ("dafoam", "DAFoam", "DAFoam runs"),
]


def _version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # broken installs raise more than ImportError (e.g. missing .so)
        return None, f"{type(exc).__name__}: {exc}".splitlines()[0]
    return getattr(module, "__version__", "?"), ""


def check():
    """Return a list of ``(label, ok, detail, needed_for)`` rows."""
    rows = []
    for module_name, label, needed_for in _COMPONENTS:
        if importlib.util.find_spec(module_name) is None:
            rows.append((label, False, "not installed", needed_for))
            continue
        version, error = _version(module_name)
        rows.append((label, version is not None, version if version else error, needed_for))

    foam = os.environ.get("WM_PROJECT_DIR")
    rows.append(("OpenFOAM environment", bool(foam), foam or "WM_PROJECT_DIR is not set (source loadDAFoam.sh)", "DAFoam runs"))
    rows.append(("DAFOAM_ROOT_PATH", bool(os.environ.get("DAFOAM_ROOT_PATH")), os.environ.get("DAFOAM_ROOT_PATH", "not set"), "DAFoam runs"))
    solver = shutil.which("simpleFoam")
    rows.append(("simpleFoam on PATH", bool(solver), solver or "not found", "DAFoam runs"))
    rows.append(("mpirun on PATH", bool(shutil.which("mpirun")), shutil.which("mpirun") or "not found", "DAFoam runs"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-fail", action="store_true", help="always exit 0")
    args = parser.parse_args(argv)

    import csdl_dafoam

    print(f"csdl_dafoam {csdl_dafoam.__version__}  (python {sys.version.split()[0]})")
    rows = check()
    width = max(len(r[0]) for r in rows)
    for label, ok, detail, needed_for in rows:
        print(f"  [{'ok' if ok else '--'}] {label:<{width}}  {detail}" + ("" if ok else f"   (needed for: {needed_for})"))

    required = ("everything", "geometry parameterization", "DAFoam runs", "mesh warping (default)")
    missing = [r for r in rows if not r[1] and r[3] in required]
    if missing:
        print(f"\n{len(missing)} required component(s) missing. See docs/installation.md.")
    return 0 if args.no_fail or not missing else 1


if __name__ == "__main__":
    sys.exit(main())
