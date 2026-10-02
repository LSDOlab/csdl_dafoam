"""Run OpenFOAM's ``checkMesh`` on a case, optionally with substituted (warped) mesh points.

This is how a warped mesh (for example from IDWarp-JAX) is validated against OpenFOAM itself,
without DAFoam. OpenFOAM is found as ``checkMesh`` on ``PATH`` (an activated conda/DAFoam
environment), or through the macOS app launcher (``openfoam2506`` etc., from OpenFOAM.app).

DAFoam applies its own copy of OpenFOAM's geometry checks (``DACheckGeometry.C``) with these
differences from the stock utility, which :func:`check_mesh` accounts for:

* the stock "cell determinant (wellposedness)" test is **not** applied. A one-cell-thick 2D mesh
  closed by ``symmetry`` patches, like the bundled airfoil, fails it by construction (every cell has
  no internal face with a spanwise normal), yet is a valid DAFoam mesh;
* the thresholds are DAFoam's ``checkMeshThreshold`` option (defaults: aspect ratio 1000,
  non-orthogonality 70, skewness 4), which equal the stock utility's.
"""
import gzip
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["find_openfoam", "write_points", "check_mesh", "MeshCheck"]

#: Failure messages of the stock utility that DAFoam's ``checkGeometry`` does not have.
_NOT_APPLIED_BY_DAFOAM = ("determinant",)


def find_openfoam():
    """Command prefix that runs a shell command inside an OpenFOAM environment, or ``None``.

    Returns a list for ``subprocess`` taking the command string as its last element.
    """
    if shutil.which("checkMesh"):  # already inside an OpenFOAM environment
        return ["bash", "-c"]
    for launcher in ("openfoam2506", "openfoam"):  # OpenFOAM.app (macOS) via Homebrew
        path = shutil.which(launcher)
        if path:
            return [path, "-c"]
    return None


def write_points(case_dir, points):
    """Replace ``constant/polyMesh/points`` of ``case_dir`` with ``points`` (ASCII, full precision)."""
    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    for stale in (poly_mesh / "points", poly_mesh / "points.gz"):
        if stale.exists():
            stale.unlink()
    body = "\n".join("(%.17g %.17g %.17g)" % tuple(p) for p in points)
    header = (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       vectorField;\n"
        '    location    "constant/polyMesh";\n    object      points;\n}\n\n'
    )
    (poly_mesh / "points").write_text(f"{header}{len(points)}\n(\n{body}\n)\n")


@dataclass
class MeshCheck:
    """Result of :func:`check_mesh`."""

    #: failures that DAFoam's own mesh check would also report (empty means DAFoam accepts the mesh)
    failures: list = field(default_factory=list)
    #: stock-utility failures DAFoam does not apply (for example the cell-determinant test)
    ignored: list = field(default_factory=list)
    max_non_orthogonality: float = float("nan")
    max_skewness: float = float("nan")
    max_aspect_ratio: float = float("nan")
    min_volume: float = float("nan")
    output: str = ""

    @property
    def dafoam_ok(self):
        return not self.failures


_FLOAT = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"  # no trailing '.' (checkMesh ends sentences with one)


def _number(pattern, text):
    match = re.search(pattern.replace("{F}", _FLOAT), text)
    return float(match.group(1)) if match else float("nan")


def check_mesh(case_dir, points=None, work_dir=None):
    """Run ``checkMesh -allGeometry -allTopology`` on a copy of ``case_dir``.

    ``points`` (``(n, 3)``) replaces the case's mesh points in the copy; the original is untouched.
    Raises ``RuntimeError`` if no OpenFOAM is available (see :func:`find_openfoam`).
    """
    runner = find_openfoam()
    if runner is None:
        raise RuntimeError("no OpenFOAM found: need checkMesh on PATH or the OpenFOAM.app launcher")

    import tempfile

    own_dir = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="checkmesh_")) if own_dir else Path(work_dir)
    try:
        case = work / "case"
        shutil.copytree(case_dir, case, dirs_exist_ok=True)
        # a failing set write must not need a time directory
        (case / "system").mkdir(exist_ok=True)
        if points is not None:
            write_points(case, points)
        completed = subprocess.run(
            [*runner, f"cd '{case}' && checkMesh -allGeometry -allTopology 2>&1"],
            capture_output=True,
            text=True,
        )
        output = completed.stdout
    finally:
        if own_dir:
            shutil.rmtree(work, ignore_errors=True)

    result = MeshCheck(output=output)
    for line in output.splitlines():
        if "***" in line:
            message = line.strip()
            if any(key in message.lower() for key in _NOT_APPLIED_BY_DAFOAM):
                result.ignored.append(message)
            else:
                result.failures.append(message)
    if "End" not in output:
        result.failures.append("checkMesh did not complete")
    result.max_non_orthogonality = _number(r"Mesh non-orthogonality Max: {F}", output)
    result.max_skewness = _number(r"Max skewness = {F}", output)
    result.max_aspect_ratio = _number(r"Max aspect ratio = {F}", output)
    result.min_volume = _number(r"Min volume = {F}", output)
    return result
