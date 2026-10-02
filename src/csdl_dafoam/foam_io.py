"""Minimal reader for ASCII OpenFOAM ``polyMesh`` files.

Lets you inspect a case's mesh (to test a geometry projection, check mesh quality after a
warp, or build the mesh dictionary IDWarp-JAX consumes) without an OpenFOAM installation. Only ASCII ``points``/``faces``/``boundary`` files, plain
or gzip-compressed, are supported.
"""
import gzip
import re
from pathlib import Path

import numpy as np

__all__ = ["read_patch_points", "list_patches", "read_mesh", "cell_volumes", "idwarp_jax_mesh_dict"]


def _read_text(directory, name):
    plain = Path(directory) / name
    if plain.is_file():
        return plain.read_text()
    zipped = Path(directory) / f"{name}.gz"
    if zipped.is_file():
        with gzip.open(zipped, "rt") as handle:
            return handle.read()
    raise FileNotFoundError(f"neither {plain} nor {zipped} exists")


def _list_body(text):
    """Return the lines between the ``N\\n(`` and matching ``)`` of a list-valued file."""
    match = re.search(r"\n(\d+)\s*\n\(", text)
    if match is None:
        raise ValueError("not a list-valued OpenFOAM file")
    count = int(match.group(1))
    body = text[match.end() : text.rindex(")")].strip().splitlines()
    if len(body) != count:
        raise ValueError(f"expected {count} entries, found {len(body)}")
    return body


def list_patches(case_dir):
    """Return ``{patch_name: {"type": ..., "nFaces": ..., "startFace": ...}}``."""
    text = _read_text(Path(case_dir) / "constant" / "polyMesh", "boundary")
    patches = {}
    for name, body in re.findall(r"\n\s*(\w+)\s*\{([^}]*)\}", text):
        fields = dict(re.findall(r"(\w+)\s+([^;]+);", body))
        if "nFaces" in fields:
            patches[name] = {
                "type": fields.get("type"),
                "nFaces": int(fields["nFaces"]),
                "startFace": int(fields["startFace"]),
            }
    return patches


def read_patch_points(case_dir, patch):
    """Unique points of boundary ``patch`` as an ``(n, 3)`` array, sorted by point index."""
    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    info = list_patches(case_dir)[patch]

    points = np.array([[float(v) for v in line.strip().strip("()").split()] for line in _list_body(_read_text(poly_mesh, "points"))])
    faces = _list_body(_read_text(poly_mesh, "faces"))
    patch_faces = faces[info["startFace"] : info["startFace"] + info["nFaces"]]
    indices = sorted({int(i) for face in patch_faces for i in re.findall(r"\d+", face)[1:]})
    return points[indices]


def _int_list(text):
    match = re.search(r"\n(\d+)\s*\n\(", text)
    return np.array(text[match.end() : text.rindex(")")].split(), dtype=np.int64)


def read_mesh(case_dir):
    """Read ``constant/polyMesh`` of a (reconstructed, not decomposed) case.

    Returns a dict with ``points`` ``(n, 3)``, ``faces`` (list of point-index arrays), ``owner``,
    ``neighbour`` (index arrays) and ``patches`` (see :func:`list_patches`).
    """
    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    points = np.array(
        [[float(v) for v in line.strip().strip("()").split()] for line in _list_body(_read_text(poly_mesh, "points"))]
    )
    faces = [
        np.array(re.findall(r"\d+", line)[1:], dtype=np.int64)
        for line in _list_body(_read_text(poly_mesh, "faces"))
    ]
    return {
        "points": points,
        "faces": faces,
        "owner": _int_list(_read_text(poly_mesh, "owner")),
        "neighbour": _int_list(_read_text(poly_mesh, "neighbour")),
        "patches": list_patches(case_dir),
    }


def cell_volumes(mesh, points=None):
    """Signed cell volumes (divergence theorem over triangle-fan faces).

    ``points`` overrides ``mesh["points"]`` (for example a warped copy). A non-positive entry
    is an inverted or collapsed cell.
    """
    points = mesh["points"] if points is None else np.asarray(points)
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    volumes = np.zeros(owner.max() + 1)
    for face_index, ids in enumerate(mesh["faces"]):
        corners = points[ids]
        centre = corners.mean(axis=0)
        flux = 0.0
        for k in range(len(ids)):
            a, b = corners[k], corners[(k + 1) % len(ids)]
            area_vector = 0.5 * np.cross(a - centre, b - centre)
            flux += np.dot((a + b + centre) / 3.0, area_vector)
        volumes[owner[face_index]] += flux / 3.0
        if face_index < len(neighbour):
            volumes[neighbour[face_index]] -= flux / 3.0
    return volumes


def idwarp_jax_mesh_dict(case_dir, surface_patches, surface_face_type=1):
    """Mesh dictionary for ``idwarp_jax.build_volume_pts_func``.

    Boundary faces only. Faces of ``surface_patches`` get type ``surface_face_type`` (the moving
    surface); every other patch gets a distinct type (2, 3, ...) that IDWarp-JAX ignores.
    Returns ``(mesh_dict, mesh)`` where ``mesh`` is :func:`read_mesh`'s output.
    """
    mesh = read_mesh(case_dir)
    unknown = set(surface_patches) - set(mesh["patches"])
    if unknown:
        raise KeyError(f"patches {sorted(unknown)} not in the case; available: {sorted(mesh['patches'])}")

    other_types = iter(t for t in range(2, 1000) if t != surface_face_type)
    face_points, face_types = [], []
    for name, info in mesh["patches"].items():
        code = surface_face_type if name in surface_patches else next(other_types)
        for face in mesh["faces"][info["startFace"] : info["startFace"] + info["nFaces"]]:
            face_points.append(face.astype(np.int32))
            face_types.append(code)
    mesh_dict = {
        "points": mesh["points"],
        "faces": {"points": face_points, "type": np.array(face_types, dtype=np.int32)},
    }
    return mesh_dict, mesh
