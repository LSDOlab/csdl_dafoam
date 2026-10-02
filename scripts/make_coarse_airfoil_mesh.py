"""Generate a very coarse one-cell-thick O-grid for the bundled NACA 0012 as an OpenFOAM ``polyMesh``, by coarsening it.

    python scripts/make_coarse_airfoil_mesh.py [--i-stride 3] [--j-stride 2] --out DIR/constant/polyMesh

Why: the bundled mesh has 4,032 cells, so a real flow solve takes seconds and an adjoint tens of seconds, too slow for a test
that should run every time. This mesh has a few hundred cells. It is for *testing the interface quickly*; its solutions are not
accurate.

How: the bundled mesh is a structured O-grid, 126 cells around the airfoil by 32 layers outward. This script recovers that
(i, j) structure from the mesh connectivity (ring 0 is the wall loop; ring j+1 is each ring-j point's outward neighbour) and keeps
every ``i-stride``-th point around and every ``j-stride``-th ring, always keeping the trailing-edge base points and the leading
edge. Every coarse point is therefore a point of the original mesh: the wall points lie on the STEP geometry, so the projection
works unchanged; the blunt trailing edge is kept; and the cells are unions of good original cells, so there is no new grid
generation that could fold at the trailing edge. The patches are those of the bundled case (``wing``, ``inout``, ``symmetry1``,
``symmetry2``), so its ``0/``, ``system/`` and ``constant/`` dictionaries apply unchanged.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np

from csdl_dafoam import cases, foam_io

SPAN = 0.1


def structured_rings(case_dir):
    """Point ids of the bundled O-grid at y = 0 as an ``(n_around, n_rings)`` array and the points, wall ring first.

    ``i`` runs from the leading edge along the upper surface; ``j`` runs from the wall outward.
    """
    mesh = foam_io.read_mesh(case_dir)
    points = mesh["points"]
    plane = mesh["patches"]["symmetry1"]
    if not np.allclose(points[np.unique(np.concatenate(mesh["faces"][plane["startFace"] : plane["startFace"] + plane["nFaces"]])), 1], 0.0):
        raise RuntimeError("symmetry1 is expected to lie in the y = 0 plane")
    neighbours = {}
    for face in mesh["faces"][plane["startFace"] : plane["startFace"] + plane["nFaces"]]:
        for k in range(4):
            a, b = int(face[k]), int(face[(k + 1) % 4])
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)

    wall = mesh["patches"]["wing"]
    wall_edges = [tuple(int(i) for i in f if abs(points[i, 1]) < 1e-12) for f in mesh["faces"][wall["startFace"] : wall["startFace"] + wall["nFaces"]]]
    ring_neighbours = {}
    for a, b in wall_edges:
        ring_neighbours.setdefault(a, []).append(b)
        ring_neighbours.setdefault(b, []).append(a)
    start = min(ring_neighbours, key=lambda i: points[i, 0])  # leading edge
    ring0, previous, current = [start], None, start
    while True:
        nxt = [n for n in ring_neighbours[current] if n != previous]
        if not nxt or nxt[0] == start:
            break
        ring0.append(nxt[0])
        previous, current = current, nxt[0]
    ring0 = np.array(ring0)
    if points[ring0[1], 2] < points[ring0[0], 2] and points[ring0[-1], 2] > points[ring0[0], 2]:
        ring0 = np.concatenate([ring0[:1], ring0[:0:-1]])  # upper surface first
    if len(ring0) != wall["nFaces"]:
        raise RuntimeError("wall loop does not close")

    rings = [ring0]
    seen = set(ring0.tolist())
    while True:
        outward = []
        for pid in rings[-1]:
            candidates = [n for n in neighbours[int(pid)] if n not in seen or n in outward]
            candidates = [n for n in neighbours[int(pid)] if n not in seen]
            if len(candidates) != 1:
                break
            outward.append(candidates[0])
        else:
            rings.append(np.array(outward))
            seen |= set(outward)
            continue
        break
    return np.array(rings).T, points


def choose_indices(n_around, n_rings, i_stride, j_stride, base=(61, 63, 65)):
    """Indices to keep: every ``i_stride``-th point (symmetric about the trailing edge), leading edge, the trailing-edge base."""
    base_lo, base_hi = min(base), max(base)
    keep = {0} | set(base)
    keep |= {i for i in range(n_around) if i % i_stride == 0 and not (base_lo - 2 <= i <= base_hi + 2)}
    i_keep = sorted(keep)
    j_keep = sorted(set(range(0, n_rings, j_stride)) | {n_rings - 1})
    return i_keep, j_keep


def cell_areas(grid):
    a, b = grid[:-1, :-1], grid[1:, :-1]
    c, d = grid[1:, 1:], grid[:-1, 1:]
    return 0.5 * ((a[..., 0] * b[..., 1] - b[..., 0] * a[..., 1]) + (b[..., 0] * c[..., 1] - c[..., 0] * b[..., 1])
                  + (c[..., 0] * d[..., 1] - d[..., 0] * c[..., 1]) + (d[..., 0] * a[..., 1] - a[..., 0] * d[..., 1]))


def periodic(grid):
    return np.concatenate([grid, grid[:1]], axis=0)


def write_polymesh(grid, directory):
    n, jp1 = grid.shape[:2]
    layers = jp1 - 1

    def pid(i, j, k):
        return k * n * jp1 + j * n + (i % n)

    points = np.zeros((2 * n * jp1, 3))
    for k, y in enumerate((0.0, SPAN)):
        for j in range(jp1):
            for i in range(n):
                points[pid(i, j, k)] = [grid[i, j, 0], y, grid[i, j, 1]]

    def cell(i, j):
        return j * n + (i % n)

    centroids = {}
    for j in range(layers):
        for i in range(n):
            ids = [pid(i, j, 0), pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i, j + 1, 0), pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)]
            centroids[cell(i, j)] = points[ids].mean(axis=0)

    def oriented(face, owner, neighbour_centroid):
        """Order ``face`` so its normal points out of ``owner`` (towards the neighbour, or away for boundary faces)."""
        p = points[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0]) + np.cross(p[2] - p[0], p[3] - p[0])
        direction = (neighbour_centroid - centroids[owner]) if neighbour_centroid is not None else (p.mean(axis=0) - centroids[owner])
        return face if np.dot(normal, direction) > 0 else face[::-1]

    internal = []
    for j in range(layers):
        for i in range(n):
            if True:  # circumferential face between (i-1, j) and (i, j)
                a, b = cell(i - 1, j), cell(i, j)
                face = [pid(i, j, 0), pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i, j, 1)]
                owner, neigh = min(a, b), max(a, b)
                internal.append((owner, neigh, oriented(face, owner, centroids[neigh])))
            if j > 0:  # radial face between (i, j-1) and (i, j)
                a, b = cell(i, j - 1), cell(i, j)
                face = [pid(i, j, 0), pid(i + 1, j, 0), pid(i + 1, j, 1), pid(i, j, 1)]
                owner, neigh = min(a, b), max(a, b)
                internal.append((owner, neigh, oriented(face, owner, centroids[neigh])))
    internal.sort(key=lambda t: (t[0], t[1]))

    patches = {"symmetry1": [], "symmetry2": [], "wing": [], "inout": []}
    for j in range(layers):
        for i in range(n):
            c = cell(i, j)
            patches["symmetry1"].append((c, oriented([pid(i, j, 0), pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i, j + 1, 0)], c, None)))
            patches["symmetry2"].append((c, oriented([pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)], c, None)))
    for i in range(n):
        patches["wing"].append((cell(i, 0), oriented([pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)], cell(i, 0), None)))
        patches["inout"].append((cell(i, layers - 1), oriented([pid(i, layers, 0), pid(i + 1, layers, 0), pid(i + 1, layers, 1), pid(i, layers, 1)], cell(i, layers - 1), None)))

    faces, owner, neighbour = [], [], []
    for o, nb, f in internal:
        faces.append(f), owner.append(o), neighbour.append(nb)
    n_internal = len(faces)
    boundary = []
    for name in ("symmetry1", "symmetry2", "wing", "inout"):
        start = len(faces)
        for o, f in sorted(patches[name], key=lambda t: t[0]):
            faces.append(f), owner.append(o)
        boundary.append((name, start, len(faces) - start))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n_cells = n * layers
    header = lambda cls, obj, note="": (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n%s    location    \"constant/polyMesh\";\n    object      %s;\n}\n\n"
        % (cls, f'    note        "{note}";\n' if note else "", obj)
    )
    note = f"nPoints:{len(points)}  nCells:{n_cells}  nFaces:{len(faces)}  nInternalFaces:{n_internal}"

    def write(name, text, gz=True):
        if gz:
            with gzip.open(directory / f"{name}.gz", "wt") as handle:
                handle.write(text)
        else:
            (directory / name).write_text(text)

    write("points", header("vectorField", "points") + f"{len(points)}\n(\n" + "\n".join("(%.17g %.17g %.17g)" % tuple(p) for p in points) + "\n)\n")
    write("faces", header("faceList", "faces") + f"{len(faces)}\n(\n" + "\n".join("4(%d %d %d %d)" % tuple(f) for f in faces) + "\n)\n")
    write("owner", header("labelList", "owner", note) + f"{len(owner)}\n(\n" + "\n".join(map(str, owner)) + "\n)\n")
    write("neighbour", header("labelList", "neighbour", note) + f"{len(neighbour)}\n(\n" + "\n".join(map(str, neighbour)) + "\n)\n")
    types = {"symmetry1": ("symmetry", "symmetry"), "symmetry2": ("symmetry", "symmetry"), "wing": ("wall", "wall"), "inout": ("patch", None)}
    body = ""
    for name, start, count in boundary:
        kind, group = types[name]
        groups = f"        inGroups        1({group});\n" if group else ""
        body += f"    {name}\n    {{\n        type            {kind};\n{groups}        nFaces          {count};\n        startFace       {start};\n    }}\n"
    write("boundary", header("polyBoundaryMesh", "boundary") + f"{len(boundary)}\n(\n{body})\n", gz=False)
    return n_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-stride", type=int, default=3, help="keep every n-th point around the airfoil")
    parser.add_argument("--j-stride", type=int, default=2, help="keep every n-th ring outward")
    parser.add_argument("--out", default=None, help="directory to write constant/polyMesh files into (default: statistics only)")
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        case = cases.copy_case("naca0012", Path(tmp) / "c", nprocs=1)
        ids, points = structured_rings(case)
    n_around, n_rings = ids.shape
    if (n_around, n_rings) != (126, 33):
        raise RuntimeError(f"unexpected O-grid shape {ids.shape}; this script is written for the bundled 126 x 32 mesh")
    i_keep, j_keep = choose_indices(n_around, n_rings, args.i_stride, args.j_stride)
    selected = ids[np.ix_(i_keep, j_keep)]
    grid = points[selected][:, :, [0, 2]]
    areas = cell_areas(periodic(grid))
    if areas.mean() < 0:
        grid = np.concatenate([grid[:1], grid[:0:-1]])
        areas = cell_areas(periodic(grid))
    print(f"{len(i_keep)} points around x {len(j_keep) - 1} layers = {len(i_keep) * (len(j_keep) - 1)} cells; "
          f"cell areas min {areas.min():.3e} max {areas.max():.3e}; inverted: {(areas <= 0).sum()}")
    if args.out:
        write_polymesh(grid, Path(args.out))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
