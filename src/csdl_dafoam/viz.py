"""Reading and plotting flow solutions of an OpenFOAM/DAFoam case, without OpenFOAM tools.

What is here, and how 2D and 3D differ
--------------------------------------
* :func:`read_solution` reads a solution time (``p``, ``U``, ``T`` by default) from a serial case or from a case decomposed
  for MPI (``processor*/<time>``), and returns the deformed mesh together with cell-centered fields on the *global* mesh.
* **2D (airfoil) cases** are one cell thick with ``symmetry`` patches on the two span ends, so a field is exactly a set of
  polygons: the faces of one of those patches, each coloured by its cell's value. :func:`plot_field_2d` draws them with
  matplotlib, with no interpolation. :func:`plot_airfoil_comparison` combines shapes, surface Cp and Mach contours for two
  designs.
* **3D (wing) cases** have no such reduction. :func:`write_patch_vtk` exports a boundary patch (the wing surface) with its
  fields as VTK PolyData for ParaView or PyVista, and :func:`plot_surface_3d` renders it with PyVista when installed. Volume
  slices and iso-surfaces of 3D cases are left to ParaView (the mesh and fields are readable through
  :func:`read_solution`, so a slice is a few lines of NumPy). The 3D functions are unit-tested only on the 2D mesh.

Matplotlib and PyVista are imported only when a plotting function is called.
"""
import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from csdl_dafoam import foam_io

__all__ = [
    "Solution",
    "latest_solution_time",
    "read_solution",
    "mach_number",
    "pressure_coefficient",
    "plot_field_2d",
    "surface_distribution",
    "plot_airfoil_comparison",
    "write_patch_vtk",
    "plot_surface_3d",
    "plot_mesh_2d",
    "read_optimization_history",
    "plot_optimization_history",
    "solution_times",
    "make_movie",
]

#: gas constants of the bundled cases (perfect gas, air)
GAMMA = 1.4
R_AIR = 287.0


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def _text(path):
    path = Path(path)
    if path.is_file():
        return path.read_text()
    gz = path.with_name(path.name + ".gz")
    if gz.is_file():
        with gzip.open(gz, "rt") as handle:
            return handle.read()
    raise FileNotFoundError(f"neither {path} nor {gz} exists")


def _exists(path):
    path = Path(path)
    return path.is_file() or path.with_name(path.name + ".gz").is_file()


def _read_internal_field(path):
    """``internalField`` of a volScalarField/volVectorField as ``("uniform", value)`` or ``("nonuniform", array)``.

    The kind is returned because a nonuniform scalar array and a uniform vector are both one-dimensional.
    """
    text = _text(path)
    match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if match:
        values = [float(v) for v in match.group(1).strip("() \n").split()]
        return "uniform", (values[0] if len(values) == 1 else np.array(values))
    match = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\n?\(", text)
    if match is None:
        raise ValueError(f"cannot parse internalField of {path}")
    count = int(match.group(2))
    body = text[match.end() :]
    if match.group(1) == "scalar":
        values = np.array(body[: body.index("\n)")].split(), dtype=float)
    else:  # vector
        values = np.array(re.findall(r"\(([^()]*)\)", body[: body.index("\n)")]), dtype=object)
        values = np.array([[float(v) for v in row.split()] for row in values])
    if len(values) != count:
        raise ValueError(f"{path}: expected {count} values, found {len(values)}")
    return "nonuniform", values


def _read_labels(path):
    text = _text(path)
    match = re.search(r"\n(\d+)\s*\n\(", text)
    return np.array(text[match.end() : text.rindex(")")].split(), dtype=np.int64)


def _read_points(path):
    text = _text(path)
    match = re.search(r"\n(\d+)\s*\n\(", text)
    rows = text[match.end() : text.rindex(")")].strip().splitlines()
    return np.array([[float(v) for v in row.strip().strip("()").split()] for row in rows])


def _is_time(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def latest_solution_time(case_dir):
    """Name of the integer-numbered time directory with the most recent *flow* solution.

    DAFoam writes the converged primal solution at an integer time (the SIMPLE iteration count) and renames it to
    ``counter * 1e-4`` when an adjoint is computed for it, so the latest solve of the current design is the largest
    integer-named directory containing a pressure field; ``0`` (the initial condition) is used only if there is nothing else.
    """
    case_dir = Path(case_dir)
    roots = sorted(case_dir.glob("processor0")) or [case_dir]
    names = [d.name for d in roots[0].iterdir() if d.is_dir() and re.fullmatch(r"\d+", d.name) and _exists(d / "p")]
    names = [n for n in names if int(n) > 0] or names
    if not names:
        raise FileNotFoundError(f"no solution time directory with a pressure field in {roots[0]}")
    return max(names, key=int)


@dataclass
class Solution:
    """A flow solution on the global mesh (deformed to the design that was solved)."""

    case_dir: Path
    time: str
    mesh: dict  #: :func:`csdl_dafoam.foam_io.read_mesh` output with ``points`` replaced by the deformed ones
    fields: dict = field(default_factory=dict)  #: name -> (n_cells,) or (n_cells, 3) array

    def patch_faces(self, patch):
        info = self.mesh["patches"][patch]
        return slice(info["startFace"], info["startFace"] + info["nFaces"])


def read_solution(case_dir, time=None, fields=("p", "U", "T")):
    """Read a solution time of a serial or MPI-decomposed case.

    ``time`` defaults to :func:`latest_solution_time`. Decomposed cases are reassembled on the global mesh through each
    processor's ``cellProcAddressing`` / ``pointProcAddressing`` (as written by ``decomposePar``), so no OpenFOAM
    installation is needed. The mesh is that of the case's ``constant/polyMesh`` with its points replaced by the deformed
    ones DAFoam wrote into the time directory.
    """
    case_dir = Path(case_dir)
    time = str(time) if time is not None else latest_solution_time(case_dir)
    mesh = foam_io.read_mesh(case_dir)
    n_cells = int(mesh["owner"].max()) + 1
    processors = sorted(case_dir.glob("processor[0-9]*"), key=lambda p: int(p.name[len("processor") :]))

    out = {}
    if not processors:  # serial
        for name in fields:
            out[name] = _expand(_read_internal_field(case_dir / time / name), n_cells)
        if _exists(case_dir / time / "polyMesh" / "points"):
            mesh["points"] = _read_points(case_dir / time / "polyMesh" / "points")
    else:
        points = mesh["points"].copy()
        for processor in processors:
            addressing = _read_labels(processor / "constant" / "polyMesh" / "cellProcAddressing")
            for name in fields:
                value = _expand(_read_internal_field(processor / time / name), len(addressing))
                if name not in out:
                    out[name] = np.full((n_cells,) + value.shape[1:], np.nan)
                out[name][addressing] = value
            if _exists(processor / time / "polyMesh" / "points"):
                point_addressing = _read_labels(processor / "constant" / "polyMesh" / "pointProcAddressing")
                points[point_addressing] = _read_points(processor / time / "polyMesh" / "points")
        mesh["points"] = points
        for name, array in out.items():
            if np.isnan(array).any():
                raise ValueError(f"field {name} has cells that no processor provided")
    return Solution(case_dir=case_dir, time=time, mesh=mesh, fields=out)


def _expand(parsed, n_cells):
    """A parsed ``internalField`` as an ``(n_cells,)`` or ``(n_cells, 3)`` array (uniform values are repeated)."""
    kind, value = parsed
    if kind == "nonuniform":
        return value
    if np.ndim(value) == 0:
        return np.full(n_cells, float(value))
    return np.tile(value, (n_cells, 1))  # a uniform vector


# --------------------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------------------
def mach_number(solution, gamma=GAMMA, gas_constant=R_AIR):
    """Cell-centered Mach number ``|U| / sqrt(gamma R T)``."""
    speed = np.linalg.norm(solution.fields["U"], axis=1)
    return speed / np.sqrt(gamma * gas_constant * solution.fields["T"])


def pressure_coefficient(pressure, p_inf, rho_inf, speed_inf):
    """``(p - p_inf) / (0.5 rho_inf U_inf^2)``."""
    return (np.asarray(pressure) - p_inf) / (0.5 * rho_inf * speed_inf**2)


def _freestream(p_inf, T_inf, speed_inf):
    rho_inf = p_inf / (R_AIR * T_inf)
    return rho_inf, speed_inf


def _midplane_patch(solution):
    """The patch lying in the plane of minimum y (a 2D case's span-end ``symmetry`` patch)."""
    points, faces = solution.mesh["points"], solution.mesh["faces"]
    y_min = points[:, 1].min()
    for name in solution.mesh["patches"]:
        sl = solution.patch_faces(name)
        if all(np.allclose(points[f, 1], y_min) for f in faces[sl][:5]):
            return name
    raise ValueError("no patch lies in the minimum-y plane; is this a 2D (one-cell-thick) case?")


# --------------------------------------------------------------------------------------
# 2D plots
# --------------------------------------------------------------------------------------
def plot_field_2d(
    solution,
    field="Mach",
    ax=None,
    plane_patch=None,
    wall_patch="wing",
    xlim=(-0.25, 1.35),
    zlim=(-0.45, 0.45),
    cmap="turbo",
    vmin=None,
    vmax=None,
    p_inf=101325.0,
    T_inf=300.0,
    speed_inf=238.0,
    colorbar=True,
):
    """Contours of a cell-centered field of a 2D case as exact polygons (one per cell, no interpolation).

    ``field`` is ``"Mach"``, ``"Cp"`` (needs the freestream values, which default to the bundled cases'), or any stored field
    (``"p"``, ``"T"``, or ``"U"`` for the speed). Returns the matplotlib axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    plane_patch = plane_patch or _midplane_patch(solution)
    mesh = solution.mesh
    sl = solution.patch_faces(plane_patch)
    faces = mesh["faces"][sl]
    owners = mesh["owner"][sl]

    if field == "Mach":
        values, label = mach_number(solution), "Mach number"
    elif field == "Cp":
        rho_inf, u_inf = _freestream(p_inf, T_inf, speed_inf)
        values, label = pressure_coefficient(solution.fields["p"], p_inf, rho_inf, u_inf), "$C_p$"
    elif field == "U":
        values, label = np.linalg.norm(solution.fields["U"], axis=1), "|U| [m/s]"
    else:
        values, label = solution.fields[field], field

    polygons = [mesh["points"][f][:, [0, 2]] for f in faces]
    collection = PolyCollection(polygons, array=values[owners], cmap=cmap, edgecolors="face", linewidths=0.3, rasterized=True)  # face-coloured edges hide anti-aliasing seams
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    if colorbar:
        ax.figure.colorbar(collection, ax=ax, label=label, shrink=0.8)

    # airfoil outline on top (the wall patch's faces projected onto the plane)
    if wall_patch in mesh["patches"]:
        wall = mesh["faces"][solution.patch_faces(wall_patch)]
        segments = [mesh["points"][f][:, [0, 2]] for f in wall]
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection(segments, colors="k", linewidths=0.6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("z / c")
    return ax


def surface_distribution(solution, wall_patch="wing", p_inf=101325.0, T_inf=300.0, speed_inf=238.0):
    """Surface pressure coefficient and outline of a 2D airfoil.

    Returns a dict with ``x``, ``z`` (face centres projected on the x-z plane, ordered around the airfoil starting at the
    leading edge along the upper surface), ``cp``, and ``upper`` (boolean, z > camber line). Wall values are those of the
    adjacent cells (the wall pressure condition is ``zeroGradient``).
    """
    mesh = solution.mesh
    sl = solution.patch_faces(wall_patch)
    faces, owners = mesh["faces"][sl], mesh["owner"][sl]
    centres = np.array([mesh["points"][f].mean(axis=0) for f in faces])
    rho_inf, u_inf = _freestream(p_inf, T_inf, speed_inf)
    cp = pressure_coefficient(solution.fields["p"][owners], p_inf, rho_inf, u_inf)
    x, z = centres[:, 0], centres[:, 2]
    # order around the airfoil: leading edge -> upper surface -> trailing edge -> lower surface -> leading edge
    middle = 0.5 * (z[np.argmin(x)] + z[np.argmax(x)])
    upper = z >= middle
    order = np.concatenate([np.flatnonzero(upper)[np.argsort(x[upper])], np.flatnonzero(~upper)[np.argsort(-x[~upper])]])
    return {"x": x[order], "z": z[order], "cp": cp[order], "upper": upper[order]}


FIELD_RANGES = {"Cp": (-1.5, 1.0), "Mach": (0.0, 1.3)}


def plot_airfoil_comparison(solutions, labels=("initial", "optimized"), field="Cp", field_range=None, title=None, **kwargs):
    """Figure comparing designs: contours of ``field`` for each, surface :math:`C_p`, and (for 2+ designs) the airfoil shapes.

    ``solutions`` is a sequence of :class:`Solution` (typically initial and final). ``field`` is ``"Cp"`` (default) or
    ``"Mach"``; ``field_range`` is the colour range (defaults per field). ``kwargs`` are passed to :func:`plot_field_2d` and
    :func:`surface_distribution` (freestream values). Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    n = len(solutions)
    vmin, vmax = field_range or FIELD_RANGES.get(field, (None, None))
    shapes = n > 1
    fig = plt.figure(figsize=(max(8.0, 5.2 * n), 9.2 if shapes else 6.6), constrained_layout=True)
    grid = fig.add_gridspec(3 if shapes else 2, n, height_ratios=[1.0, 0.8, 0.8] if shapes else [1.0, 0.8])
    freestream = {k: v for k, v in kwargs.items() if k in ("p_inf", "T_inf", "speed_inf")}
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    for k, (solution, label) in enumerate(zip(solutions, labels)):
        ax = fig.add_subplot(grid[0, k])
        plot_field_2d(solution, field, ax=ax, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_title(label)

    ax_cp = fig.add_subplot(grid[1, :])
    ax_shape = fig.add_subplot(grid[2, :]) if shapes else None
    for k, (solution, label) in enumerate(zip(solutions, labels)):
        dist = surface_distribution(solution, **freestream)
        ax_cp.plot(dist["x"], dist["cp"], "-", color=colors[k % 4], lw=1.4, label=label)
        if shapes:
            ax_shape.plot(dist["x"], dist["z"], "-", color=colors[k % 4], lw=1.4, label=label)
    ax_cp.invert_yaxis()
    ax_cp.set_xlabel("x / c")
    ax_cp.set_ylabel("$C_p$")
    ax_cp.set_title("surface pressure coefficient (wall-adjacent cells)")
    ax_cp.grid(alpha=0.3)
    if n > 1:
        ax_cp.legend()
    if shapes:
        ax_shape.set_aspect("equal")
        ax_shape.set_xlabel("x / c")
        ax_shape.set_ylabel("z / c")
        ax_shape.set_title("airfoil shape")
        ax_shape.grid(alpha=0.3)
        ax_shape.legend()
    if title:
        fig.suptitle(title)
    return fig


def solution_times(case_dir):
    """All flow-solution times of a case, in order (strings, as the directories are named).

    During an optimization DAFoam keeps the converged solution of every gradient evaluation as time
    ``0.0001``, ``0.0002``, ... (the 1st, 2nd, ... optimizer iteration); a final ``2`` etc. may follow.
    """
    case_dir = Path(case_dir)
    first = sorted(case_dir.glob("processor0")) or [case_dir]
    times = [d.name for d in first[0].iterdir() if d.is_dir() and _is_time(d.name) and _exists(d / "p") and float(d.name) > 0]
    return sorted(times, key=float)


def make_movie(
    case_dir,
    output,
    history,
    table=None,
    field="Cp",
    field_range=None,
    cl_target=None,
    fps=6,
    dpi=100,
    **kwargs,
):
    """Dashboard movie of a 2D optimization, one frame per optimizer iteration, drawn from the solutions DAFoam kept.

    Each frame shows the ``field`` contours, surface :math:`C_p` and airfoil shape of that iteration's flow solution (the
    initial ones in grey), and the histories of CD, CL, and (with ``table``) optimality and feasibility up to that iteration.

    Parameters
    ----------
    case_dir : path
        Case directory holding the solution time directories.
    output : path or list of paths
        ``.gif`` (always possible; Pillow ships with matplotlib) and/or ``.mp4`` (needs ``ffmpeg`` on the PATH).
    history : list of dict
        ``AirfoilModel.history``: ``CD``, ``CL`` and ``time`` per gradient evaluation.
    table : dict, optional
        :func:`read_optimization_history`. With it there is one frame per major iteration (the gradient evaluation of
        iteration *k* is ``ngev[k]``), and optimality and feasibility are plotted. Without it, one frame per gradient evaluation.
    cl_target : float, optional
        Drawn as a line on the CL panel.

    Returns the list of files written.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    history = [e for e in history]
    if not history or any(e.get("time") is None for e in history):
        raise ValueError("history must have a solution 'time' for every entry (AirfoilModel.history from this version)")
    if table is not None and "ngev" in table:
        picks = [min(max(int(g) - 1, 0), len(history) - 1) for g in table["ngev"]]
        iteration = np.asarray(table["major"], dtype=int)
    else:
        picks = list(range(len(history)))
        iteration = np.arange(len(history))
        table = None
    cd = np.array([history[g]["CD"] for g in picks]) * 1e4
    cl = np.array([history[g]["CL"] for g in picks])
    outputs = [output] if isinstance(output, (str, Path)) else list(output)
    vmin, vmax = field_range or FIELD_RANGES.get(field, (None, None))
    freestream = {k: v for k, v in kwargs.items() if k in ("p_inf", "T_inf", "speed_inf")}
    labels = {"Cp": "$C_p$", "Mach": "Mach number"}

    # Explicit layout (inches) so that every edge lines up: the three left panels share one width and one x range (so the x ticks
    # align vertically), the colour bar sits flush against the contour panel, and both columns share their top and bottom edges.
    xlim, zlim = (-0.25, 1.35), (-0.45, 0.45)
    shape_zlim = (-0.1, 0.1)
    fig_w, fig_h = 13.0, 8.0
    left_x, left_w = 0.85, 6.9
    top, bottom, gap = fig_h - 0.45, 0.7, 0.15
    h_field = left_w * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])  # equal aspect fills the box exactly
    h_shape = left_w * (shape_zlim[1] - shape_zlim[0]) / (xlim[1] - xlim[0])

    def box(x, y, w, h):
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_field = box(left_x, top - h_field, left_w, h_field)
    ax_shape = box(left_x, bottom, left_w, h_shape)
    cp_bottom = bottom + h_shape + gap
    ax_cp = box(left_x, cp_bottom, left_w, top - h_field - gap - cp_bottom)
    cax = box(left_x + left_w + 0.15, top - h_field, 0.16, h_field)
    n_right = 3 if table is not None else 2
    right_x, right_w, right_gap = 9.35, 3.5, 0.2
    right_h = (top - bottom - (n_right - 1) * right_gap) / n_right
    right = [box(right_x, top - (r + 1) * right_h - r * right_gap, right_w, right_h) for r in range(n_right)]
    fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap=kwargs.get("cmap", "turbo")), cax=cax)
    cax.set_title(labels.get(field, field), fontsize=10)

    # the static curves, with a marker moved each frame
    ax_cd, ax_cl = right[0], right[1]
    ax_cd.plot(iteration, cd, "-", color="tab:blue", lw=1.3)
    ax_cd.set_ylabel("CD [counts]")
    ax_cl.plot(iteration, cl, "-", color="tab:green", lw=1.3)
    ax_cl.set_ylabel("CL")
    if cl_target is not None:
        ax_cl.axhline(cl_target, color="tab:red", ls="--", lw=0.8)
    markers = [ax_cd.plot([], [], "o", color="k", ms=6)[0], ax_cl.plot([], [], "o", color="k", ms=6)[0]]
    if table is not None:
        ax_conv = right[2]
        ax_conv.semilogy(iteration, np.maximum(table["opt"], 1e-12), "-", lw=1.3, label="optimality")
        ax_conv.semilogy(iteration, np.maximum(table["feas"], 1e-12), "-", lw=1.3, label="feasibility (lift constraint)")
        ax_conv.set_ylabel("convergence measures")
        ax_conv.legend(loc="upper right", fontsize=8)
        markers += [ax_conv.plot([], [], "o", color="k", ms=6)[0], ax_conv.plot([], [], "o", color="k", ms=6)[0]]
    for ax in right:
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(iteration.min() - 1, iteration.max() + 1)
    for ax in right[:-1]:
        ax.tick_params(labelbottom=False)
    right[-1].set_xlabel("optimizer iteration")

    first = surface_distribution(read_solution(case_dir, history[picks[0]]["time"]), **freestream)

    def draw(frame):
        g = picks[frame]
        solution = read_solution(case_dir, history[g]["time"])
        ax_field.clear()
        plot_field_2d(solution, field, ax=ax_field, vmin=vmin, vmax=vmax, colorbar=False, xlim=xlim, zlim=zlim, **kwargs)
        ax_field.set_xlabel("")
        ax_field.tick_params(labelbottom=False)
        ax_field.set_title(f"iteration {iteration[frame]}:  CD = {cd[frame]:.1f} counts,  CL = {cl[frame]:.3f}")
        ax_cp.clear()
        ax_cp.plot(first["x"], first["cp"], "-", color="0.6", lw=1.0, label="initial")
        dist = surface_distribution(solution, **freestream)
        ax_cp.plot(dist["x"], dist["cp"], "-", color="tab:red", lw=1.6, label="current")
        ax_cp.set_xlim(*xlim)
        ax_cp.set_ylim(1.2, -2.0)
        ax_cp.tick_params(labelbottom=False)
        ax_cp.set_ylabel("surface $C_p$")
        ax_cp.grid(alpha=0.3)
        ax_cp.legend(loc="center right", fontsize=9)
        ax_shape.clear()
        ax_shape.plot(first["x"], first["z"], "-", color="0.6", lw=1.0, label="initial")
        ax_shape.plot(dist["x"], dist["z"], "-", color="tab:red", lw=1.6, label="current")
        ax_shape.set_xlim(*xlim)
        ax_shape.set_ylim(*shape_zlim)
        ax_shape.set_aspect("equal")
        ax_shape.set_xlabel("x / c")
        ax_shape.set_ylabel("z / c")
        ax_shape.set_yticks([-0.1, 0.0, 0.1])
        ax_shape.grid(alpha=0.3)
        ax_shape.legend(loc="center right", fontsize=8)
        x = iteration[frame]
        markers[0].set_data([x], [cd[frame]])
        markers[1].set_data([x], [cl[frame]])
        if table is not None:
            markers[2].set_data([x], [max(table["opt"][frame], 1e-12)])
            markers[3].set_data([x], [max(table["feas"][frame], 1e-12)])

    written = []
    for path in outputs:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif suffix == ".mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("writing .mp4 needs ffmpeg on the PATH; write a .gif instead or install ffmpeg")
            writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        else:
            raise ValueError(f"output must end in .gif or .mp4; got {path}")
        with writer.saving(fig, str(path), dpi):
            for frame in range(len(picks)):
                draw(frame)
                writer.grab_frame()
            for _ in range(fps):  # hold the last frame for a second
                writer.grab_frame()
        written.append(path)
    plt.close(fig)
    return written


def plot_mesh_2d(case_or_solution, ax=None, plane_patch=None, wall_patch="wing", xlim=(-0.15, 1.15), zlim=(-0.3, 0.3), linewidth=0.4):
    """Draw the cells of a 2D case (the faces of its span-end plane) with their edges; works on a case directory or a :class:`Solution`."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if isinstance(case_or_solution, Solution):
        mesh, solution = case_or_solution.mesh, case_or_solution
    else:
        mesh = foam_io.read_mesh(case_or_solution)
        solution = Solution(case_dir=Path(case_or_solution), time="0", mesh=mesh)
    plane_patch = plane_patch or _midplane_patch(solution)
    faces = mesh["faces"][solution.patch_faces(plane_patch)]
    ax.add_collection(PolyCollection([mesh["points"][f][:, [0, 2]] for f in faces], facecolors="none", edgecolors="0.35", linewidths=linewidth))
    if wall_patch in mesh["patches"]:
        wall = mesh["faces"][solution.patch_faces(wall_patch)]
        ax.add_collection(LineCollection([mesh["points"][f][:, [0, 2]] for f in wall], colors="k", linewidths=1.0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("z / c")
    return ax


def read_optimization_history(path):
    """Parse the iteration table of a modOpt ``modopt_summary.out`` into ``{column name: array}`` (``major``, ``obj``, ``opt``, ``feas``, ...).

    ``path`` may be the file or the output directory of a run (``ASO_2DAF_rank0_outputs``; its newest run is used).
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("**/modopt_summary.out"))
        if not candidates:
            raise FileNotFoundError(f"no modopt_summary.out under {path}")
        path = candidates[-1]
    header, rows = None, []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "#" and len(words) > 2 and header is None:
            header = words[1:]
        elif header is not None and re.fullmatch(r"\d+", words[0]):
            try:
                rows.append([float(w) for w in words[: len(header) + 1]])
            except ValueError:
                continue
    if header is None or not rows:
        raise ValueError(f"no iteration table found in {path}")
    table = np.array([r for r in rows if len(r) == len(header) + 1])
    return {name: table[:, i + 1] for i, name in enumerate(header)}


def plot_optimization_history(history, objective_scale=1e4, objective_label="CD [counts]", target=None, axes=None, iterations=None, cl_target=None):
    """Convergence plot: drag, then lift (if ``iterations`` is given), then optimality and feasibility on a log scale.

    ``history`` is from :func:`read_optimization_history`. ``iterations`` is ``AirfoilModel.history`` (``CD`` and ``CL`` per
    iteration); with it the figure has three panels: CD, CL with its ``cl_target`` line, and the convergence measures.
    Without it, CD comes from modOpt's objective column and the figure has two panels.
    """
    import matplotlib.pyplot as plt

    n_panels = 3 if iterations else 2
    if axes is None:
        fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3.0 * n_panels), sharex=True, constrained_layout=True)
    if iterations:
        if "ngev" in history:  # one point per major iteration: the gradient evaluation of iterate k is number ngev[k]
            iterations = [iterations[min(max(int(g) - 1, 0), len(iterations) - 1)] for g in history["ngev"]]
        k = history["major"] if len(iterations) == len(history["major"]) else np.arange(len(iterations))
        cd = np.array([e["CD"] for e in iterations]) * objective_scale
        cl = np.array([e["CL"] for e in iterations])
        axes[0].plot(k, cd, "o-", ms=3.5, lw=1.2)
        axes[1].plot(k, cl, "o-", ms=3.5, lw=1.2, color="tab:green")
        axes[1].set_ylabel("CL")
        axes[1].grid(alpha=0.3)
        if cl_target is not None:
            axes[1].axhline(cl_target, color="tab:red", ls="--", lw=0.8, label=f"target {cl_target}")
            axes[1].legend(loc="lower right")
        measures, x = axes[2], history["major"]
    else:
        axes[0].plot(history["major"], history["obj"] * objective_scale, "o-", ms=3.5, lw=1.2)
        measures, x = axes[1], history["major"]
    axes[0].set_ylabel(objective_label)
    axes[0].grid(alpha=0.3)
    if target is not None:
        axes[0].axhline(target, color="tab:red", ls="--", lw=0.8)
    measures.semilogy(x, np.maximum(history["opt"], 1e-12), "o-", ms=3.5, lw=1.2, label="optimality")
    measures.semilogy(x, np.maximum(history["feas"], 1e-12), "s-", ms=3.5, lw=1.2, label="constraint violation (lift)")
    measures.set_xlabel("optimizer iteration")
    measures.set_ylabel("convergence measures")
    measures.grid(alpha=0.3, which="both")
    measures.legend(loc="upper right")
    return axes[0].figure


# --------------------------------------------------------------------------------------
# 3D (and general) surface export
# --------------------------------------------------------------------------------------
def write_patch_vtk(solution, patch, path, fields=("p", "Mach", "T")):
    """Write a boundary patch (for example the wing surface of a 3D case) with its cell values as legacy-VTK PolyData.

    Each polygon carries the value of the cell adjacent to it. Open the file in ParaView or PyVista. Fields may be any stored
    field name or ``"Mach"``. Works for any patch of any (2D or 3D) case.
    """
    mesh = solution.mesh
    sl = solution.patch_faces(patch)
    faces, owners = mesh["faces"][sl], mesh["owner"][sl]
    used = np.unique(np.concatenate(faces))
    local = {int(g): i for i, g in enumerate(used)}

    lines = ["# vtk DataFile Version 3.0", f"{patch} of {Path(solution.case_dir).name} at time {solution.time}", "ASCII", "DATASET POLYDATA"]
    lines.append(f"POINTS {len(used)} double")
    lines += [" ".join(f"{c:.12g}" for c in mesh["points"][g]) for g in used]
    total = sum(len(f) + 1 for f in faces)
    lines.append(f"POLYGONS {len(faces)} {total}")
    lines += [f"{len(f)} " + " ".join(str(local[int(i)]) for i in f) for f in faces]
    lines.append(f"CELL_DATA {len(faces)}")
    for name in fields:
        values = mach_number(solution) if name == "Mach" else solution.fields[name]
        values = np.asarray(values)[owners]
        if values.ndim == 1:
            lines.append(f"SCALARS {name} double 1")
            lines.append("LOOKUP_TABLE default")
            lines += [f"{v:.12g}" for v in values]
        else:
            lines.append(f"VECTORS {name} double")
            lines += [" ".join(f"{c:.12g}" for c in row) for row in values]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def plot_surface_3d(solution, patch="wing", field="Mach", path=None, **kwargs):
    """Render a boundary patch coloured by a cell field with PyVista (optional dependency). Returns the PyVista mesh.

    Writes a screenshot if ``path`` is given (off-screen). Meant for 3D cases; see the module docstring.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("plot_surface_3d needs pyvista (it is a dependency of lsdo_geo); or open write_patch_vtk's file in ParaView") from exc
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vtk = write_patch_vtk(solution, patch, Path(tmp) / "patch.vtk", fields=(field,) if field != "Mach" else ("Mach",))
        surface = pv.read(str(vtk))
    if path is not None:
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(surface, scalars=field, show_edges=False, **kwargs)
        plotter.screenshot(str(path))
        plotter.close()
    return surface


# --------------------------------------------------------------------------------------
# Command line:  python -m csdl_dafoam.viz CASE [CASE2 ...] -o figure.png
# --------------------------------------------------------------------------------------
def main(argv=None):
    """``python -m csdl_dafoam.viz CASE_DIR [CASE_DIR2] [--time T] [--field Mach|Cp|p|T|U] [-o out.png] [--vtk-patch wing]``.

    One case: a contour plot of ``--field`` (default Mach). Two cases: the comparison figure (Mach contours, surface Cp, shapes).
    ``--vtk-patch`` also writes that boundary patch with its fields as VTK for ParaView (for 3D cases, use this).
    """
    import argparse

    import matplotlib

    parser = argparse.ArgumentParser(description=(main.__doc__ or "").splitlines()[0])
    parser.add_argument("cases", nargs="+", help="one or two OpenFOAM case directories (serial or decomposed)")
    parser.add_argument("--time", default=None, help="solution time (default: latest flow solution)")
    parser.add_argument("--field", default="Mach")
    parser.add_argument("-o", "--output", default="flow.png")
    parser.add_argument("--vtk-patch", default=None, help="also write this boundary patch as VTK (next to the figure)")
    parser.add_argument("--labels", default=None, help="comma-separated labels for the comparison figure")
    args = parser.parse_args(argv)
    if len(args.cases) > 2:
        parser.error("give one or two cases")

    matplotlib.use("Agg")
    solutions = [read_solution(c, args.time) for c in args.cases]
    if len(solutions) == 1:
        ax = plot_field_2d(solutions[0], args.field)
        ax.figure.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        labels = args.labels.split(",") if args.labels else [Path(c).name for c in args.cases]
        plot_airfoil_comparison(solutions, labels=labels).savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    if args.vtk_patch:
        for case, solution in zip(args.cases, solutions):
            out = Path(args.output).with_name(f"{Path(case).name}_{args.vtk_patch}.vtk")
            write_patch_vtk(solution, args.vtk_patch, out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
