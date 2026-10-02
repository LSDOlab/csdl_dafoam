"""Flow-solution reading and plotting. The decomposed fixture is real DAFoam output (4 MPI ranks, the bundled coarse NACA 0012
case, 336 cells, from a 3-iteration optimization with ``examples/airfoil_optimization.py --case naca0012_coarse``): the flow
solution at the first design evaluated (time 0.0001) and at the final design (time 2)."""
import tarfile
from pathlib import Path

import numpy as np
import pytest

from csdl_dafoam import cases, viz

DATA = Path(__file__).parent / "data" / "snapshot_4proc_naca0012_coarse.tgz"


@pytest.fixture(scope="module")
def decomposed_case(tmp_path_factory):
    case = cases.copy_case("naca0012_coarse", tmp_path_factory.mktemp("viz") / "case", nprocs=4)
    with tarfile.open(DATA) as archive:
        archive.extractall(case)
    return case


@pytest.fixture(scope="module")
def final(decomposed_case):
    return viz.read_solution(decomposed_case)


@pytest.fixture(scope="module")
def initial(decomposed_case):
    return viz.read_solution(decomposed_case, time="0.0001")


def test_latest_solution_time_is_the_integer_flow_directory(decomposed_case):
    # processor0 holds "2" (latest primal solve) and "0.0001" (a renamed earlier solve): only the former counts
    assert viz.latest_solution_time(decomposed_case) == "2"


def test_decomposed_fields_are_reassembled_on_the_global_mesh(final):
    assert final.fields["p"].shape == (336,)
    assert final.fields["U"].shape == (336, 3)
    assert final.fields["T"].shape == (336,)
    assert not np.isnan(final.fields["p"]).any()
    # a freestream of 238 m/s at 3 deg: the far-field cells (the outermost layer) carry it
    far = final.mesh["owner"][final.patch_faces("inout")]
    speed = np.linalg.norm(final.fields["U"][far], axis=1)
    assert 200 < np.median(speed) < 280
    # suction peak (a supersonic pocket, p below 1 atm) and stagnation (above 1 atm)
    assert 0.4e5 < final.fields["p"].min() < 1.0e5 and 1.0e5 < final.fields["p"].max() < 1.4e5
    assert 0.0 < viz.mach_number(final).max() < 2.0


def test_deformed_points_replace_the_undeformed_ones(initial, final):
    wall = final.mesh["faces"][final.patch_faces("wing")]
    ids = np.unique(np.concatenate(wall))
    moved = np.abs(final.mesh["points"][ids, 2] - initial.mesh["points"][ids, 2]).max()
    assert moved > 1e-3, "initial and final designs must have different wall coordinates"
    # the span coordinate never changes in a 2D design
    np.testing.assert_allclose(final.mesh["points"][:, 1], initial.mesh["points"][:, 1], atol=1e-12)


def test_surface_distribution_is_ordered_around_the_airfoil(final):
    dist = viz.surface_distribution(final)
    n = len(dist["x"])
    assert n == 42
    assert dist["x"][0] < 0.02, "starts at the leading edge"
    up = dist["upper"]
    assert up[: up.sum()].all() and not up[up.sum() :].any(), "upper surface first, then lower"
    assert np.all(np.diff(dist["x"][up]) > 0) and np.all(np.diff(dist["x"][~up]) < 0)
    assert dist["cp"].max() > 0.3 and dist["cp"].min() < -0.3  # stagnation and suction


def test_two_designs_have_different_surface_pressure(initial, final):
    a, b = viz.surface_distribution(initial), viz.surface_distribution(final)
    assert np.abs(a["cp"] - b["cp"]).max() > 0.05


def test_plots_are_written(initial, final, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = viz.plot_field_2d(final, "Mach")
    assert len(ax.collections) >= 2  # cell polygons + outline
    ax.figure.savefig(tmp_path / "mach.png", dpi=60)
    plt.close("all")
    fig = viz.plot_airfoil_comparison([initial, final], title="test")
    fig.savefig(tmp_path / "comparison.png", dpi=60)
    plt.close("all")
    assert (tmp_path / "mach.png").stat().st_size > 5_000
    assert (tmp_path / "comparison.png").stat().st_size > 10_000


def test_single_design_figure_has_no_shape_panel(final, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    one = viz.plot_airfoil_comparison([final], labels=["x"])
    two = viz.plot_airfoil_comparison([final, final], labels=["a", "b"], field="Mach")
    assert len(one.axes) < len(two.axes)
    plt.close("all")


def test_solution_times_lists_every_kept_solution_in_order(decomposed_case):
    assert viz.solution_times(decomposed_case) == ["0.0001", "2"]


def test_movie_dashboard_has_a_frame_per_iteration_plus_a_hold(decomposed_case, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    PIL = pytest.importorskip("PIL.Image")
    history = [{"CD": 0.0117, "CL": 0.47, "time": "0.0001"}, {"CD": 0.0090, "CL": 0.49, "time": "0.0001"}, {"CD": 0.0077, "CL": 0.499, "time": "2"}]
    table = {"major": np.array([0.0, 1.0]), "ngev": np.array([1.0, 3.0]), "opt": np.array([0.06, 1e-3]), "feas": np.array([7e-3, 1e-6])}
    (path,) = viz.make_movie(decomposed_case, tmp_path / "m.gif", history, table=table, cl_target=0.5, fps=2, dpi=40)
    with PIL.open(path) as image:
        assert image.n_frames >= 2
    assert path.stat().st_size > 2_000
    # without the table: one frame per gradient evaluation, two panels of history
    (plain,) = viz.make_movie(decomposed_case, tmp_path / "p.gif", history, fps=2, dpi=40)
    assert plain.exists()
    with pytest.raises(ValueError, match="gif or .mp4"):
        viz.make_movie(decomposed_case, tmp_path / "m.avi", history)
    with pytest.raises(ValueError, match="time"):
        viz.make_movie(decomposed_case, tmp_path / "x.gif", [{"CD": 1.0, "CL": 1.0, "time": None}])


def test_history_with_lift_has_three_panels(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = {"major": np.arange(4.0), "obj": np.linspace(0.012, 0.008, 4), "opt": np.logspace(-1, -3, 4), "feas": np.logspace(-2, -5, 4)}
    iterations = [{"CD": 0.012 - 0.001 * i, "CL": 0.45 + 0.015 * i} for i in range(4)]
    fig = viz.plot_optimization_history(table, iterations=iterations, cl_target=0.5)
    assert len(fig.axes) == 3
    assert len(viz.plot_optimization_history(table).axes) == 2
    plt.close("all")


def test_patch_vtk_export(final, tmp_path):
    path = viz.write_patch_vtk(final, "wing", tmp_path / "wing.vtk")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# vtk DataFile") and "DATASET POLYDATA" in lines
    assert "POINTS 84 double" in lines
    assert any(line.startswith("POLYGONS 42 ") for line in lines)
    assert "SCALARS p double 1" in lines and "SCALARS Mach double 1" in lines and "SCALARS T double 1" in lines
    # point indices stay within the written points
    start = lines.index(next(l for l in lines if l.startswith("POLYGONS")))
    indices = [int(i) for l in lines[start + 1 : start + 43] for i in l.split()[1:]]
    assert max(indices) < 84


# ---- serial cases, synthetic fields --------------------------------------------------
def write_field(path, kind, internal, extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class vol%sField;\n    object x;\n}\n\n"
        "dimensions [0 0 0 0 0 0 0];\n\ninternalField %s\n\nboundaryField\n{\n}\n" % (kind, internal)
    )


@pytest.fixture
def serial_case(tmp_path):
    case = cases.copy_case("naca0012_coarse", tmp_path / "c", nprocs=1)
    n = 336
    x = np.linspace(0, 1, n)
    write_field(case / "100" / "p", "Scalar", "nonuniform List<scalar> \n%d\n(\n%s\n)\n;" % (n, "\n".join(f"{101325 + 1000 * v:.10g}" for v in x)))
    write_field(case / "100" / "U", "Vector", "uniform (238 0 0);")
    write_field(case / "100" / "T", "Scalar", "uniform 300;")
    return case


def test_serial_case_with_uniform_and_nonuniform_fields(serial_case):
    solution = viz.read_solution(serial_case)
    assert solution.time == "100"
    assert solution.fields["p"].shape == (336,) and solution.fields["p"][0] == pytest.approx(101325)
    assert solution.fields["U"].shape == (336, 3) and np.all(solution.fields["U"][:, 0] == 238)
    assert viz.mach_number(solution) == pytest.approx(238 / np.sqrt(1.4 * 287 * 300), rel=1e-12)


def test_cp_is_zero_in_the_freestream(serial_case, tmp_path):
    write_field(serial_case / "100" / "p", "Scalar", "uniform 101325;")
    solution = viz.read_solution(serial_case)
    dist = viz.surface_distribution(solution)
    np.testing.assert_allclose(dist["cp"], 0.0, atol=1e-12)


def test_missing_fields_and_times_raise(serial_case):
    with pytest.raises(FileNotFoundError):
        viz.read_solution(serial_case, time="7")
    with pytest.raises(FileNotFoundError):
        viz.latest_solution_time(serial_case.parent)  # not a case
