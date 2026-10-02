"""The bundled 336-cell test mesh (``naca0012_coarse``) and the script that generates it."""
import gzip
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from csdl_dafoam import cases, foam_check, foam_io

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_coarse_airfoil_mesh.py"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("make_coarse_airfoil_mesh", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    return cases.copy_case("naca0012_coarse", tmp_path_factory.mktemp("coarse") / "c", nprocs=1)


def read_text(path):
    path = Path(path)
    if path.is_file():
        return path.read_text()
    with gzip.open(str(path) + ".gz", "rt") as handle:
        return handle.read()


def test_bundled_mesh_is_what_the_script_generates(generator, tmp_path):
    """Guards against the two drifting apart: edit the script or the mesh, regenerate the other."""
    with tempfile.TemporaryDirectory() as tmp:
        fine = cases.copy_case("naca0012", Path(tmp) / "c", nprocs=1)
        ids, points = generator.structured_rings(fine)
    i_keep, j_keep = generator.choose_indices(ids.shape[0], ids.shape[1], 3, 4)
    grid = points[ids[np.ix_(i_keep, j_keep)]][:, :, [0, 2]]
    if generator.cell_areas(generator.periodic(grid)).mean() < 0:
        grid = np.concatenate([grid[:1], grid[:0:-1]])
    generator.write_polymesh(grid, tmp_path)

    bundled = cases._data_dir() / "cases" / "naca0012_coarse_overlay" / "constant" / "polyMesh"
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        assert read_text(tmp_path / name) == read_text(bundled / name), name


def test_topology(coarse):
    mesh = foam_io.read_mesh(coarse)
    n_cells = int(mesh["owner"].max()) + 1
    assert n_cells == 336
    patches = {k: v["nFaces"] for k, v in mesh["patches"].items()}
    assert patches == {"symmetry1": 336, "symmetry2": 336, "wing": 42, "inout": 42}
    # every cell is a hexahedron: 6 faces counting both sides of internal faces and all boundary faces
    counts = np.bincount(np.concatenate([mesh["owner"], mesh["neighbour"]]), minlength=n_cells)
    assert np.all(counts == 6)
    assert np.all(mesh["owner"][: len(mesh["neighbour"])] < mesh["neighbour"]), "OpenFOAM needs owner < neighbour"
    assert np.all(np.diff(mesh["owner"][: len(mesh["neighbour"])]) >= 0), "internal faces sorted by owner"


def test_geometry_is_valid_and_close_to_the_bundled_domain(coarse, tmp_path):
    mesh = foam_io.read_mesh(coarse)
    volumes = foam_io.cell_volumes(mesh)
    assert volumes.min() > 0
    fine = foam_io.read_mesh(cases.copy_case("naca0012", tmp_path / "fine", nprocs=1))
    assert volumes.sum() == pytest.approx(foam_io.cell_volumes(fine).sum(), rel=0.02)
    wall = foam_io.read_patch_points(coarse, "wing")
    assert wall.shape == (84, 3)  # 42 points at each of the two span planes
    assert wall[:, 0].min() == pytest.approx(0.0, abs=1e-12) and wall[:, 0].max() == pytest.approx(0.99882687, abs=1e-6)


def test_wall_points_lie_on_the_step_geometry(coarse, tmp_path):
    pytest.importorskip("lsdo_geo")
    import csdl_alpha as csdl

    from csdl_dafoam.geometry import load_geometry

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        geometry = load_geometry(cases.geometry_path(), tmp_path)
        wall = foam_io.read_patch_points(coarse, "wing")
        projection = geometry.project(wall, grid_search_density_parameter=1, projection_tolerance=1e-3, grid_search_density_cutoff=50, plot=False, num_workers=1)
        error = np.abs(np.asarray(geometry.evaluate(projection, plot=False).value) - wall).max()
    finally:
        recorder.stop()
    assert error < 1e-4


@pytest.mark.openfoam
@pytest.mark.skipif(foam_check.find_openfoam() is None, reason="no OpenFOAM available")
def test_openfoam_accepts_the_mesh(coarse):
    result = foam_check.check_mesh(coarse)
    assert result.dafoam_ok, result.failures
    assert result.max_non_orthogonality < 70 and result.max_skewness < 4 and result.max_aspect_ratio < 1000
