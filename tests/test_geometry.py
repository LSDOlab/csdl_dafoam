"""Geometry pipeline on the real bundled STEP file and CFD wall mesh (real lsdo_geo)."""
import csdl_alpha as csdl
import numpy as np
import pytest

pytest.importorskip("lsdo_geo")
pytest.importorskip("lsdo_function_spaces")

from csdl_dafoam import cases, foam_io  # noqa: E402
from csdl_dafoam.geometry import (  # noqa: E402
    load_geometry,
    read_geometry_pickle,
    setup_airfoil_geometry,
    setup_geometry,
    write_geometry_pickle,
)


@pytest.fixture(scope="module")
def wall_points(tmp_path_factory):
    case = cases.copy_case("naca0012", tmp_path_factory.mktemp("g") / "c")
    return foam_io.read_patch_points(case, "wing")


@pytest.fixture
def recorder():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()


def project_and_parameterize(wall_points, thickness=None, camber=None, num_chordwise=5, cache_dir=None, **kw):
    """Returns (wall points re-evaluated on the deformed geometry as an (n, 3) array)."""
    geometry = load_geometry(cases.geometry_path(), cache_dir)
    projection = geometry.project(
        wall_points, grid_search_density_parameter=1, projection_tolerance=1e-3,
        grid_search_density_cutoff=50, plot=False, num_workers=1,
    )
    n = num_chordwise - 2
    th = csdl.Variable(value=np.zeros(n) if thickness is None else np.asarray(thickness, float))
    cm = csdl.Variable(value=np.zeros(n) if camber is None else np.asarray(camber, float))
    geometry = setup_airfoil_geometry(geometry, num_chordwise, th, cm, **kw)
    return np.asarray(geometry.evaluate(projection, plot=False).value)


def test_projection_of_the_cfd_wall_onto_the_step_geometry_is_tight(recorder, wall_points, tmp_path):
    undeformed = project_and_parameterize(wall_points, cache_dir=tmp_path)
    assert np.abs(undeformed - wall_points).max() < 1e-4  # the CFD mesh lies on the STEP surface


def test_positive_thickness_thickens_both_surfaces(recorder, wall_points, tmp_path):
    base = project_and_parameterize(wall_points, cache_dir=tmp_path)
    thick = project_and_parameterize(wall_points, thickness=[20, 20, 20], cache_dir=tmp_path)
    upper = base[:, 2] > 0.01
    lower = base[:, 2] < -0.01
    assert (thick[upper, 2] - base[upper, 2]).min() >= -1e-9
    assert (thick[lower, 2] - base[lower, 2]).max() <= 1e-9
    assert (thick[upper, 2] - base[upper, 2]).max() > 1e-3
    # the span (y) and chord extremes do not move
    np.testing.assert_allclose(thick[:, 1], base[:, 1], atol=1e-9)


def test_camber_shifts_the_whole_section_vertically(recorder, wall_points, tmp_path):
    base = project_and_parameterize(wall_points, cache_dir=tmp_path)
    cambered = project_and_parameterize(wall_points, camber=[5, 5, 5], cache_dir=tmp_path)
    assert (cambered[:, 2] - base[:, 2]).min() > -1e-9
    assert (cambered[:, 2] - base[:, 2]).max() > 1e-3


def test_camber_is_normalized_by_control_point_spacing_not_chord(recorder, wall_points, tmp_path):
    """Documents an inherited quirk (docs/how-it-works.md): a camber dof of 5 moves a control point by
    5% of chord/4, i.e. 0.0125, not 0.05."""
    base = project_and_parameterize(wall_points, cache_dir=tmp_path)
    cambered = project_and_parameterize(wall_points, camber=[5, 5, 5], cache_dir=tmp_path)
    assert (cambered[:, 2] - base[:, 2]).max() == pytest.approx(0.05 * 0.25, rel=0.1)


def test_dict_wrapper_matches_the_function(recorder, wall_points, tmp_path):
    geometry = load_geometry(cases.geometry_path(), tmp_path)
    th = csdl.Variable(value=np.zeros(3))
    cm = csdl.Variable(value=np.zeros(3))
    out = setup_geometry(geometry, {
        "num_ffd_coefficients_chordwise": 5,
        "percent_change_in_thickness_dof": th,
        "normalized_percent_camber_change_dof": cm,
    })
    assert out is geometry


def test_import_cache_is_reused_and_keyed_by_the_step_file(recorder, tmp_path):
    import shutil

    first = load_geometry(cases.geometry_path(), tmp_path)
    caches = list(tmp_path.glob("*_import.pickle"))
    assert len(caches) == 1
    stamp = caches[0].stat().st_mtime_ns
    load_geometry(cases.geometry_path(), tmp_path)
    assert caches[0].stat().st_mtime_ns == stamp, "second load must read the cache, not rewrite it"

    # a different STEP file (different content) gets its own cache file
    edited = tmp_path / "edited.stp"
    shutil.copy(cases.geometry_path(), edited)
    edited.write_text(edited.read_text() + "\n/* edited */\n")
    load_geometry(edited, tmp_path)
    assert len(list(tmp_path.glob("*_import.pickle"))) == 2
    assert first is not None


def test_corrupt_cache_is_regenerated(recorder, tmp_path):
    load_geometry(cases.geometry_path(), tmp_path)
    (cache,) = tmp_path.glob("*_import.pickle")
    cache.write_bytes(b"not a pickle")
    geometry = load_geometry(cases.geometry_path(), tmp_path)  # must not raise
    assert geometry is not None
    assert cache.stat().st_size > 100


def test_geometry_pickle_roundtrip_preserves_coefficients(recorder, tmp_path):
    geometry = load_geometry(cases.geometry_path(), tmp_path)
    path = tmp_path / "g.pickle"
    write_geometry_pickle(geometry, path)
    again = read_geometry_pickle(path)
    for key, function in geometry.functions.items():
        np.testing.assert_array_equal(again.functions[key].coefficients.value, function.coefficients.value)
