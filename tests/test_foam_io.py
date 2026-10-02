import numpy as np
import pytest

from csdl_dafoam import cases, foam_io


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return cases.copy_case("naca0012", tmp_path_factory.mktemp("case") / "c")


def test_list_patches(case):
    patches = foam_io.list_patches(case)
    assert set(patches) == {"symmetry1", "symmetry2", "wing", "inout"}
    assert patches["wing"]["type"] == "wall"
    assert patches["wing"]["nFaces"] == 126
    assert patches["symmetry1"]["nFaces"] == 4032  # one cell thick: one face per cell


def test_wing_points_describe_a_unit_chord_naca0012(case):
    pts = foam_io.read_patch_points(case, "wing")
    assert pts.shape == (252, 3)  # 126 faces, two spanwise points each
    np.testing.assert_allclose(pts.min(axis=0), [0.0, 0.0, -0.0600], atol=1e-3)
    np.testing.assert_allclose(pts.max(axis=0), [1.0, 0.1, 0.0600], atol=2e-3)
    # NACA 0012: half-thickness 0.06 at the maximum-thickness station near x = 0.3
    assert abs(pts[:, 2]).max() == pytest.approx(0.06, abs=1e-3)
    assert pts[np.argmax(np.abs(pts[:, 2])), 0] == pytest.approx(0.3, abs=0.05)


def test_unknown_patch_raises(case):
    with pytest.raises(KeyError):
        foam_io.read_patch_points(case, "nope")
