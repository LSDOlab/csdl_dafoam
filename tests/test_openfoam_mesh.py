"""Warped meshes against OpenFOAM's own checkMesh (needs an OpenFOAM: PATH or OpenFOAM.app).

    pytest -m openfoam

Runs on macOS with ``brew install gerlero/openfoam/openfoam@2506``, or inside the conda/DAFoam
environment on Linux.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("idwarp_jax")

from csdl_dafoam import cases, foam_check, foam_io  # noqa: E402
from csdl_dafoam.jax_warp import JaxIDWarp  # noqa: E402

pytestmark = [
    pytest.mark.openfoam,
    pytest.mark.skipif(foam_check.find_openfoam() is None, reason="no OpenFOAM available"),
]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return cases.copy_case("naca0012", tmp_path_factory.mktemp("om") / "c", nprocs=1)


@pytest.fixture(scope="module")
def warper(case):
    return JaxIDWarp(case, ["wing"])


def bump(points, thickness=0.0, camber=0.0):
    target = points.copy()
    s = np.sin(np.pi * points[:, 0])
    target[:, 2] += thickness * 0.06 * s * np.sign(points[:, 2]) + camber * s
    return target


def test_bundled_mesh_passes_dafoams_checks(case):
    result = foam_check.check_mesh(case)
    assert result.dafoam_ok, result.failures
    # the stock utility's determinant test fails on every cell of a one-cell-thick symmetry-closed mesh
    assert any("determinant" in m.lower() for m in result.ignored)
    assert result.max_aspect_ratio < 1000 and result.max_non_orthogonality < 70 and result.max_skewness < 4


@pytest.mark.parametrize("thickness, camber", [(0.25, 0.0), (0.0, 0.0125), (0.5, 0.05)])
def test_idwarp_jax_meshes_pass_dafoams_checks(case, warper, thickness, camber):
    """The warped meshes are checked by OpenFOAM itself, including a perturbation ~4x larger than the
    optimizer's first steps."""
    warped = warper.warp(bump(warper.wall_points, thickness, camber))
    result = foam_check.check_mesh(case, points=warped)
    assert result.dafoam_ok, result.failures
    assert result.min_volume > 0
    # OpenFOAM's volumes agree with ours (cross-check of foam_io.cell_volumes)
    ours = foam_io.cell_volumes(warper.mesh, warped)
    assert result.min_volume == pytest.approx(ours.min(), rel=1e-6)


def test_a_broken_mesh_is_reported(case, warper):
    rng = np.random.default_rng(0)
    wrecked = warper.points + rng.standard_normal(warper.points.shape) * 0.05  # tangled
    result = foam_check.check_mesh(case, points=wrecked)
    assert not result.dafoam_ok
