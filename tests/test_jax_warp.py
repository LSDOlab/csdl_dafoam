"""IDWarp-JAX integration on the real bundled mesh."""
import csdl_alpha as csdl
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("idwarp_jax")

from csdl_dafoam import cases, foam_io  # noqa: E402
from csdl_dafoam.jax_warp import JaxIDWarp, JaxMeshWarper, match_local_points  # noqa: E402


@pytest.fixture(scope="module")
def warper(tmp_path_factory):
    case = cases.copy_case("naca0012", tmp_path_factory.mktemp("jw") / "c")
    return JaxIDWarp(case, ["wing"])


def bump(points, thickness=0.0, camber=0.0):
    """A smooth, FFD-like displacement of the airfoil wall."""
    target = points.copy()
    s = np.sin(np.pi * points[:, 0])
    target[:, 2] += thickness * 0.06 * s * np.sign(points[:, 2]) + camber * s
    return target


def test_moving_surface_is_the_wing_patch_in_openfoam_point_order(warper):
    assert warper.wall_points.shape == (252, 3)
    np.testing.assert_allclose(warper.wall_points, warper.points[warper.wall_ids])
    assert np.all(np.diff(warper.wall_ids) > 0), "IDWarp-JAX orders the wall by ascending point index"


def test_undeformed_surface_leaves_the_mesh_unchanged(warper):
    np.testing.assert_allclose(warper.warp(warper.wall_points), warper.points, atol=1e-9)


def test_wall_follows_the_prescribed_surface_and_span_is_preserved(warper):
    target = bump(warper.wall_points, thickness=0.25, camber=0.01)
    volume = warper.warp(target)
    np.testing.assert_allclose(volume[warper.wall_ids], target, atol=1e-4)
    np.testing.assert_allclose(volume[:, 1], warper.points[:, 1], atol=1e-12)  # 2D: no span motion


@pytest.mark.parametrize("thickness, camber", [(0.25, 0.0), (0.0, 0.0125), (0.5, 0.05)])
def test_warped_mesh_has_no_inverted_cells(warper, thickness, camber):
    """Includes a perturbation about 4x larger than the optimizer's first steps."""
    original = foam_io.cell_volumes(warper.mesh)
    assert original.min() > 0
    volume = warper.warp(bump(warper.wall_points, thickness, camber))
    warped = foam_io.cell_volumes(warper.mesh, volume)
    assert warped.min() > 0, "a cell inverted"
    assert (warped / original).min() > 0.4, "a cell lost more than 60% of its volume"


@pytest.mark.slow
def test_vjp_matches_finite_differences(warper):
    rng = np.random.default_rng(0)
    target = bump(warper.wall_points, thickness=0.2, camber=0.01)
    seed = rng.standard_normal(warper.points.shape)
    direction = rng.standard_normal(target.shape) * 1e-3
    h = 1e-6
    fd = (np.sum(seed * warper.warp(target + h * direction)) - np.sum(seed * warper.warp(target - h * direction))) / (2 * h)
    assert np.sum(warper.vjp(target, seed) * direction) == pytest.approx(fd, rel=1e-5)


def test_match_local_points_handles_shuffles_and_duplicates(warper):
    rng = np.random.default_rng(2)
    index = np.concatenate([rng.permutation(len(warper.points))[:500], np.arange(10)])
    np.testing.assert_array_equal(match_local_points(warper.points, warper.points[index]), index)


def test_match_local_points_rejects_a_foreign_mesh(warper):
    with pytest.raises(ValueError, match="no match"):
        match_local_points(warper.points, warper.points[:50] + 1e-3)


def test_csdl_operation_scatters_seeds_of_shared_points_and_matches_the_global_vjp(warper, recorder):
    """Two 'ranks' hold overlapping point sets; summing their reverse results equals the global VJP of
    the summed seeds (which is what the MPI region's sum over ranks computes)."""
    rng = np.random.default_rng(3)
    n = len(warper.points)
    ranks = [np.concatenate([rng.permutation(n)[:300], [5, 6]]), np.concatenate([rng.permutation(n)[:300], [5, 6]])]
    seeds = [rng.standard_normal((len(r), 3)) for r in ranks]

    target = bump(warper.wall_points, thickness=0.2)
    x_surf = csdl.Variable(value=target.flatten(), name="x_surf")
    cotangents = []
    for r, seed in zip(ranks, seeds):
        op = JaxMeshWarper(warper, r)
        x_vol = op.evaluate(x_surf)
        # CSDL evaluates reverse mode of sum(seed * x_vol)
        cotangents.append(csdl.sum(x_vol * seed.flatten()))
    total = cotangents[0] + cotangents[1]
    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)
    sim.run()
    got = np.asarray(sim.compute_totals([total], [x_surf])[total, x_surf]).flatten()

    global_seed = np.zeros_like(warper.points)
    for r, seed in zip(ranks, seeds):
        np.add.at(global_seed, r, seed)
    np.testing.assert_allclose(got, warper.vjp(target, global_seed).flatten(), rtol=1e-8, atol=1e-12)
