import csdl_alpha as csdl
import numpy as np
from fakes import FakeWarpingDAFoam

from csdl_dafoam.mesh_warp import DAFoamMeshWarper


def test_warp_and_its_adjoint(recorder, tmp_path):
    fake = FakeWarpingDAFoam(run_directory=str(tmp_path))
    x_surf = csdl.Variable(value=np.random.default_rng(3).standard_normal(12), name="x_surf")
    x_vol = DAFoamMeshWarper(fake).evaluate(x_surf)
    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)
    sim.run()
    np.testing.assert_allclose(sim[x_vol], fake.W @ x_surf.value + fake.mesh.x_vol0, rtol=1e-12)

    totals = sim.compute_totals([x_vol], [x_surf])
    np.testing.assert_allclose(totals[x_vol, x_surf], fake.W, rtol=1e-10, atol=1e-12)
