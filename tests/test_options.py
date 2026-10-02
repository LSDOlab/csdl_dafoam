import pytest

from csdl_dafoam.options import idwarp_options, naca0012_options


def test_force_scaling_normalizes_to_coefficients():
    opts = naca0012_options(U0=238.0, p0=101325.0, T0=300.0, A0=0.1)
    rho0 = 101325.0 / 300.0 / 287.0
    expected = 1.0 / (0.5 * 238.0**2 * 0.1 * rho0)
    assert opts["function"]["CD"]["scale"] == pytest.approx(expected)
    assert opts["function"]["CL"]["scale"] == pytest.approx(expected)


def test_freestream_values_propagate():
    opts = naca0012_options(U0=250.0, p0=90000.0, T0=280.0, nuTilda0=4.5e-9)
    bc = opts["primalBC"]
    assert bc["U0"]["value"] == [250.0, 0.0, 0.0]
    assert bc["p0"]["value"] == [90000.0]
    assert bc["T0"]["value"] == [280.0]
    assert bc["nuTilda0"]["value"] == [4.5e-9]
    assert opts["normalizeStates"]["nuTilda"] == pytest.approx(4.5e-8)


def test_inputs_cover_everything_the_interface_can_build():
    opts = naca0012_options()
    types = {k: v["type"] for k, v in opts["inputInfo"].items()}
    assert types == {
        "aero_vol_coords": "volCoord",
        "patch_velocity": "patchVelocity",
        "pressure": "patchVar",
        "temperature": "patchVar",
    }
    # the force functions must refer to an input that exists
    for fn in opts["function"].values():
        assert fn["patchVelocityInputName"] in opts["inputInfo"]


def test_idwarp_options_use_absolute_case_path(tmp_path):
    opts = idwarp_options(tmp_path)
    assert opts["gridFile"] == str(tmp_path.resolve())
    assert opts["fileType"] == "OpenFOAM"


def test_options_for_case_picks_matching_turbulence_seed():
    from csdl_dafoam.options import options_for_case

    assert options_for_case("naca0012")["primalBC"]["nuTilda0"]["value"] == [4.5e-5]
    assert options_for_case("naca0012_euler")["primalBC"]["nuTilda0"]["value"] == [4.5e-9]
    assert options_for_case("naca0012_coarse")["primalBC"]["nuTilda0"]["value"] == [4.5e-9]
    assert options_for_case("naca0012_euler", U0=250.0)["primalBC"]["U0"]["value"][0] == 250.0
    with pytest.raises(KeyError):
        options_for_case("nope")


def test_adjoint_tolerance_is_configurable():
    from csdl_dafoam.options import options_for_case

    assert naca0012_options()["adjEqnOption"]["gmresRelTol"] == 1e-6
    assert options_for_case("naca0012_coarse", adjoint_rel_tol=1e-3)["adjEqnOption"]["gmresRelTol"] == 1e-3
