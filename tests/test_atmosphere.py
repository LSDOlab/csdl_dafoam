import csdl_alpha as csdl
import numpy as np
import pytest

from csdl_dafoam.atmosphere import compute_ambient_conditions_group, if_below_then_else


@pytest.fixture
def inline():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield
    rec.stop()


def ambient(h_m):
    group = compute_ambient_conditions_group(csdl.Variable(value=h_m))
    return {name: float(np.ravel(getattr(group, name).value)[0]) for name in ("T_K", "P_Pa", "rho_kg_m3", "a_m_s", "mu_kg_m_s", "nu_m2_s")}


def test_sea_level(inline):
    a = ambient(0.0)
    assert a["T_K"] == pytest.approx(300.0)  # the package's reference case uses 300 K, not 288.15 K
    assert a["P_Pa"] == pytest.approx(101325.0)
    assert a["rho_kg_m3"] == pytest.approx(101325.0 / (287.05 * 300.0))
    assert a["a_m_s"] == pytest.approx(np.sqrt(1.4 * 287.05 * 300.0))
    # the reference case's U0 = 238 m/s is the documented ~Mach 0.69
    assert 238.0 / a["a_m_s"] == pytest.approx(0.686, abs=2e-3)


def test_troposphere_lapse_rate(inline):
    a = ambient(5000.0)
    assert a["T_K"] == pytest.approx(300.0 - 0.0065 * 5000.0)
    assert a["P_Pa"] == pytest.approx(101325.0 * (a["T_K"] / 300.0) ** (9.80665 / (0.0065 * 287.05)))


def test_stratosphere_is_isothermal(inline):
    low, high = ambient(13000.0), ambient(18000.0)
    assert low["T_K"] == pytest.approx(216.65)
    assert high["T_K"] == pytest.approx(216.65)
    assert high["P_Pa"] < low["P_Pa"] < 22632.06
    assert high["P_Pa"] == pytest.approx(22632.06 * np.exp(-9.80665 * (18000.0 - 11000.0) / (287.05 * 216.65)))


def test_density_and_pressure_decrease_with_altitude(inline):
    rho = [ambient(h)["rho_kg_m3"] for h in (0.0, 3000.0, 6000.0, 9000.0)]
    assert all(a > b for a, b in zip(rho, rho[1:]))


def test_viscosity_follows_sutherland(inline):
    a = ambient(0.0)
    expected = 1.716e-5 * (300.0 / 273.15) ** 1.5 * (273.15 + 110.4) / (300.0 + 110.4)
    assert a["mu_kg_m_s"] == pytest.approx(expected)
    assert a["nu_m2_s"] == pytest.approx(a["mu_kg_m_s"] / a["rho_kg_m3"])


def test_if_below_then_else_switches(inline):
    below = if_below_then_else(csdl.Variable(value=1.0), 5.0, 10.0, 20.0, rho=50.0)
    above = if_below_then_else(csdl.Variable(value=9.0), 5.0, 10.0, 20.0, rho=50.0)
    assert float(np.ravel(below.value)[0]) == pytest.approx(10.0)
    assert float(np.ravel(above.value)[0]) == pytest.approx(20.0)
