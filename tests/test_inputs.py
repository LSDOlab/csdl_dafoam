import csdl_alpha as csdl
import numpy as np
import pytest
from fakes import FakeLinearDAFoam

from csdl_dafoam.atmosphere import compute_ambient_conditions_group
from csdl_dafoam.inputs import compute_dafoam_input_variables


class Instance:
    def __init__(self, input_info):
        self.input_info = input_info

    def getOption(self, name):
        assert name == "inputInfo"
        return self.input_info


@pytest.fixture
def inline():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield
    rec.stop()


def groups(**flight):
    ambient = compute_ambient_conditions_group(csdl.Variable(value=0.0))
    group = csdl.VariableGroup()
    group.angle_of_attack = csdl.Variable(value=3.0)
    for key, value in flight.items():
        setattr(group, key, csdl.Variable(value=value))
    return ambient, group


def test_all_supported_input_types(inline):
    ambient, flight = groups(airspeed_m_s=238.0)
    x_vol = csdl.Variable(value=np.arange(6.0))
    out = compute_dafoam_input_variables(FakeLinearDAFoam(), ambient, flight, x_vol)

    assert out.aero_vol_coords is x_vol
    np.testing.assert_allclose(out.patch_velocity.value, [238.0, 3.0])
    assert float(np.ravel(out.pressure.value)[0]) == pytest.approx(101325.0)
    assert float(np.ravel(out.temperature.value)[0]) == pytest.approx(300.0)


def test_mach_number_is_converted_to_airspeed(inline):
    ambient, flight = groups(mach_number=0.7)
    out = compute_dafoam_input_variables(
        Instance({"v": {"type": "patchVelocity"}}), ambient, flight, csdl.Variable(value=np.zeros(3))
    )
    speed_of_sound = float(np.ravel(ambient.a_m_s.value)[0])
    np.testing.assert_allclose(out.v.value, [0.7 * speed_of_sound, 3.0])
    assert hasattr(flight, "airspeed_m_s")  # added to the group for reuse


@pytest.mark.parametrize(
    "info, message",
    [
        ({"u": {"type": "patchVar", "varName": "nuTilda"}}, "nuTilda"),
        ({"u": {"type": "designVar"}}, "designVar"),
    ],
)
def test_unsupported_inputs_raise(inline, info, message):
    ambient, flight = groups(airspeed_m_s=238.0)
    with pytest.raises(NotImplementedError, match=message):
        compute_dafoam_input_variables(Instance(info), ambient, flight, csdl.Variable(value=np.zeros(3)))
