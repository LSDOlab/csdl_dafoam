"""Build the CSDL variables DAFoam declares as inputs (``inputInfo``)."""
import csdl_alpha as csdl

__all__ = ["compute_dafoam_input_variables"]


def compute_dafoam_input_variables(
    dafoam_instance,
    ambient_conditions_group: csdl.VariableGroup,
    flight_conditions_group: csdl.VariableGroup,
    aerodynamic_volume_coordinates: csdl.Variable,
):
    """Create one CSDL variable per entry of the DAFoam ``inputInfo`` option.

    ``ambient_conditions_group`` must provide ``T_K``, ``P_Pa`` and ``a_m_s`` (see
    :func:`csdl_dafoam.atmosphere.compute_ambient_conditions_group`).
    ``flight_conditions_group`` must provide ``angle_of_attack`` [deg] and either
    ``airspeed_m_s`` or ``mach_number``; if only the Mach number is given,
    ``airspeed_m_s`` is computed from it and added to the group.

    Supported ``inputInfo`` types are ``volCoord``, ``patchVelocity`` and ``patchVar``
    (for ``p`` and ``T``). Anything else raises ``NotImplementedError``.
    """
    dafoam_input_variables_group = csdl.VariableGroup()

    # Read inputInfo listed in daOptions (these should be set by user)
    dafoam_input_dict = dafoam_instance.getOption("inputInfo")

    for input_name in dafoam_input_dict.keys():
        input_type = dafoam_input_dict[input_name]["type"]

        if input_type == "volCoord":
            setattr(dafoam_input_variables_group, input_name, aerodynamic_volume_coordinates)

        # patchVelocity: [airspeed, angle of attack in degrees]
        elif input_type == "patchVelocity":
            if not hasattr(flight_conditions_group, "airspeed_m_s"):
                flight_conditions_group.airspeed_m_s = (
                    flight_conditions_group.mach_number * ambient_conditions_group.a_m_s
                )

            patch_velocity = csdl.concatenate(
                (flight_conditions_group.airspeed_m_s, flight_conditions_group.angle_of_attack)
            )
            setattr(dafoam_input_variables_group, input_name, patch_velocity)

        elif input_type == "patchVar":
            input_variable_name = dafoam_input_dict[input_name]["varName"]

            if input_variable_name == "p":
                setattr(dafoam_input_variables_group, input_name, ambient_conditions_group.P_Pa)
            elif input_variable_name == "T":
                setattr(dafoam_input_variables_group, input_name, ambient_conditions_group.T_K)
            else:
                raise NotImplementedError(
                    f'unable to create csdl variable for "{input_name}" '
                    f'(patchVar for "{input_variable_name}" interpretation not implemented)'
                )

        else:
            raise NotImplementedError(
                f'unable to create csdl variable for "{input_name}" (type "{input_type}" interpretation not implemented)'
            )

    return dafoam_input_variables_group
