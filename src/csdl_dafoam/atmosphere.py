"""Differentiable standard-atmosphere model (CSDL) used to set freestream conditions."""
import csdl_alpha as csdl

# Constants
T0_K = 300.0  # sea-level temperature [K]. NOTE: 300 K rather than the ISA 288.15 K, to
#               match the reference NACA0012 case (U0 = 238 m/s is Mach ~0.69 at 300 K).
T1_K = 216.65  # temperature at the tropopause [K]
L_K_M = 0.0065  # temperature lapse rate [K/m]
H_TROPOPAUSE_M = 11000.0  # tropopause altitude [m]
P0_PA = 101325.0  # sea-level pressure [Pa]
P1_PA = 22632.06  # pressure at the tropopause [Pa]
G_M_S2 = 9.80665  # gravitational acceleration [m/s^2]
R_M2_S2_K = 287.05  # specific gas constant of dry air [J/(kg K)]
GAMMA = 1.4  # ratio of specific heats
MU_REF_KG_M_S = 1.716e-5  # Sutherland reference viscosity [kg/(m s)]
T_REF_K = 273.15  # Sutherland reference temperature [K]
S_K = 110.4  # Sutherland constant [K]


def if_below_then_else(value, upper_bound, then_value, else_value, rho=1.0):
    """Smooth (tanh) switch: ``then_value`` below ``upper_bound``, ``else_value`` above."""
    then_weight = 0.5 * (csdl.tanh((upper_bound - value) * rho) + 1)
    else_weight = 1 - then_weight
    return then_weight * then_value + else_weight * else_value


def compute_ambient_conditions_group(h_m):
    """Ambient conditions at geometric altitude ``h_m`` [m].

    Returns a ``csdl.VariableGroup`` with ``T_K``, ``P_Pa``, ``rho_kg_m3``, ``a_m_s``,
    ``mu_kg_m_s`` and ``nu_m2_s``.
    """
    T_K = if_below_then_else(h_m, H_TROPOPAUSE_M, T0_K - L_K_M * h_m, T1_K)

    P_below_Pa = P0_PA * (T_K / T0_K) ** (G_M_S2 / (L_K_M * R_M2_S2_K))
    P_above_Pa = P1_PA * csdl.exp(-G_M_S2 * (h_m - H_TROPOPAUSE_M) / (R_M2_S2_K * T_K))
    P_Pa = if_below_then_else(h_m, H_TROPOPAUSE_M, P_below_Pa, P_above_Pa)

    rho_kg_m3 = P_Pa / R_M2_S2_K / T_K
    a_m_s = (GAMMA * R_M2_S2_K * T_K) ** 0.5
    mu_kg_m_s = MU_REF_KG_M_S * (T_K / T_REF_K) ** 1.5 * (T_REF_K + S_K) / (T_K + S_K)
    nu_m2_s = mu_kg_m_s / rho_kg_m3

    ambient_conditions_group = csdl.VariableGroup()
    ambient_conditions_group.T_K = T_K
    ambient_conditions_group.P_Pa = P_Pa
    ambient_conditions_group.rho_kg_m3 = rho_kg_m3
    ambient_conditions_group.a_m_s = a_m_s
    ambient_conditions_group.mu_kg_m_s = mu_kg_m_s
    ambient_conditions_group.nu_m2_s = nu_m2_s
    return ambient_conditions_group
