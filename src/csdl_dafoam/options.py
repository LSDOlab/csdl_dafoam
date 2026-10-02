"""DAFoam / IDWarp option dictionaries for the bundled NACA 0012 cases."""
from pathlib import Path

__all__ = ["R_AIR", "naca0012_options", "options_for_case", "idwarp_options"]

R_AIR = 287.0  # gas constant used by the reference case to form its normalizing density


def naca0012_options(
    U0=238.0,
    p0=101325.0,
    T0=300.0,
    nuTilda0=4.5e-5,
    A0=0.1,
    primal_min_res_tol=1.0e-8,
    adjoint_rel_tol=1.0e-6,
):
    """DAFoam ``daOptions`` for the bundled NACA 0012 cases (``DARhoSimpleCFoam``).

    Parameters
    ----------
    U0, p0, T0 : float
        Reference freestream speed [m/s], pressure [Pa], temperature [K]. They set the
        far-field boundary values, the state normalization and the force scaling
        ``1 / (0.5 rho0 U0^2 A0)`` that turns forces into CL and CD.
    nuTilda0 : float
        Far-field Spalart-Allmaras variable [m^2/s]; use ~4.5e-9 with the Euler-mode case.
    A0 : float
        Reference area [m^2]: chord (1 m) times the case span (0.1 m).
    adjoint_rel_tol : float
        Relative tolerance of the adjoint's GMRES solve. The default 1e-6 is DAFoam's tutorial value. Measured on the bundled 4,032-cell
        mesh (``docs/examples.md``): 1e-4 is about 20% faster per adjoint with derivatives changed by 3e-5 (relative), and 1e-3 is about
        29% faster with 7e-4, both far below finite-difference noise.
    """
    rho0 = p0 / T0 / R_AIR
    force_scale = 1.0 / (0.5 * U0 * U0 * A0 * rho0)

    return {
        "designSurfaces": ["wing"],
        "solverName": "DARhoSimpleCFoam",
        "primalMinResTol": primal_min_res_tol,
        "primalBC": {
            "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
            "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
            "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
            "useWallFunction": True,
        },
        "function": {
            "CD": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "patchVelocityInputName": "patch_velocity",
                "direction": [1, 0, 0],
                "scale": force_scale,
            },
            "CL": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "patchVelocityInputName": "patch_velocity",
                "scale": force_scale,
            },
        },
        "adjEqnOption": {
            "gmresRelTol": adjoint_rel_tol,
            "pcFillLevel": 1,
            "jacMatReOrdering": "rcm",
            "useNonZeroInitGuess": False,
        },
        # transonic preconditioner to speed up the adjoint convergence
        "transonicPCOption": 1,
        "normalizeStates": {
            "U": U0,
            "p": p0,
            "T": T0,
            "nuTilda": nuTilda0 * 10.0,
            "phi": 1.0,
        },
        "inputInfo": {
            "aero_vol_coords": {
                "type": "volCoord",
                "components": ["solver", "function"],
            },
            "patch_velocity": {
                "type": "patchVelocity",
                "patches": ["inout"],
                "flowAxis": "x",
                "normalAxis": "z",
                "components": ["solver", "function"],
            },
            "pressure": {
                "type": "patchVar",
                "varName": "p",
                "varType": "scalar",
                "patches": ["inout"],
                "components": ["solver", "function"],
            },
            "temperature": {
                "type": "patchVar",
                "varName": "T",
                "varType": "scalar",
                "patches": ["inout"],
                "components": ["solver", "function"],
            },
        },
    }


def options_for_case(name, **overrides):
    """DAFoam options matching a bundled case from :func:`csdl_dafoam.cases.list_cases`.

    The Euler-mode case has its viscosity (and with it the far-field turbulence seed)
    lowered by 1e4, so its default ``nuTilda0`` is 4.5e-9 instead of 4.5e-5.
    """
    defaults = {"naca0012": {}, "naca0012_euler": {"nuTilda0": 4.5e-9}, "naca0012_coarse": {"nuTilda0": 4.5e-9}}
    if name not in defaults:
        raise KeyError(f"unknown case {name!r}; available: {sorted(defaults)}")
    return naca0012_options(**{**defaults[name], **overrides})


def idwarp_options(case_dir):
    """IDWarp options reading the volume mesh from ``case_dir`` (``constant/polyMesh``)."""
    return {
        "gridFile": str(Path(case_dir).resolve()),
        "fileType": "OpenFOAM",
        "symmetryPlanes": [],
    }
