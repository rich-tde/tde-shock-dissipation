import re
import warnings
import os

import numpy as np
import unyt as u

################################################################################
# Shock Tube helpers                                                           # 
################################################################################

def get_at_x(snap,
           quantity, 
           x):
    """
    Get quantity at x=indices by nearest neighbor, for 1D only.
    """
    try:
        distance = np.abs(snap.x - x)
    except u.UnitOperationError:
        distance = np.abs(snap.x - x * snap.x.units)
    i = np.argmin(distance)
    q_x0 = quantity[i]
    return q_x0

def parse_gamma(snap_dir, default=5/3):
    """Parse ideal-gas gamma from path, or return default."""
    m = re.search(r'Gamma([0-9.]+)', snap_dir)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    warnings.warn(
        f"Failed to parse gamma from directory {snap_dir}, using default={default}"
    )
    return default

def parse_ic(
    snap_dir,
    default_P_L=1.0,
    default_P_R=0.1,
    default_rho_L=1.0,
    default_rho_R=0.125,
):
    """
    Parse PL, PR, DL, DR from path, or return defaults.
    Requires full PL..PR..DL..DR match.
    """
    m = re.search(
        r'PL([0-9.]+)PR([0-9.]+)DL([0-9.]+)DR([0-9.]+)',
        snap_dir
    )

    if m:
        try:
            P_L   = float(m.group(1))
            P_R   = float(m.group(2))
            rho_L = float(m.group(3))
            rho_R = float(m.group(4))
            return P_L, P_R, rho_L, rho_R
        except ValueError:
            pass
    
    warnings.warn(
        "No PL/PR/DL/DR block found in path, using sod shock defaults"
    )
    return default_P_L, default_P_R, default_rho_L, default_rho_R

def fetch_ic(
    snap_dir,
    default_P_L=1.0,
    default_P_R=0.1,
    default_rho_L=1.0,
    default_rho_R=0.125,
):
    """
    Fetch ICs from files in snap_dir:
      - leftdensity.txt
      - rightdensity.txt
      - leftpressure.txt
      - rightpressure.txt

    Each file must contain a single number.
    """

    files = {
        "rho_L": ("leftdensity.txt",  default_rho_L),
        "rho_R": ("rightdensity.txt", default_rho_R),
        "P_L":   ("leftpressure.txt", default_P_L),
        "P_R":   ("rightpressure.txt", default_P_R),
    }

    values = {}

    for key, (fname, default) in files.items():
        path = os.path.join(snap_dir, fname)
        try:
            with open(path, "r") as f:
                values[key] = float(f.read().strip())
        except Exception:
            warnings.warn(
                f"Failed to read {fname}, using default={default}: {path}"
            )
            values[key] = default

    return values["P_L"], values["P_R"], values["rho_L"], values["rho_R"]

def get_shock_tube_front(x, diss, right=True):
    """Get shock front from max dissipation for 1D shock tube.
    Setting right=True gets only the right propagating shock.
    """
    diss = diss.copy() # avoid modifying the original array
    if right is True:
        diss[x.value < 0] = 0
    i_sh = np.argmax(diss)
    if right is True:
        assert x[i_sh] > 0, "Shock front should be positive"

    return i_sh

################################################################################
# Physical equations                                                           #
################################################################################

def P_poisson(rho, P_ref, rho_ref, gamma=5/3):
    return P_ref * (rho / rho_ref) ** gamma

def P_hugoniot(rho, P_ref, rho_ref, gamma=5/3): # uses rho2, P2 as reference point
    return P_ref * ((gamma + 1)*rho - (gamma - 1)*rho_ref) / ((gamma + 1)*rho_ref - (gamma - 1)*rho)

def P_rayleigh(v, v1, v2, P1, P2, gamma=5/3):
    return P1 + (P2 - P1) * (v - v1) / (v2 - v1)

def dp2s(rho, p, gamma=5/3):
    """
    Using the Sackur-Tetrode equation to calculate specific entropy of a
    ideal gas, given pressure and density.

    Note: the Sackur-Tetrode equation as written here is only for 5/3 idea
    gas...
    """
    sie = p / (rho * (gamma - 1)) # specific internal energy
    s = u.kb / u.mh * (
        np.log(u.mh / rho * ((4*np.pi*u.mh**2*sie)/(3*u.h**2))**(3/2)) + 5/2
        ) 
    return s