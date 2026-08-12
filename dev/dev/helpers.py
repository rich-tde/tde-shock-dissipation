import os
import re
import warnings

import numpy as np
import unyt as u
from scipy.integrate import solve_ivp

################################################################################
# Shock Tube helpers                                                           #
################################################################################


def get_at_x(snap, quantity, x):
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


def parse_gamma(snap_dir, default=5 / 3):
    """Parse ideal-gas gamma from path, or return default."""
    m = re.search(r"Gamma([0-9.]+)", snap_dir)
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
    m = re.search(r"PL([0-9.]+)PR([0-9.]+)DL([0-9.]+)DR([0-9.]+)", snap_dir)

    if m:
        try:
            P_L = float(m.group(1))
            P_R = float(m.group(2))
            rho_L = float(m.group(3))
            rho_R = float(m.group(4))
            return P_L, P_R, rho_L, rho_R
        except ValueError:
            pass

    warnings.warn("No PL/PR/DL/DR block found in path, using sod shock defaults")
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
        "rho_L": ("leftdensity.txt", default_rho_L),
        "rho_R": ("rightdensity.txt", default_rho_R),
        "P_L": ("leftpressure.txt", default_P_L),
        "P_R": ("rightpressure.txt", default_P_R),
    }

    values = {}

    for key, (fname, default) in files.items():
        path = os.path.join(snap_dir, fname)
        try:
            with open(path, "r") as f:
                values[key] = float(f.read().strip())
        except Exception:
            warnings.warn(f"Failed to read {fname}, using default={default}: {path}")
            values[key] = default

    return values["P_L"], values["P_R"], values["rho_L"], values["rho_R"]


def get_shock_tube_front(x, diss, right=True):
    """Get shock front from max dissipation for 1D shock tube.
    Setting right=True gets only the right propagating shock.
    """
    diss = diss.copy()  # avoid modifying the original array
    if right is True:
        diss[x.value < 0] = 0
    i_sh = np.argmax(diss)
    if right is True:
        assert x[i_sh] > 0, "Shock front should be positive"

    return i_sh


################################################################################
# Physical equations                                                           #
################################################################################


def P_poisson(rho, P_ref, rho_ref, gamma=5 / 3):
    return P_ref * (rho / rho_ref) ** gamma


def P_hugoniot(rho, P_ref, rho_ref, gamma=5 / 3):  # uses rho2, P2 as reference point
    return (
        P_ref
        * ((gamma + 1) * rho - (gamma - 1) * rho_ref)
        / ((gamma + 1) * rho_ref - (gamma - 1) * rho)
    )


def P_rayleigh(v, v1, v2, P1, P2):
    return P1 + (P2 - P1) * (v - v1) / (v2 - v1)


def dp2s(rho, p):
    """
    Using the Sackur-Tetrode equation to calculate specific entropy of a
    ideal gas, given pressure and density.

    Note: the Sackur-Tetrode equation as written here is only for 5/3 idea
    gas!
    """
    gamma = 5 / 3
    sie = p / (rho * (gamma - 1))  # specific internal energy
    s = (
        u.kb
        / u.mh
        * (
            np.log(u.mh / rho * ((4 * np.pi * u.mh**2 * sie) / (3 * u.h**2)) ** (3 / 2))
            + 5 / 2
        )
    )
    return s


def delta(M, gamma=5 / 3):
    R = 1 / ((gamma - 1) / (gamma + 1) + 2 / (gamma + 1) / M**2)  # R = rho2/rho1
    delta = (
        2
        / (gamma * (gamma - 1) * M**2 * R)
        * ((2 * gamma * M**2 - (gamma - 1)) / (gamma + 1) - R**gamma)
    )
    return delta


class TDEBasics:
    """Frequently used scales for a tidal disruption event.

    Parameters are unyt masses/radii and the dimensionless penetration
    factor ``beta``.  The calculated quantities are available as attributes::

        tde = TDEBasics(Mbh, Mstar, Rstar, beta=1)
        tde.r_p
        tde.r_a
        tde.t_fb
    """

    def __init__(self, Mbh, Mstar, Rstar, beta=1):
        self.Mbh = Mbh
        self.Mstar = Mstar
        self.Rstar = Rstar
        self.beta = beta

        self.r_t = Rstar * (Mbh / Mstar) ** (1 / 3)
        self.r_p = self.r_t / beta
        self.r_a = Rstar * (Mbh / Mstar) ** (2 / 3)
        self.t_fb = (
            np.pi
            / np.sqrt(2)
            * np.sqrt(Rstar**3 / (u.G * Mstar))
            * np.sqrt(Mbh / Mstar)
        )


################################################################################
# Orbit / reference frame                                                      #
################################################################################


def _paczynski_orbit(t, state, GM, Rg):
    x, y, vx, vy = state
    r = np.hypot(x, y)
    accel = -GM / (r * (r - Rg) ** 2)
    return vx, vy, accel * x, accel * y


def get_true_anomaly(t, Mbh, Rp):
    """Integrate the Paczynski-Wiita orbit from pericenter to time t.

    Mirrors GetTrueAnomaly in RICH's test.cpp. The star starts at pericenter
    (Rp, 0) moving in -y; the returned (x, y, vx, vy) is the orbiting frame's
    origin in the BH frame, as unyt quantities.

    solve_ivp cannot carry unyt through the state vector, so we integrate in
    cgs floats and reattach units on the way out.
    """
    GM = (u.G * Mbh).in_cgs()
    Rg = (u.G * Mbh / u.c**2).in_cgs()
    Rp = Rp.in_cgs()
    t = t.in_cgs()
    v_p = np.sqrt(2 * GM / (Rp - Rg))
    state0 = [Rp.value, 0.0, 0.0, -v_p.value]
    sol = solve_ivp(
        _paczynski_orbit,
        (0.0, t.value),
        state0,
        args=(GM.value, Rg.value),
        method="RK45",
        rtol=1e-8,
        atol=1e-11,
        first_step=t.value * 1e-5 if t.value > 0 else None,
    )
    x, y, vx, vy = sol.y[:, -1]
    return x * u.cm, y * u.cm, vx * u.cm / u.s, vy * u.cm / u.s


def reference_frame_offset(t, Mbh, Mstar, Rstar, beta):
    """Offset (dx, dy, dvx, dvy) from the orbiting frame to the BH frame.

    Add these to a snapshot's (X, Y, vx, vy) to place it in the
    BH-centered frame. z and vz are unchanged since the orbit is planar.
    """
    Rt = Rstar * (Mbh / Mstar) ** (1 / 3)
    Rp = Rt / beta
    return get_true_anomaly(t, Mbh, Rp)
