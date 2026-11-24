"""TDE scales, the Paczynski–Wiita orbit and the black-hole frame, in RICH code units (G = 1)."""

import os
import re

import numpy as np
from scipy.integrate import solve_ivp

from dev.datapaths import TDE_PARAMETERS


def pericenter(run):
    """Tidal radius for beta = 1, in code lengths."""
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    return r_star * (m_bh / m_star) ** (1 / 3)


def schwarzschild_radius(m_bh):
    """The ``Rg = 4.21 M / 1e6`` (≈ 2GM/c²) of RICH's potential -M/(r - Rg), in code lengths."""
    return 4.21 * m_bh / 1e6


def pw_orbit_derivatives(_, state, m_bh):
    x, y, vx, vy = state
    radius = np.hypot(x, y)
    acceleration = -m_bh / (radius * (radius - schwarzschild_radius(m_bh)) ** 2)
    return vx, vy, acceleration * x, acceleration * y


def needs_frame_switch(run, path):
    """Whether a snapshot is stored in the star frame and must be moved to the BH frame."""
    if run == "ss24":
        return os.path.basename(os.path.dirname(path)) == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", os.path.basename(path)) is not None


def star_frame_offset(run, time):
    """Position and velocity ``(x, y, vx, vy)`` of the star-frame origin in the BH frame.

    Integrates RICH's parabolic Paczynski–Wiita orbit from pericentre at code time 0.
    """
    m_bh = TDE_PARAMETERS[run][0]
    r_p = pericenter(run)
    speed = np.sqrt(2 * m_bh / (r_p - schwarzschild_radius(m_bh)))
    solution = solve_ivp(
        pw_orbit_derivatives,
        (0.0, time),
        (r_p, 0.0, 0.0, -speed),
        args=(m_bh,),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    return solution.y[:, -1]


def bh_frame_positions(snap, run, path):
    """Cell positions in the BH frame, as unyt arrays."""
    import richio

    x, y, z = snap.X, snap.Y, snap.Z
    if needs_frame_switch(run, path):
        time = float(np.ravel(snap.time.to_value(richio.units.tscale))[0])
        offset_x, offset_y, _, _ = star_frame_offset(run, time)
        x = x + offset_x * richio.units.lscale
        y = y + offset_y * richio.units.lscale
    return x, y, z


def returning_orbit(run, time_tfb, radius_scale=1.0):
    """Ballistic PW orbit of the debris returning at ``time_tfb``, from pericentre onwards.

    Returns x, y (code lengths) at 50000 times and the index of the next pericentre.
    """
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    start = radius_scale * pericenter(run)
    energy_spread = m_star * (m_bh / m_star) ** (1 / 3) / r_star
    binding_energy = -energy_spread * time_tfb ** (-2 / 3)
    speed = np.sqrt(2 * (m_bh / (start - schwarzschild_radius(m_bh)) + binding_energy))
    period = 2 * np.pi * m_bh / (-2 * binding_energy) ** 1.5

    solution = solve_ivp(
        pw_orbit_derivatives,
        (0.0, 1.6 * period),
        (start, 0.0, 0.0, -speed),
        args=(m_bh,),
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=period / 5000,
    )
    time = np.linspace(0.0, solution.t[-1], 50000)
    x, y = solution.sol(time)[:2]
    radius = np.hypot(x, y)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    return x, y, minima[time[minima] > 0.5 * period][0]


def returning_orbit_window(
    run,
    time_tfb,
    half_length_rp,
    resolution,
    radius_scale=1.0,
    phase_offset_deg=0.0,
    center_shift=0.0,
):
    """Evenly spaced azimuths along the returning orbit, around its pericentre.

    The window spans ``±half_length_rp`` of arc (in units of r_p), is rotated by
    ``phase_offset_deg`` and slid along the orbit by ``center_shift`` code lengths.
    Returns ``phi`` (zero at pericentre), ``x/r_p``, ``y/r_p`` and the signed arc
    coordinate ``s`` in code lengths (positive before pericentre).
    """
    r_p = pericenter(run)
    x, y, peri = returning_orbit(run, time_tfb, radius_scale)
    x_rp, y_rp = x / r_p, y / r_p
    phi = np.unwrap(np.arctan2(y, x))
    phi -= phi[peri]
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_rp), np.diff(y_rp)))))

    phi_start, phi_end = np.interp(
        arc[peri] + np.array([half_length_rp, -half_length_rp]), arc, phi
    )
    offset = np.deg2rad(phase_offset_deg)
    window_phi = np.linspace(phi_start + offset, phi_end + offset, resolution, endpoint=False)
    window_arc = np.interp(window_phi, phi[::-1], arc[::-1])
    s = -(window_arc - arc[peri]) * r_p + center_shift
    window_phi = np.interp(window_arc - center_shift / r_p, arc, phi)
    return (
        window_phi,
        np.interp(window_phi, phi[::-1], x_rp[::-1]),
        np.interp(window_phi, phi[::-1], y_rp[::-1]),
        s,
    )
