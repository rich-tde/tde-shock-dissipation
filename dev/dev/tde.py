"""Run-level TDE physics of the RICH runs: scales, Paczynski–Wiita potential and orbit, BH frame."""

import os
import re
from itertools import combinations

import numpy as np
import richio
import unyt as u
from scipy.integrate import solve_ivp

from dev.datapaths import SNAPSHOT_TFB, TDE_PARAMETERS
from dev.helpers import TDEBasics, positions_and_time, snapshot_time

CODE_LENGTH = richio.units.lscale
CODE_TIME = richio.units.tscale
FALLBACK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "FallbackRate")
)


def tde_basics(run):
    """``TDEBasics`` of a run."""
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    return TDEBasics(
        m_bh * richio.units.mscale, m_star * richio.units.mscale, r_star * CODE_LENGTH
    )


def softening_radius(run):
    """Radius of RICH's harmonic gravity core: 0.5 r_p for 1e6, 0.6 r_p otherwise."""
    return (0.5 if run == "1e6" else 0.6) * tde_basics(run).r_p


def schwarzschild_radius(run):
    """RICH's Rg = 4.21 code lengths per 1e6 M_sun (≈ 2GM/c²) in its potential -GM/(r - Rg)."""
    return 4.21 * TDE_PARAMETERS[run][0] / 1e6 * CODE_LENGTH


def pw_potential(radius, run):
    """RICH's softened Paczynski–Wiita potential per unit mass."""
    gm = tde_basics(run).Mbh * u.G
    r_s, h = schwarzschild_radius(run), softening_radius(run)
    potential = -gm / (np.maximum(radius, h) - r_s)
    inner = radius < h
    potential[inner] += gm / (2 * h * (h - r_s) ** 2) * (radius[inner] ** 2 - h**2)
    return potential


def _pw_acceleration(_, state, gm, r_s):
    x, y, vx, vy = state
    radius = np.hypot(x, y)
    acceleration = -gm / (radius * (radius - r_s) ** 2)
    return vx, vy, acceleration * x, acceleration * y


def _pw_orbit(run, start, speed, duration, **options):
    """solve_ivp of RICH's PW orbit from ``(start, 0)`` with velocity ``(0, -speed)``, in code units."""
    gm = (tde_basics(run).Mbh * u.G).to_value(CODE_LENGTH**3 / CODE_TIME**2)
    return solve_ivp(
        _pw_acceleration,
        (0.0, duration.to_value(CODE_TIME)),
        (
            start.to_value(CODE_LENGTH),
            0.0,
            0.0,
            -speed.to_value(CODE_LENGTH / CODE_TIME),
        ),
        args=(gm, schwarzschild_radius(run).to_value(CODE_LENGTH)),
        **options,
    )


def needs_frame_switch(run, path):
    """Whether a snapshot is stored in the star frame and must be moved to the BH frame."""
    if run == "ss24":
        return os.path.basename(os.path.dirname(path)) == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", os.path.basename(path)) is not None


def star_frame_offset(run, time):
    """Position and velocity ``(x, y, vx, vy)`` of the star-frame origin in the BH frame at ``time``.

    RICH moves the star frame along the parabolic PW orbit through pericentre at time 0.
    """
    r_p = tde_basics(run).r_p
    speed = np.sqrt(2 * tde_basics(run).Mbh * u.G / (r_p - schwarzschild_radius(run)))
    solution = _pw_orbit(run, r_p, speed, time, method="DOP853", rtol=1e-10, atol=1e-12)
    x, y, vx, vy = solution.y[:, -1]
    velocity = CODE_LENGTH / CODE_TIME
    return x * CODE_LENGTH, y * CODE_LENGTH, vx * velocity, vy * velocity


def bh_frame_positions(snap, run, path):
    """Cell positions in the BH frame."""
    x, y, z, time = positions_and_time(snap)
    if needs_frame_switch(run, path):
        offset_x, offset_y, _, _ = star_frame_offset(run, time)
        x, y = x + offset_x, y + offset_y
    return x, y, z


def bh_frame_velocities(snap, run, path):
    """Cell velocities in the BH frame."""
    vx, vy, vz = snap.vx, snap.vy, snap.vz
    if needs_frame_switch(run, path):
        _, _, offset_vx, offset_vy = star_frame_offset(run, snapshot_time(snap))
        vx, vy = vx + offset_vx, vy + offset_vy
    return vx, vy, vz


def returning_orbit(run, time_tfb, radius_scale=1.0):
    """Ballistic PW orbit of the debris returning at ``time_tfb``, from pericentre onwards.

    Returns x, y at 50000 times and the index of the next pericentre.
    """
    tde = tde_basics(run)
    energy = -tde.energy_spread * time_tfb ** (-2 / 3)
    start = radius_scale * tde.r_p
    speed = np.sqrt(2 * (tde.Mbh * u.G / (start - schwarzschild_radius(run)) + energy))
    period = tde.orbital_period(energy).to_value(CODE_TIME)
    solution = _pw_orbit(
        run,
        start,
        speed,
        1.6 * period * CODE_TIME,
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=period / 5000,
    )
    time = np.linspace(0.0, solution.t[-1], 50000)
    x, y = solution.sol(time)[:2]
    radius = np.hypot(x, y)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    return x * CODE_LENGTH, y * CODE_LENGTH, minima[time[minima] > 0.5 * period][0]


def returning_orbit_window(
    run,
    time_tfb,
    half_length_rp,
    resolution,
    radius_scale=1.0,
    phase_offset_deg=0.0,
    center_shift=0 * CODE_LENGTH,
):
    """Evenly spaced azimuths along the returning orbit, around its pericentre.

    The window spans ``±half_length_rp`` of arc (in units of r_p), is rotated by
    ``phase_offset_deg`` and slid along the orbit by ``center_shift``. Returns
    ``phi`` (zero at pericentre), ``x/r_p``, ``y/r_p`` and the signed arc
    coordinate ``s`` (positive before pericentre).
    """
    r_p = tde_basics(run).r_p
    x, y, peri = returning_orbit(run, time_tfb, radius_scale)
    x_rp, y_rp = (
        (x / r_p).to_value("dimensionless"),
        (y / r_p).to_value("dimensionless"),
    )
    shift = (center_shift / r_p).to_value("dimensionless")
    phi = np.unwrap(np.arctan2(y_rp, x_rp))
    phi -= phi[peri]
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_rp), np.diff(y_rp)))))

    phi_start, phi_end = np.interp(
        arc[peri] + np.array([half_length_rp, -half_length_rp]), arc, phi
    )
    offset = np.deg2rad(phase_offset_deg)
    window_phi = np.linspace(phi_start + offset, phi_end + offset, resolution, endpoint=False)
    window_arc = np.interp(window_phi, phi[::-1], arc[::-1])
    s = (arc[peri] - window_arc + shift) * r_p
    window_phi = np.interp(window_arc - shift, arc, phi)
    return (
        window_phi,
        np.interp(window_phi, phi[::-1], x_rp[::-1]),
        np.interp(window_phi, phi[::-1], y_rp[::-1]),
        s,
    )


def ballistic_fallback(run, tfb=0.15):
    """Ballistic fallback rate ``(t/t_fb, Mdot_fb)`` of the stellar debris at ``tfb``, cached after the first call."""
    path = os.path.join(FALLBACK_DIR, f"energy-{run}-t{tfb:g}.txt")
    if not os.path.exists(path):
        snapshot = SNAPSHOT_TFB(run, tfb)
        np.savetxt(
            path,
            np.column_stack(_debris_fallback(run, snapshot)),
            header=f"t/t_fb, Mdot_fb [Msun/yr] from {snapshot}",
        )
    time, rate = np.loadtxt(path, unpack=True)
    return time, u.unyt_array(rate, "Msun/yr")


def _debris_fallback(run, path):
    tde = tde_basics(run)
    snap = richio.load(path)
    star = snap.mask_star_ratio()
    x, y, z = bh_frame_positions(snap, run, path)
    vx, vy, vz = bh_frame_velocities(snap, run, path)
    radius = np.sqrt(x[star] ** 2 + y[star] ** 2 + z[star] ** 2)
    specific_energy = 0.5 * (vx[star] ** 2 + vy[star] ** 2 + vz[star] ** 2) + pw_potential(
        radius, run
    )
    edges = (np.linspace(-10, 5, 601) * tde.energy_spread).to(specific_energy.units)
    mass, _ = np.histogram(specific_energy, bins=edges, weights=(snap.density * snap.volume)[star])
    centre = 0.5 * (edges[1:] + edges[:-1])
    bound = centre < 0
    period = tde.orbital_period(centre[bound])
    rate = mass[bound] / np.diff(edges)[bound] * 2 * -centre[bound] / (3 * period)
    return (period / tde.t_fb).to_value("dimensionless"), rate.to_value("Msun/yr")


def mass_scaling(curves, time):
    """Local index d log y / d log M_bh on ``time`` between every pair of runs in ``curves`` = {run: (t, y)}."""
    values = {
        run: np.interp(time, t, y, left=np.nan, right=np.nan) for run, (t, y) in curves.items()
    }
    return {
        (run_a, run_b): np.log10(values[run_a] / values[run_b])
        / np.log10(TDE_PARAMETERS[run_a][0] / TDE_PARAMETERS[run_b][0])
        for run_a, run_b in combinations(curves, 2)
    }
