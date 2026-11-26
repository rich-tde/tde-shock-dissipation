"""TDE scales, the Paczynski–Wiita orbit and the black-hole frame, in RICH code units (G = 1)."""

import os
import re
from itertools import combinations

import numpy as np
import unyt as u
from scipy.integrate import solve_ivp

from dev.datapaths import SNAPSHOT_TFB, TDE_PARAMETERS, snapshot_number
from dev.helpers import positions_and_time


def pericenter(run):
    """Tidal radius for beta = 1, in code lengths."""
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    return r_star * (m_bh / m_star) ** (1 / 3)


def bh_gm(run):
    """G M_bh with unyt's G."""
    import richio

    return TDE_PARAMETERS[run][0] * richio.units.mscale * u.G


def code_gm(run):
    """``bh_gm`` as a float in code units, for the orbit integrations."""
    import richio

    units = richio.units
    return float(bh_gm(run).to_value(units.lscale**3 / units.tscale**2))


def fallback_time(run):
    """Return time of the most bound debris for beta = 1, in code times."""
    import richio

    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p = pericenter(run) * richio.units.lscale
    delta_epsilon = bh_gm(run) * r_star * richio.units.lscale / r_p**2
    t_fb = 2 * np.pi * bh_gm(run) / (2 * delta_epsilon) ** 1.5
    return float(t_fb.to_value(richio.units.tscale))


def mass_scaling(curves, time):
    """Local index d log y / d log M_bh on ``time`` between every pair of runs in ``curves`` = {run: (t, y)}."""
    values = {
        run: np.interp(time, t, y, left=np.nan, right=np.nan)
        for run, (t, y) in curves.items()
    }
    return {
        (run_a, run_b): np.log10(values[run_a] / values[run_b])
        / np.log10(TDE_PARAMETERS[run_a][0] / TDE_PARAMETERS[run_b][0])
        for run_a, run_b in combinations(curves, 2)
    }


def softening_radius(run):
    """Radius of RICH's harmonic gravity core, in code lengths: 0.5 r_p for 1e6, 0.6 r_p otherwise."""
    return (0.5 if run == "1e6" else 0.6) * pericenter(run)


def schwarzschild_radius(m_bh):
    """The ``Rg = 4.21 M / 1e6`` (≈ 2GM/c²) of RICH's potential -M/(r - Rg), in code lengths."""
    return 4.21 * m_bh / 1e6


def pw_potential(radius, run):
    """RICH's softened Paczynski–Wiita potential per unit mass at a unyt ``radius``."""
    import richio

    units = richio.units
    m_bh = TDE_PARAMETERS[run][0]
    gm = bh_gm(run)
    r_s = schwarzschild_radius(m_bh) * units.lscale
    h = softening_radius(run) * units.lscale
    potential = -gm / (np.maximum(radius, h) - r_s)
    inner = radius < h
    potential[inner] += gm / (2 * h * (h - r_s) ** 2) * (radius[inner] ** 2 - h**2)
    return potential


def pw_orbit_derivatives(_, state, gm, r_s):
    x, y, vx, vy = state
    radius = np.hypot(x, y)
    acceleration = -gm / (radius * (radius - r_s) ** 2)
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
    gm = code_gm(run)
    r_s = schwarzschild_radius(TDE_PARAMETERS[run][0])
    r_p = pericenter(run)
    speed = np.sqrt(2 * gm / (r_p - r_s))
    solution = solve_ivp(
        pw_orbit_derivatives,
        (0.0, time),
        (r_p, 0.0, 0.0, -speed),
        args=(gm, r_s),
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


def _cache_key(header):
    return re.sub(
        r"(actual_t_over_tfb|t_fb_day|DeltaE_erg_per_g)=([^;\n]+)",
        lambda match: f"{match[1]}={float(match[2]):.12g}",
        header,
    )


def ballistic_fallback(run, output, tfb=0.5):
    """Ballistic ``(T/t_fb, Mdot_fb)`` from the stellar debris energies at ``tfb``, cached as text at ``output``."""
    import richio

    units = richio.units
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    gm = bh_gm(run)
    r_p = pericenter(run) * units.lscale
    delta_epsilon = gm * r_star * units.lscale / r_p**2
    t_fb = fallback_time(run) * units.tscale

    path = SNAPSHOT_TFB(run, tfb)
    snap = richio.load(path)
    x, y, z, snap_time = positions_and_time(snap)
    time = (snap_time / t_fb).to("dimensionless")
    source = os.stat(path)
    settings = (
        "T_over_tfb mdot_Msun_per_yr\n"
        f"run={run}; source={path}\n"
        f"source_size={source.st_size}; source_mtime_ns={source.st_mtime_ns}\n"
        f"snapshot={snapshot_number(path)}; requested_t_over_tfb={tfb:g}; actual_t_over_tfb={float(time):.17g}\n"
        f"M_BH_code_mass={m_bh:g}; M_STAR_code_mass={m_star:g}; R_STAR_code_length={r_star:g}; beta=1\n"
        f"G=unyt.G; r_s_code_length={schwarzschild_radius(m_bh):.17g}; h_over_rp={softening_radius(run) / pericenter(run):g}\n"
        "potential=-GM/(max(r,h)-rs)+GM*(r^2-h^2)/(2*h*(h-rs)^2) for r<h\n"
        "selection=abs(Star-1)<1e-3; mass=Density*Volume; epsilon=v^2/2+potential\n"
        "energy_edges_over_DeltaE=linspace(-10,5,601); DeltaE=GM*R_STAR/r_p^2\n"
        "T=2*pi*GM/(-2*epsilon)^1.5; mdot=(dM/depsilon)*2*abs(epsilon)/(3*T)\n"
        f"t_fb_day={t_fb.to_value('day'):.17g}; DeltaE_erg_per_g={delta_epsilon.to_value('erg/g'):.17g}"
    )
    expected = "".join(f"# {line}\n" for line in settings.splitlines())
    if os.path.exists(output):
        with open(output) as handle:
            stored = "".join(handle.readline() for _ in settings.splitlines())
        if _cache_key(stored) == _cache_key(expected):
            table = np.loadtxt(output, ndmin=2)
            print(f"{run}: reuse {output}; energy snapshot t/t_fb={float(time):.6f}")
            return u.unyt_array(table[:, 0], "dimensionless"), u.unyt_array(
                table[:, 1], "Msun/yr"
            )

    vx, vy, vz = snap.vx, snap.vy, snap.vz
    if needs_frame_switch(run, path):
        offset_x, offset_y, offset_vx, offset_vy = star_frame_offset(
            run, snap_time.to_value(units.tscale)
        )
        velocity = units.lscale / units.tscale
        x, y = x + offset_x * units.lscale, y + offset_y * units.lscale
        vx, vy = vx + offset_vx * velocity, vy + offset_vy * velocity
    star = snap.mask_star_ratio()
    radius = np.sqrt(x[star] ** 2 + y[star] ** 2 + z[star] ** 2)
    speed_squared = vx[star] ** 2 + vy[star] ** 2 + vz[star] ** 2
    mass = (snap.density * snap.volume)[star]
    epsilon = (0.5 * speed_squared + pw_potential(radius, run)).to(delta_epsilon.units)

    energy_edges = np.linspace(-10, 5, 601) * delta_epsilon
    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    bin_mass, _ = np.histogram(epsilon, bins=energy_edges, weights=mass)
    dM_depsilon = bin_mass / np.diff(energy_edges)
    bound = energy < 0
    return_time = 2 * np.pi * gm / (-2 * energy[bound]) ** 1.5
    rate = (dM_depsilon[bound] * 2 * np.abs(energy[bound]) / (3 * return_time)).to(
        "Msun/yr"
    )
    return_time = (return_time / t_fb).to("dimensionless")
    below = epsilon < energy_edges[0]
    above = epsilon > energy_edges[-1]
    assert np.isclose(
        (bin_mass.sum() + mass[below].sum() + mass[above].sum()) / mass.sum(), 1
    )

    diagnostics = (
        f"\nselected_cells={len(mass)}; selected_mass_Msun={mass.sum().to_value('Msun'):.17g}; "
        f"bound_mass_Msun={mass[epsilon < 0].sum().to_value('Msun'):.17g}\n"
        f"below_bins_cells={below.sum()}; below_bins_mass_Msun={mass[below].sum().to_value('Msun'):.17g}\n"
        f"above_bins_cells={above.sum()}; above_bins_mass_Msun={mass[above].sum().to_value('Msun'):.17g}"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savetxt(
        output + ".partial",
        np.column_stack(
            [return_time.to_value("dimensionless"), rate.to_value("Msun/yr")]
        ),
        header=settings + diagnostics,
    )
    os.replace(output + ".partial", output)
    print(f"{run}: wrote {output}; energy snapshot t/t_fb={float(time):.6f}")
    return return_time, rate


def returning_orbit(run, time_tfb, radius_scale=1.0):
    """Ballistic PW orbit of the debris returning at ``time_tfb``, from pericentre onwards.

    Returns x, y (code lengths) at 50000 times and the index of the next pericentre.
    """
    m_bh, _, r_star = TDE_PARAMETERS[run]
    gm = code_gm(run)
    r_s = schwarzschild_radius(m_bh)
    start = radius_scale * pericenter(run)
    energy_spread = gm * r_star / pericenter(run) ** 2
    binding_energy = -energy_spread * time_tfb ** (-2 / 3)
    speed = np.sqrt(2 * (gm / (start - r_s) + binding_energy))
    period = 2 * np.pi * gm / (-2 * binding_energy) ** 1.5

    solution = solve_ivp(
        pw_orbit_derivatives,
        (0.0, 1.6 * period),
        (start, 0.0, 0.0, -speed),
        args=(gm, r_s),
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=period / 5000,
    )
    time = np.linspace(0.0, solution.t[-1], 50000)
    x, y = solution.sol(time)[:2]
    radius = np.hypot(x, y)
    minima = (
        np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    )
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
    window_phi = np.linspace(
        phi_start + offset, phi_end + offset, resolution, endpoint=False
    )
    window_arc = np.interp(window_phi, phi[::-1], arc[::-1])
    s = -(window_arc - arc[peri]) * r_p + center_shift
    window_phi = np.interp(window_arc - center_shift / r_p, arc, phi)
    return (
        window_phi,
        np.interp(window_phi, phi[::-1], x_rp[::-1]),
        np.interp(window_phi, phi[::-1], y_rp[::-1]),
        s,
    )
