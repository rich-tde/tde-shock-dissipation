r"""Cache the ballistic fallback estimate from the snapshot nearest 0.5 t_fb.

Match ``0.1-orbital-energy-spread.ipynb``: histogram native stellar-cell masses
in specific orbital energy, then apply the Keplerian return-time mapping to
negative-energy bin centres. The softened Paczynski-Wiita potential uses
``h = 0.6*r_p`` for the S60 1e4/1e5 runs and ``h = 0.5*r_p`` for the S50
1e6 run. Thermal energy and stellar self-gravity are excluded.

Input files
-----------
``dev.SNAPSHOT_TFB(run, 0.5)``
    A BH-frame ``snap_full_<n>.h5`` from the current ``dev.datapaths`` cooling
    runs. Read native positions, velocities, Density, Volume, Star and Time
    through ``richio.load``. Stellar selection is ``abs(Star - 1) < 1e-3``.

Output files
------------
The supplied ``output`` text path
    ``np.loadtxt(output, ndmin=2)`` gives float64 shape ``(400, 2)``.
    Column 0 is inferred return time ``T/t_fb`` from disruption, not a delay
    after the selected snapshot. Column 1 is the absolute fallback rate in
    ``Msun/yr``. Rows correspond to increasing negative-energy bin centres;
    no interpolation between bins or time integration is performed. Empty
    bins have zero rate. All values are linear, not logarithms.

    Comment headers record the source path and modification metadata, actual
    snapshot time, stellar/BH parameters, potential, binning, physical scales,
    selected/bound mass and excluded counts/masses. Matching cached settings
    are reused; changed settings or input-file metadata trigger recomputation.
    A sibling ``.partial`` file stages an atomic replacement.

Usage
-----
This is an importable helper without a CLI. From the repository root, with
``dev`` and ``works/dynamics`` on ``PYTHONPATH``::

    from fallback_energy import compute_energy_fallback

    time, rate = compute_energy_fallback(
        "1e4", "data/processed/FallbackRate/energy-1e4.txt"
    )

The returned arrays retain dimensionless and ``Msun/yr`` units, respectively.

Loading examples
----------------
Restore units when loading the plain-text table directly::

    import numpy as np
    import unyt as u

    table = np.loadtxt("data/processed/FallbackRate/energy-1e4.txt", ndmin=2)
    time = u.unyt_array(table[:, 0], "dimensionless")
    rate = u.unyt_array(table[:, 1], "Msun/yr")
"""

import os
import re

import numpy as np
import unyt as u
from dev.datapaths import TDE_PARAMETERS

import dev
import richio


def _cache_key(header):
    # Ignore last-bit CPU differences in derived scales, not input settings.
    return re.sub(
        r"(actual_t_over_tfb|t_fb_day|DeltaE_erg_per_g)=([^;\n]+)",
        lambda match: f"{match[1]}={float(match[2]):.12g}",
        header,
    )


def compute_energy_fallback(run, output):
    """Return unitful ``(T/t_fb, mdot)`` and save or reuse the text cache."""
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    M_BH = mbh * richio.units.mscale
    M_STAR = mstar * richio.units.mscale
    R_STAR = rstar * richio.units.lscale
    G_CODE = richio.units.lscale**3 / (richio.units.mscale * richio.units.tscale**2)
    gm = G_CODE * M_BH
    r_p = R_STAR * (M_BH / M_STAR) ** (1 / 3)  # beta = 1
    r_s = 4.21 * (M_BH / (1e6 * richio.units.mscale)) * richio.units.lscale
    softening = 0.5 if run == "1e6" else 0.6
    h = softening * r_p
    delta_epsilon = gm * R_STAR / r_p**2
    t_fb = 2 * np.pi * gm / (2 * delta_epsilon) ** 1.5

    path = dev.SNAPSHOT_TFB(run, 0.5)
    assert os.path.basename(path).startswith("snap_full_"), "Need a BH-frame snapshot"
    snap = richio.load(path)
    time = (snap.time.squeeze() / t_fb).to("dimensionless")
    source = os.stat(path)
    settings = (
        "T_over_tfb mdot_Msun_per_yr\n"
        f"run={run}; source={path}\n"
        f"source_size={source.st_size}; source_mtime_ns={source.st_mtime_ns}\n"
        f"snapshot={snap.snapnum}; requested_t_over_tfb=0.5; actual_t_over_tfb={float(time):.17g}\n"
        f"M_BH_code_mass={mbh:g}; M_STAR_code_mass={mstar:g}; R_STAR_code_length={rstar:g}; beta=1\n"
        f"G_code=1; r_s_code_length={4.21 * mbh / 1e6:.17g}; h_over_rp={softening:g}\n"
        "potential=-GM/(max(r,h)-rs)+GM*(r^2-h^2)/(2*h*(h-rs)^2) for r<h\n"
        "selection=abs(Star-1)<1e-3; mass=Density*Volume; epsilon=v^2/2+potential\n"
        "energy_edges_over_DeltaE=linspace(-10,5,601); DeltaE=GM*R_STAR/r_p^2\n"
        "T=2*pi*GM/(-2*epsilon)^1.5; mdot=(dM/depsilon)*2*abs(epsilon)/(3*T)\n"
        f"t_fb_day={t_fb.to_value('day'):.17g}; DeltaE_erg_per_g={delta_epsilon.to_value('erg/g'):.17g}"
    )
    output = os.fspath(output)
    expected = "".join(f"# {line}\n" for line in settings.splitlines())
    if os.path.exists(output):
        with open(output) as handle:
            stored = "".join(handle.readline() for _ in settings.splitlines())
        matches = _cache_key(stored) == _cache_key(expected)
        if matches:
            table = np.loadtxt(output, ndmin=2)
            print(f"{run}: reuse {output}; energy snapshot t/t_fb={float(time):.6f}")
            return u.unyt_array(table[:, 0], "dimensionless"), u.unyt_array(
                table[:, 1], "Msun/yr"
            )

    star = snap.mask_star_ratio()
    radius = np.sqrt(snap.X[star] ** 2 + snap.Y[star] ** 2 + snap.Z[star] ** 2)
    speed_squared = snap.vx[star] ** 2 + snap.vy[star] ** 2 + snap.vz[star] ** 2
    mass = (snap.density * snap.volume)[star]
    potential = -gm / (np.maximum(radius, h) - r_s)
    inner = radius < h
    potential[inner] += gm / (2 * h * (h - r_s) ** 2) * (radius[inner] ** 2 - h**2)
    epsilon = 0.5 * speed_squared + potential
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
    print(diagnostics.lstrip())
    return return_time, rate
