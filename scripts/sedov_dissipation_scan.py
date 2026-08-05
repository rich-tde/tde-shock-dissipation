#!/usr/bin/env python3
"""Scan a Sedov run and compare three dissipation measures on every snapshot.

For each snapshot this computes
  * diss_rich      -- sum(snap.dissipation * snap.volume)  (RICH's own estimate)
  * diss_analytic  -- DISS_FRACTION * E / t  (self-similar Sedov rate)
  * diss_sf        -- shock-finder estimate: sum over surface cells of
                      1/2 rho0 M_T^3 c0^3 A_cell delta(M_T)
plus shock radii (analytic, peak of rich dissipation, shock-finder median),
median Mach numbers from the T/P/rho jumps, the analytic Mach number, cell
counts and surface areas.

Outputs (all under OUTPUT_DIR):
  * sedov_dissipation.txt        -- one row per snapshot, unyt-annotated
                                    (units line in the header; load with the
                                    richio unit registry, see the plotting
                                    notebook)
  * profiles/sedov_<n>.png       -- per-snapshot profile-fit figure
                                    (rho, P, v_r vs the L&L 106 solution,
                                    plus the rich dissipation profile)

Plotting of the dissipation-vs-t comparison lives in
works/shockstudy/1.2-sedov-dissipation-vs-t.ipynb so the look can be tuned
without redoing the calculation.

Run inside the richanalysis env:  python scripts/sedov_dissipation_scan.py
"""

import glob
import os
import re
import time as walltime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

import richio
from richio import shockfinder as sf

# ── Config ───────────────────────────────────────────────────────────────────
# Dataset: uncomment one.  P0 is the ambient (pre-shock) pressure of that IC.
# INPUT_DIR, P0 = "/home/hey4/RICH/alice_data/raw/Sedov3DSchaal15", 1e-4  # 125k cells, E=1
# INPUT_DIR, P0 = "/home/hey4/RICH/alice_data/raw/Sedov3DSpringel10", 1e-4  # 262k cells, E=1
# INPUT_DIR, P0 = "/home/hey4/RICH/alice_data/raw/Sedov3D", (5 / 3 - 1) * 0.1  # 5M cells, E~3347, hot ambient (sie0=0.1)
# INPUT_DIR, P0 = "/home/hey4/rich_tde/data/raw/Sedov3DSchaal15+", 1e-4  # 125k cells, E=1, longer run (to snap 332)
INPUT_DIR, P0 = (
    "/home/hey4/rich_tde/data/raw/Springel10+",
    1e-4,
)  # 262k cells, E=1, longer run (to snap 448)
SNAP_PATTERN = "sedov_*.h5"
ONLY = None  # e.g. [750] to run a subset of snapshot numbers; None = all

# Outputs land in a per-dataset subfolder, e.g. .../sedov_dissipation/Sedov3DSchaal15
OUTPUT_BASE = "/zfsstore/user/hey4/rich_tde/data/processed/sedov_dissipation"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, os.path.basename(INPUT_DIR))
TABLE_NAME = "sedov_dissipation.txt"

GAMMA = 5 / 3
KAPPA = 0.49  # Sedov energy integral for gamma=5/3: R = (E/(kappa rho0))^(1/5) t^(2/5)
DISS_FRACTION = 0.46  # analytic dE_diss/dt = DISS_FRACTION * E / t (gamma=5/3)
RHO0 = 1.0  # ambient (pre-shock) density   [code density]

NBINS_R = 200  # radial bins for locating the peak of the rich dissipation
SUBSAMPLE = 50  # plot every Nth cell in the profile scatter
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join(OUTPUT_DIR, "profiles"), exist_ok=True)

# Pre-shock state with units
rho0 = RHO0 * richio.units.get_unit("Density")
p0 = P0 * richio.units.get_unit("Pressure")
c0 = (GAMMA * p0 / rho0) ** 0.5

# Sedov-Taylor self-similar solution (Landau & Lifshitz, Fluid Mechanics,
# sec. 106).  Everything is parametrised by V in (1/gamma, 2/(gamma+1)].
nu1 = -(13 * GAMMA**2 - 7 * GAMMA + 12) / ((3 * GAMMA - 1) * (2 * GAMMA + 1))
nu2 = 5 * (GAMMA - 1) / (2 * GAMMA + 1)
nu3 = 3 / (2 * GAMMA + 1)
nu4 = -nu1 / (2 - GAMMA)
nu5 = -2 / (2 - GAMMA)

V_par = 1 / GAMMA + np.geomspace(1e-9, 2 / (GAMMA + 1) - 1 / GAMMA, 400)
A = (0.5 * (GAMMA + 1) * V_par) ** (-2)
B = ((GAMMA + 1) / (7 - GAMMA) * (5 - (3 * GAMMA - 1) * V_par)) ** nu1
C = ((GAMMA + 1) / (GAMMA - 1) * (GAMMA * V_par - 1)) ** nu2
xi_par = (A * B * C) ** 0.2  # r / R_shock
G_par = (
    (GAMMA + 1)
    / (GAMMA - 1)
    * ((GAMMA + 1) / (GAMMA - 1) * (GAMMA * V_par - 1)) ** nu3
    * ((GAMMA + 1) / (7 - GAMMA) * (5 - (3 * GAMMA - 1) * V_par)) ** nu4
    * ((GAMMA + 1) / (GAMMA - 1) * (1 - V_par)) ** nu5
)  # rho / rho0
Z_par = (
    GAMMA * (GAMMA - 1) * (1 - V_par) * V_par**2 / (2 * (GAMMA * V_par - 1))
)  # c^2 t^2 / r^2 * 25/4

# Snapshot list, sorted by number
files = sorted(
    glob.glob(os.path.join(INPUT_DIR, SNAP_PATTERN)),
    key=lambda f: int(re.search(r"(\d+)", os.path.basename(f)).group(1)),
)
if ONLY is not None:
    files = [
        f
        for f in files
        if int(re.search(r"(\d+)", os.path.basename(f)).group(1)) in ONLY
    ]
print(f"{len(files)} snapshots in {INPUT_DIR}")

# Accumulated per-snapshot rows (lists of unyt scalars / floats)
cols = {
    "time": [],
    "diss_rich": [],
    "diss_analytic": [],
    "diss_sf": [],
    "mach_T_med": [],
    "mach_P_med": [],
    "mach_rho_med": [],
    "mach_analytic": [],
    "R_analytic": [],
    "R_rich": [],
    "R_sf": [],
    "E_total": [],
    "n_zone": [],
    "n_surface": [],
    "area_sf": [],
    "area_analytic": [],
}

for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = walltime.perf_counter()
    snap = richio.load(path)
    t = snap.time

    if t <= 0:
        print(f"{name}: t = {t}, no shock yet -- skipped")
        continue

    r_sim = (snap.X**2 + snap.Y**2 + snap.Z**2) ** 0.5
    vr_sim = (snap.vx * snap.X + snap.vy * snap.Y + snap.vz * snap.Z) / r_sim
    v2 = snap.vx**2 + snap.vy**2 + snap.vz**2
    Ediss_cell = snap.dissipation * snap.volume

    # Total energy and the analytic Sedov quantities
    E = np.sum(
        snap.sie * snap.density * snap.volume + snap.density * snap.volume * v2 / 2
    )
    diss_rich = np.sum(Ediss_cell)
    diss_an = DISS_FRACTION * E / t
    R_an = (E / KAPPA / rho0) ** (1 / 5) * t ** (2 / 5)
    vsh = 2 / 5 * R_an / t
    mach_an = float(vsh / c0)

    # R_rich: radius where the binned rich dissipation peaks
    r_edges = np.linspace(0, float(r_sim.max().v), NBINS_R + 1)
    Ediss_binned, _ = np.histogram(r_sim.v, bins=r_edges, weights=Ediss_cell.v)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    R_rich = r_centers[np.argmax(Ediss_binned)] * R_an.units

    # Shock finder
    vor = sf.build_voronoi(snap)
    shock_zone = sf.find_shock_zone(snap, vor)
    result = sf.find_shock_surface(snap, vor, shock_zone)
    n_zone = int(shock_zone.sum())
    n_surface = int(result.surface_mask.sum())

    # Effective face area per surface cell: sphere-equivalent cross section
    A_cell = (3 / 4 * snap.volume[result.surface_mask]) ** (2 / 3) * np.pi ** (1 / 3)
    M_T = result.mach_T
    good = np.isfinite(M_T) & (M_T > 1)
    diss_sf = np.sum(
        1
        / 2
        * rho0
        * M_T[good] ** 3
        * c0**3
        * A_cell[good]
        * sf.delta(M_T[good], gamma=GAMMA)
    )
    area_sf = np.sum(A_cell)
    area_an = 4 * np.pi * R_an**2
    R_sf = (
        np.median(r_sim[result.surface_mask]) if n_surface > 0 else np.nan * R_an.units
    )

    med = lambda M: (
        float(np.median(M[np.isfinite(M)])) if np.isfinite(M).any() else np.nan
    )
    mach_T_med, mach_P_med, mach_rho_med = (
        med(result.mach_T),
        med(result.mach_P),
        med(result.mach_rho),
    )

    cols["time"].append(t)
    cols["diss_rich"].append(diss_rich)
    cols["diss_analytic"].append(diss_an.in_units(diss_rich.units))
    cols["diss_sf"].append(diss_sf.in_units(diss_rich.units))
    cols["mach_T_med"].append(mach_T_med)
    cols["mach_P_med"].append(mach_P_med)
    cols["mach_rho_med"].append(mach_rho_med)
    cols["mach_analytic"].append(mach_an)
    cols["R_analytic"].append(R_an)
    cols["R_rich"].append(R_rich)
    cols["R_sf"].append(R_sf.in_units(R_an.units))
    cols["E_total"].append(E)
    cols["n_zone"].append(n_zone)
    cols["n_surface"].append(n_surface)
    cols["area_sf"].append(area_sf.in_units(area_an.units))
    cols["area_analytic"].append(area_an)

    # ── Profile-fit figure (visual check that the Sedov solution applies) ────
    r_an_prof = xi_par * R_an
    rho_an_prof = rho0 * G_par
    p_an_prof = rho_an_prof * (4 * r_an_prof**2 * Z_par / (25 * t**2)) / GAMMA
    v_an_prof = 2 * r_an_prof * V_par / (5 * t)

    panels = [
        (snap.density, rho_an_prof, r"$\rho$"),
        (snap.pressure, p_an_prof, r"$p$"),
        (vr_sim, v_an_prof, r"$v_r$"),
        (Ediss_cell, None, r"$\dot E_{\rm diss}$ per cell"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (sim, an, label) in zip(axes, panels):
        ax.scatter(
            r_sim[::SUBSAMPLE],
            sim[::SUBSAMPLE],
            s=3,
            marker="o",
            facecolors="none",
            edgecolors="C0",
            alpha=0.3,
            label="simulation",
        )
        if an is not None:
            ax.plot(r_an_prof, an, color="k", lw=2, label="Sedov (L&L 106)")
        ax.axvline(
            float(R_an.v), ls="--", color="gray", lw=1, label=r"$R_{\rm analytic}$"
        )
        ax.axvline(
            float(R_rich.v), ls=":", color="tomato", lw=1, label=r"$R_{\rm rich}$"
        )
        ax.axvline(
            float(R_sf.v), ls="-.", color="darkorange", lw=1, label=r"$R_{\rm sf}$"
        )
        ax.set_xlabel(f"r  [{r_sim.units}]")
        ax.set_ylabel(label)
        ax.set_xlim(0, float(snap.box[3].v))
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name}   t = {t:.4f}   M_an = {mach_an:.1f}")
    fig.tight_layout()
    figpath = os.path.join(OUTPUT_DIR, "profiles", f"{name}.png")
    fig.savefig(figpath, dpi=130)
    plt.close(fig)

    print(
        f"{name}: t={float(t.v):.4f}  diss rich/an/sf = "
        f"{float(diss_rich.v):.3e}/{float(diss_an.v):.3e}/{float(diss_sf.v):.3e}  "
        f"R an/rich/sf = {float(R_an.v):.3f}/{float(R_rich.v):.3f}/{float(R_sf.v):.3f}  "
        f"M_T={mach_T_med:.1f}  ({walltime.perf_counter() - t0:.0f} s)"
    )

# ── Save table ────────────────────────────────────────────────────────────────
# One unyt column per key; unyt.savetxt writes the units as the last header line.
order = np.argsort([float(x.v) for x in cols["time"]])
arrays = []
for key, vals in cols.items():
    if hasattr(vals[0], "units"):
        u = vals[0].units
        arr = unyt.unyt_array([float(v.in_units(u).v) for v in vals], u)[order]
    else:
        arr = unyt.unyt_array(np.asarray(vals, dtype=float), "")[order]
    arrays.append(arr)

header = (
    "Sedov dissipation scan  (see scripts/sedov_dissipation_scan.py)\n"
    f"input: {INPUT_DIR}\n"
    f"gamma = {GAMMA:.6f}, kappa = {KAPPA}, diss_fraction = {DISS_FRACTION}\n"
    f"pre-shock state: rho0 = {rho0}, P0 = {p0}, c0 = {c0}\n"
    f"total energy E_total is a per-snapshot column (should be conserved)\n"
    "analytic: R = (E/(kappa rho0))^(1/5) t^(2/5), dE_diss/dt = diss_fraction*E/t\n"
    "sf: sum over surface cells of 1/2 rho0 M_T^3 c0^3 A_cell delta(M_T),\n"
    "    A_cell = (3/4 V_cell)^(2/3) pi^(1/3);  R_rich = peak of binned rich dissipation\n"
    "columns: " + "  ".join(cols.keys())
)
table_path = os.path.join(OUTPUT_DIR, TABLE_NAME)
unyt.savetxt(table_path, arrays, header=header, delimiter="\t")
print(f"\nTable  -> {table_path}")
print(f"Plots  -> {os.path.join(OUTPUT_DIR, 'profiles')}/")
