#!/usr/bin/env python3
"""Overlay returning Paczynski-Wiita debris orbits on nozzle density slices."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rich-tde-pw-orbit-test")

REPO = Path("/home/hey4/rich_tde")
sys.path.insert(0, str(REPO / "dev"))

import dev  # noqa: E402, F401 -- applies the repository plotting style
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dev.datapaths import TDE_PARAMETERS  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from richio.plots import scalar_map  # noqa: E402


SELECTED = {
    "1e4": (77, 0.999),
    "1e5": (161, 0.606),
    "1e6": (877, 1.099),
}
RESOLUTION = 768
XY_WINDOW_RP = (-1.0, 2.0, -1.5, 1.5)
OUTPUT = REPO / "data/processed/PresentationNozzleSlices/pw-orbit-test.png"


def pw_acceleration(_, state, mass, schwarzschild_radius):
    x, y, vx, vy = state
    radius = np.hypot(x, y)
    factor = -mass / (radius * (radius - schwarzschild_radius) ** 2)
    return vx, vy, factor * x, factor * y


def returning_orbit(run, time_tfb):
    mass, stellar_mass, stellar_radius = TDE_PARAMETERS[run]
    pericenter = stellar_radius * (mass / stellar_mass) ** (1 / 3)
    schwarzschild_radius = 4.21 * mass / 1e6
    energy_spread = stellar_mass * (mass / stellar_mass) ** (1 / 3) / stellar_radius
    binding_energy = -energy_spread * time_tfb ** (-2 / 3)
    pericenter_speed = np.sqrt(
        2 * (mass / (pericenter - schwarzschild_radius) + binding_energy)
    )
    newtonian_period = 2 * np.pi * mass / (-2 * binding_energy) ** 1.5

    solution = solve_ivp(
        pw_acceleration,
        (0.0, 1.6 * newtonian_period),
        (pericenter, 0.0, 0.0, -pericenter_speed),
        args=(mass, schwarzschild_radius),
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=newtonian_period / 5000,
    )
    sample_time = np.linspace(0.0, solution.t[-1], 50000)
    x, y = solution.sol(sample_time)[:2]
    radius = np.hypot(x, y)
    minima = np.where((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:]))[0] + 1
    next_pericenter = minima[minima > 0.5 * len(radius)][0]

    near = radius < 2.3 * pericenter
    before = np.where(~near[:next_pericenter])[0][-1] + 1
    after = next_pericenter + np.where(~near[next_pericenter:])[0][0]
    x = x[before:after] / pericenter
    y = y[before:after] / pericenter
    phi = np.unwrap(np.arctan2(y, x))
    phi -= phi[np.argmin(np.hypot(x, y))]
    return x, y, phi


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
for ax, (run, (snapshot, time_tfb)) in zip(axes, SELECTED.items(), strict=True):
    cache = (
        REPO
        / f"data/processed/NozzleZoomSlices/{run}/grids"
        / f"nozzle_zoom_snap_{snapshot:04d}_{RESOLUTION}.npz"
    )
    with np.load(cache) as data:
        x_grid = np.asarray(data["x_rp"])
        y_grid = np.asarray(data["y_rp"])
        density = np.asarray(data["density"])

    _, image = scalar_map(
        density,
        x_grid,
        y_grid,
        ax=ax,
        cmap="twilight",
        colorbar_label=r"$\log_{10}(\rho/[\mathrm{g\,cm^{-3}}])$",
        log_scale=False,
        aspect_equal=False,
        shading="auto",
        rasterized=True,
    )
    image.colorbar.remove()
    orbit_x, orbit_y, orbit_phi = returning_orbit(run, time_tfb)
    ax.plot(orbit_x, orbit_y, color="red", linewidth=1.6)
    ax.scatter(
        orbit_x[np.argmin(abs(orbit_phi))],
        orbit_y[np.argmin(abs(orbit_phi))],
        s=18,
        color="red",
        zorder=3,
    )
    ax.set_xlim(XY_WINDOW_RP[:2])
    ax.set_ylim(XY_WINDOW_RP[2:])
    ax.set_box_aspect(1)
    ax.set_title(rf"$10^{{{run[-1]}}}\,M_\odot$")
    ax.set_xlabel(r"$x/r_p$")

axes[0].set_ylabel(r"$y/r_p$")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=240)
print(OUTPUT)
