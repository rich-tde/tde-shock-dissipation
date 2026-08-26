#!/usr/bin/env python3
"""Plot the standalone four-region dissipation time series."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ediss-four-regions")

import dev
import matplotlib.pyplot as plt
import numpy as np
import unyt as u

import richio


DATA_DIR = Path("/home/hey4/rich_tde/data/processed/EdissFourRegions")
FIGURE_DIR = DATA_DIR / "figures"
POWER_UNIT = "code_length**2*code_mass/code_time**3"
TIME_UNIT = "code_time"
REGIONS = (
    ("Nozzle: $x>0$, $r<3r_p$", "C3", "-"),
    ("Stream--disk: $x>0$, $r\\geq3r_p$", "C0", "-"),
    ("Outgoing: $-r_a<x<0$, $y<0$", "C2", "--"),
    ("Incoming: $-r_a<x<0$, $y>0$", "C1", ":"),
)
CONFIGS = {
    "1e4": (0.47, 0.5, 1e4, 0.1),
    "1e5": (0.47, 0.5, 1e5, 0.1),
    "1e6": (1.0, 1.0, 1e6, 0.7),
}


def physical_scales(label: str):
    rstar_value, mstar_value, mbh_value, minimum_tfb = CONFIGS[label]
    rstar = rstar_value * richio.units.lscale
    mstar = mstar_value * richio.units.mscale
    mbh = mbh_value * richio.units.mscale
    r_p = rstar * (mbh / mstar) ** (1 / 3)
    tmin = np.pi / np.sqrt(2) * (rstar**3 / u.G / mstar) ** 0.5 * (mbh / mstar) ** 0.5
    delta = u.G * mbh / (4 * r_p) * mstar / 2
    return tmin, delta, minimum_tfb


def load_mode(label: str):
    path = DATA_DIR / f"Ediss-t-four-regions-{label}-n10.txt"
    raw = np.loadtxt(path, delimiter="\t").T
    raw = raw[:, np.argsort(raw[1], kind="stable")]
    tmin, delta, minimum_tfb = physical_scales(label)
    raw = raw[:, raw[2] >= minimum_tfb]
    times = u.unyt_array(raw[1], TIME_UNIT, registry=richio.units.registry)
    power = u.unyt_array(raw[3:7], POWER_UNIT, registry=richio.units.registry)
    return raw[2], times, power, tmin, delta


def decorate(axes, ylabel: str):
    for axis, label in zip(axes, CONFIGS):
        axis.set_xlabel(r"$t/t_{\rm fb}$", fontsize=14)
        axis.set_title(
            rf"$M_\bullet=10^{{{label[-1]}}}\,M_\odot$", fontsize=15
        )
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(ylabel, fontsize=14)
    handles = [
        plt.Line2D([], [], color=color, linestyle=style, label=name)
        for name, color, style in REGIONS
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="best")


def plot_rate():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    for axis, label in zip(axes, CONFIGS):
        tfb, _, power, tmin, delta = load_mode(label)
        normalized = power / delta * tmin
        for row, (_, color, style) in zip(normalized, REGIONS):
            axis.plot(tfb, row, color=color, linestyle=style, linewidth=1.35)
        axis.set_yscale("log")
    decorate(axes, r"$\dot E_{\rm diss}/\Delta E_c\;[1/t_{\rm fb}]$")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIGURE_DIR / "Ediss-rate-over-Delta-vs-t-four-regions.png", dpi=240)
    fig.savefig(FIGURE_DIR / "Ediss-rate-over-Delta-vs-t-four-regions.pdf")
    plt.close(fig)


def plot_cumulative():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    for axis, label in zip(axes, CONFIGS):
        tfb, times, power, _, delta = load_mode(label)
        dt = times[1:] - times[:-1]
        increments = 0.5 * (power[:, 1:] + power[:, :-1]) * dt
        cumulative = np.concatenate(
            [np.zeros((4, 1)), np.cumsum((increments / delta).to_value("dimensionless"), axis=1)],
            axis=1,
        )
        for row, (_, color, style) in zip(cumulative, REGIONS):
            axis.plot(tfb, np.where(row > 0, row, np.nan), color=color, linestyle=style, linewidth=1.35)
        axis.set_yscale("log")
    decorate(axes, r"$E_{\rm diss}(<t)/\Delta E_c$")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIGURE_DIR / "Ediss-cumulative-over-Delta-vs-t-four-regions.png", dpi=240)
    fig.savefig(FIGURE_DIR / "Ediss-cumulative-over-Delta-vs-t-four-regions.pdf")
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_rate()
    plot_cumulative()
    print(f"Saved four figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
