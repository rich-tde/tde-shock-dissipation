#!/usr/bin/env python3
r"""Plot nozzle-split dissipation rates and cumulative circularization fractions.

Sort the input powers by time, keep ``t/t_fb>=0.1`` (1e4/1e5) or ``>=0.7``
(1e6), and plot ``Ediss_power*t_fb/Delta_Ec`` with
``Delta_Ec=(G*Mbh/(4*r_p))*(Mstar/2)``. Trapezoidal integration gives
cumulative ``E_diss/Delta_Ec`` starting at the first retained sample, not at
disruption. Ten-point sampling is a diagnostic, not a converged time integral.

Input files
-----------
``data/processed/EdissFourRegions/Ediss-t-four-regions-<run>-n10.txt``
    Generated with ``Ediss-t.py --regions nozzle-split --npoints 10`` and
    modes 1/2/3 for ``1e4``/``1e5``/``1e6``. Override ``--data-dir`` and
    ``--input-pattern`` (a filename template containing ``{run}``).
    ``np.loadtxt(path, ndmin=2)`` gives ``float64``, shape ``(N, 7)``:

    * Column 0: ``SNAPNUM``, integer-valued snapshot ID.
    * Column 1: ``TIME``, code time.
    * Column 2: ``TFALLBACK``, dimensionless ``t/t_fb``.
    * Column 3: ``EDISS_NOZZLE``, ``x>0`` and spherical ``r<3*r_p``.
    * Column 4: ``EDISS_STREAM_DISK``, ``x>0`` and ``r>=3*r_p``.
    * Column 5: ``EDISS_OUTGOING``, ``-r_a<x<0,y<0``.
    * Column 6: ``EDISS_INCOMING``, ``-r_a<x<0,y>0``.

    Columns 3--6 are powers in ``code_mass*code_length**2/code_time**3``;
    the outer region is excluded. Standard Ediss tables have different
    region meanings and fail the header check. Lines beginning ``#`` are
    comments (column names, units, and selection).

Output files
------------
``OUT/Ediss-rate-over-Delta-vs-t-four-regions.png`` and ``.pdf``
    Four regional power curves normalized by ``Delta_Ec/t_fb``. One panel
    per selected run, in ``--run`` order; y axes are logarithmic.
``OUT/Ediss-cumulative-over-Delta-vs-t-four-regions.png`` and ``.pdf``
    Integrated dimensionless energy fractions in the same region/run order.
    Nonpositive cumulative values are hidden on the logarithmic axes.

``OUT`` defaults to ``<data-dir>/figures``; override ``--output-dir``. There
are four figure files in total, not saved numerical arrays of the curves.
PNGs are raster images, saved at 240 dpi. Loading via
``np.asarray(Image.open(path).convert("RGBA"))`` gives ``uint8``, shape
``(H, W, 4)``: row, column, red/green/blue/alpha channels (0--255).
H/W depend on the selected runs, layout and style cropping; pixels include
axes/text/color and are not physical measurements. PDFs are single-page
vector-capable figures intended for a PDF viewer, not ``np.load``.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/plot-Ediss-t-four-regions.py
    python works/shock-tde/plot-Ediss-t-four-regions.py --run 1e4 \
        --input-pattern 'Ediss-t-four-regions-{run}.txt' \
        --output-dir data/processed/EdissFourRegions/full-figures

Repeat ``--run`` to choose panels; the default is all three. Existing PNGs
and PDFs are individually retained unless ``--overwrite`` is supplied.
Choose separate output directories for changed inputs or selections.

Loading examples
----------------
Open the generated rate figure and inspect its pixel array::

    from pathlib import Path
    from PIL import Image
    import numpy as np

    root = Path("data/processed/EdissFourRegions")
    path = root / "figures/Ediss-rate-over-Delta-vs-t-four-regions.png"
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"))
    print(pixels.shape, pixels.dtype)  # (H, W, 4), uint8

For numerical analysis, read the input table rather than the rendered PNG::

    import unyt as u
    import richio

    table = np.loadtxt(root / "Ediss-t-four-regions-1e4-n10.txt", ndmin=2)
    order = np.argsort(table[:, 1], kind="stable")
    table = table[order]
    table = table[table[:, 2] >= 0.1]
    time_tfb = table[:, 2]
    power = u.unyt_array(
        table[:, 3:7], "code_mass*code_length**2/code_time**3",
        registry=richio.units.registry,
    ).to("erg/s")
    # power is (Nselected, 4): nozzle, stream-disk, outgoing, incoming.
    nozzle_power_erg_s = power[:, 0].to_value("erg/s")
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import dev  # noqa: F401  # isort: skip  # Apply the study style before pyplot.
import matplotlib.pyplot as plt
import numpy as np
import typer
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
    stellar_radius_value, stellar_mass_value, black_hole_mass_value, minimum_tfb = (
        CONFIGS[label]
    )
    stellar_radius = stellar_radius_value * richio.units.lscale
    stellar_mass = stellar_mass_value * richio.units.mscale
    black_hole_mass = black_hole_mass_value * richio.units.mscale
    pericenter_radius = stellar_radius * (black_hole_mass / stellar_mass) ** (1 / 3)
    fallback_time = (
        np.pi
        / np.sqrt(2)
        * (stellar_radius**3 / u.G / stellar_mass) ** 0.5
        * (black_hole_mass / stellar_mass) ** 0.5
    )
    circularization_energy = (
        u.G * black_hole_mass / (4 * pericenter_radius) * stellar_mass / 2
    )
    return fallback_time, circularization_energy, minimum_tfb


def load_mode(label: str, data_dir: Path, input_pattern: str):
    path = data_dir / input_pattern.format(run=label)
    with path.open() as handle:
        header = handle.readline().lstrip("# ").split()
    if (
        header
        != "SNAPNUM TIME TFALLBACK EDISS_NOZZLE EDISS_STREAM_DISK EDISS_OUTGOING EDISS_INCOMING".split()
    ):
        raise ValueError(f"{path} is not a nozzle-split dissipation table")
    raw = np.loadtxt(path, delimiter="\t", ndmin=2).T
    raw = raw[:, np.argsort(raw[1], kind="stable")]
    fallback_time, circularization_energy, minimum_tfb = physical_scales(label)
    raw = raw[:, raw[2] >= minimum_tfb]
    times = u.unyt_array(raw[1], TIME_UNIT, registry=richio.units.registry)
    power = u.unyt_array(raw[3:7], POWER_UNIT, registry=richio.units.registry)
    return raw[2], times, power, fallback_time, circularization_energy


def decorate(axes, ylabel: str, runs):
    for axis, label in zip(axes, runs):
        axis.set_xlabel(r"$t/t_{\rm fb}$", fontsize=14)
        axis.set_title(rf"$M_\bullet=10^{{{label[-1]}}}\,M_\odot$", fontsize=15)
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(ylabel, fontsize=14)
    handles = [
        plt.Line2D([], [], color=color, linestyle=style, label=name)
        for name, color, style in REGIONS
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="best")


def plot_rate(runs, data_dir, input_pattern, output_dir, overwrite):
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.1 * len(runs), 3.8), sharey=True, squeeze=False
    )
    axes = axes[0]
    for axis, label in zip(axes, runs):
        tfb, _, power, fallback_time, circularization_energy = load_mode(
            label, data_dir, input_pattern
        )
        normalized = power / circularization_energy * fallback_time
        for row, (_, color, style) in zip(normalized, REGIONS):
            axis.plot(tfb, row, color=color, linestyle=style, linewidth=1.35)
        axis.set_yscale("log")
    decorate(axes, r"$\dot E_{\rm diss}/\Delta E_c\;[1/t_{\rm fb}]$", runs)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_dir, "Ediss-rate-over-Delta-vs-t-four-regions", overwrite)
    plt.close(fig)


def plot_cumulative(runs, data_dir, input_pattern, output_dir, overwrite):
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.1 * len(runs), 3.8), sharey=True, squeeze=False
    )
    axes = axes[0]
    for axis, label in zip(axes, runs):
        tfb, times, power, _, circularization_energy = load_mode(
            label, data_dir, input_pattern
        )
        dt = times[1:] - times[:-1]
        increments = 0.5 * (power[:, 1:] + power[:, :-1]) * dt
        cumulative = np.concatenate(
            [
                np.zeros((4, 1)),
                np.cumsum(
                    (increments / circularization_energy).to_value("dimensionless"),
                    axis=1,
                ),
            ],
            axis=1,
        )
        for row, (_, color, style) in zip(cumulative, REGIONS):
            axis.plot(
                tfb,
                np.where(row > 0, row, np.nan),
                color=color,
                linestyle=style,
                linewidth=1.35,
            )
        axis.set_yscale("log")
    decorate(axes, r"$E_{\rm diss}(<t)/\Delta E_c$", runs)
    fig.tight_layout(w_pad=2.2)
    save_figure(
        fig, output_dir, "Ediss-cumulative-over-Delta-vs-t-four-regions", overwrite
    )
    plt.close(fig)


def save_figure(fig, output_dir, stem, overwrite):
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        if overwrite or not path.exists():
            fig.savefig(path, dpi=240)
            print(f"Saved {path}")
        else:
            print(f"Retained {path}; use --overwrite to replace")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_dir: Path = typer.Option(
        DATA_DIR, help="Directory containing nozzle-split time-series tables."
    ),
    output_dir: Path | None = typer.Option(
        None, help="Figure directory; default DATA_DIR/figures."
    ),
    input_pattern: str = typer.Option(
        "Ediss-t-four-regions-{run}-n10.txt",
        help="Input filename pattern; {run} expands to 1e4, 1e5, or 1e6.",
    ),
    run: list[str] | None = typer.Option(
        None, help="Repeat for selected runs (1e4, 1e5, 1e6); default all."
    ),
    overwrite: bool = typer.Option(False, help="Replace existing PNG/PDF figures."),
):
    """Plot power and cumulative energy from existing nozzle-split tables."""
    runs = run or list(CONFIGS)
    if any(label not in CONFIGS for label in runs):
        raise typer.BadParameter("--run must be 1e4, 1e5 or 1e6")
    output_dir = output_dir or data_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rate(runs, data_dir, input_pattern, output_dir, overwrite)
    plot_cumulative(runs, data_dir, input_pattern, output_dir, overwrite)


if __name__ == "__main__":
    app()
