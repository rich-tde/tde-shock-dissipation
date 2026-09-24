#!/usr/bin/env python3
"""Overlay ballistic returning orbits on existing nozzle XY density caches.

For each panel, integrate a Paczynski-Wiita orbit with binding energy from
the cached ``t/t_fb``, select the next pericentre passage, and overlay the
path and pericentre marker on log10 CGS density. The ballistic path is a
visual guide, not a fit to simulated gas or evidence of a shock. This tool
reads existing grids; it does not load raw snapshots or save orbit arrays.

Input files
-----------
``data/processed/NozzleZoomSlices/RUN/grids/nozzle_zoom_snap_NNNN_RES.npz``
produced by ``nozzle-zoom-slices.py``. The absolute default root is relative
to ``/home/hey4/rich_tde``; ``--cache-root`` overrides it. Default cases are
``1e4:77``, ``1e5:161``, ``1e6:877``, at ``RES=768``. Select alternatives
with repeated ``--case RUN:NNNN`` and ``--resolution``. Required NPZ keys:

``x_rp``, ``y_rp`` : float64, shape ``(RES,)``
    BH-frame sample coordinates divided by pericentre radius; dimensionless.
``density`` : float64, shape ``(RES, RES)``
    Log10 density [g/cm^3]. Entry ``[i, j]`` belongs to ``x_rp[i], y_rp[j]``.
    Nonpositive/nonfinite original densities are NaN.
``time_tfb`` : float64, shape ``()``
    Dimensionless time/fallback time; read with ``.item()``.

Input arrays are loaded with ``np.load``. Create missing caches with
``nozzle-zoom-slices.py --mode MODE --snapshot NNNN`` before running this tool.
Stellar/BH parameters come from ``dev.datapaths.TDE_PARAMETERS``.

Output files
------------
``data/processed/PresentationNozzleSlices/pw-orbit-test.png``
    Default raster figure; override with ``--output`` (PNG or PDF). One
    density/orbit panel per requested case, in argument order, with axes
    ``x/r_p`` and ``y/r_p``. Pixel dimensions depend on the number of cases
    and ``--dpi``. PNG pixels loaded through Pillow are uint8, shape
    ``(height, width, channels)`` (normally four RGBA channels), not a
    scientific density grid. PDF is a document opened with a PDF viewer.
    No numerical output file is written: use the input NPZ density arrays
    for quantitative work, rather than inferring values from image pixels.

Usage
-----
Run from ``/home/hey4/rich_tde`` with the richanalysis Python environment::

    python works/shock-tde/pw-orbit-nozzle-slice-test.py --list-only
    python works/shock-tde/pw-orbit-nozzle-slice-test.py
    python works/shock-tde/pw-orbit-nozzle-slice-test.py --case 1e4:77 --output data/processed/PresentationNozzleSlices/orbit-1e4-77.png

Existing output is skipped unless ``--overwrite`` is supplied, even when
case selection or plotting settings change. Use another ``--output`` path
to retain variants. Nothing runs on import.

Loading examples
----------------
Inspect the default PNG after running the default command::

    from PIL import Image
    import numpy as np

    path = "data/processed/PresentationNozzleSlices/pw-orbit-test.png"
    with Image.open(path) as image:
        print(image.format, image.size, image.mode)  # size is (width, height).
        pixels = np.asarray(image).copy()
    print(pixels.shape, pixels.dtype)  # (height, width, channels), uint8.

Load the underlying numerical density grid independently of the figure::

    from pathlib import Path
    import numpy as np

    root = Path("data/processed/NozzleZoomSlices/1e4/grids")
    with np.load(root / "nozzle_zoom_snap_0077_768.npz") as data:
        density_cgs = 10.0 ** data["density"]
        time_tfb = data["time_tfb"].item()
    print(time_tfb, np.nanmax(density_cgs))  # Dimensionless time, g/cm^3.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

REPO = Path("/home/hey4/rich_tde")
sys.path.insert(0, str(REPO / "dev"))

import dev  # noqa: E402, F401 -- applies the repository plotting style

# Apply the repository plotting style before importing pyplot.
# isort: split

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402
from dev.datapaths import TDE_PARAMETERS  # noqa: E402
from richio.plots import scalar_map  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

DEFAULT_CASES = ("1e4:77", "1e5:161", "1e6:877")
CACHE_ROOT = REPO / "data/processed/NozzleZoomSlices"
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


def main(
    case: list[str] | None = typer.Option(
        None,
        help="RUN:SNAPNUM; repeat for panels. Defaults: 1e4:77, 1e5:161, 1e6:877.",
    ),
    cache_root: Path = typer.Option(
        CACHE_ROOT,
        help="Input root containing RUN/grids/nozzle_zoom_snap_NNNN_RES.npz.",
    ),
    resolution: int = typer.Option(
        RESOLUTION, min=16, help="Resolution suffix of existing input caches."
    ),
    output: Path = typer.Option(OUTPUT, help="Output figure path (.png or .pdf)."),
    dpi: int = typer.Option(240, min=50, help="Output raster resolution."),
    overwrite: bool = typer.Option(
        False, help="Replace the output figure if it exists."
    ),
    list_only: bool = typer.Option(False, help="List required cache files and exit."),
) -> None:
    """Plot returning ballistic PW orbits over existing XY nozzle density caches."""
    selected = []
    for item in case or DEFAULT_CASES:
        run, snapshot = item.split(":")
        if run not in TDE_PARAMETERS:
            raise typer.BadParameter("Run must be 1e4, 1e5, or 1e6.")
        path = (
            cache_root
            / run
            / "grids"
            / f"nozzle_zoom_snap_{int(snapshot):04d}_{resolution}.npz"
        )
        selected.append((run, path))
        print(f"Input: {path}")
    print(f"Output: {output}")
    if list_only:
        return
    if output.exists() and not overwrite:
        print("Output exists; use --overwrite to replace it.")
        return

    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(3.8 * len(selected), 3.9),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (run, cache) in zip(axes.flat, selected, strict=True):
        with np.load(cache) as data:
            x_grid = np.asarray(data["x_rp"])
            y_grid = np.asarray(data["y_rp"])
            density = np.asarray(data["density"])
            time_tfb = float(data["time_tfb"])
        print(f"{run}: t/t_fb={time_tfb:.6g}")
        _, image = scalar_map(
            density,
            x_grid,
            y_grid,
            ax=axis,
            cmap="twilight",
            colorbar_label=r"$\log_{10}(\rho/[\mathrm{g\,cm^{-3}}])$",
            log_scale=False,
            aspect_equal=False,
            shading="auto",
            rasterized=True,
        )
        image.colorbar.remove()
        orbit_x, orbit_y, orbit_phi = returning_orbit(run, time_tfb)
        axis.plot(orbit_x, orbit_y, color="red", linewidth=1.6)
        axis.scatter(
            orbit_x[np.argmin(abs(orbit_phi))],
            orbit_y[np.argmin(abs(orbit_phi))],
            s=18,
            color="red",
            zorder=3,
        )
        axis.set_xlim(XY_WINDOW_RP[:2])
        axis.set_ylim(XY_WINDOW_RP[2:])
        axis.set_box_aspect(1)
        axis.set_title(rf"$10^{{{run[-1]}}}\,M_\odot$")
        axis.set_xlabel(r"$x/r_p$")
    axes[0, 0].set_ylabel(r"$y/r_p$")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
