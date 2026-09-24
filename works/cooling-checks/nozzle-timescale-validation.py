#!/usr/bin/env python3
"""Compare nozzle cooling maps at selected sampling-grid resolutions.

Integrate a nearest-cell Cartesian grid along z inside spherical radius
``r < 3 r_p``, after correcting moving-frame coordinates to the BH frame.
Let ``Sigma = integral(rho dz)``, ``H = integral(rho |z| dz)/Sigma`` and
``vzbar = integral(rho |vz| dz)/Sigma``. The times are emission
``tc = integral(rho sie dz)/integral(alpha_P a_rad T**4 c dz)``, vertical flow
``tv = H/vzbar``, diffusion ``tdiff = H tau_R/c``, and photon escape
``tesc = H (1 + tau_R)/c``, with ``tau_R = integral(alpha_R dz)``.
Here emission is gross thermal emission, not net radiative energy exchange.
The ratio ``max(tc, tesc)/tv`` is a diagnostic, not evidence of a causal cooling
change. The accepted wedge has projected radius ``0.6 <= R/r_p <= 1.75`` and
angular half-width 4.5 degrees about the supplied/native-peak direction.
This diagnoses gridding sensitivity; native-cell resolution remains a separate
question.

Input files
-----------
``--mode 1/2/3`` selects ``1e4/1e5/1e6`` and default snapshots 108/142/961.
``--snapshot-number`` overrides the snapshot. ``dev.datapaths.DATAPATHS`` resolves
RICH ``snap_full_<n>.h5`` or ``snap_<n>.h5`` files. Required ``richio`` fields are
time, X/Y/Z, rho, T, vz, sie and dissipation. The second input is
``<direction-root>/<run>/directions/direction_snap_<NNNN>.npz`` produced by
``nozzle-wedge-validation.py``. Default ``--direction-root`` is
``data/processed/CoolingChecks/nozzle-timescale-series/stage1-wedge-selection``.
Read scalar ``direction_peak_{x,y,z}_rp_dirmin_0p6`` for ``1e4/1e5``, or
``..._0p8`` for ``1e6``, as the native peak coordinates in pericentre units.

Output files
------------
Default ``--output-root`` is
``data/processed/CoolingChecks/nozzle-timescale-series/stage2-timescales``.
Under ``<output-root>/<run>/`` the files are:

``maps/timescales_snap_<NNNN>_<grid>.npz``
    One physical map archive per grid, with the full schema below. ``<grid>``
    is the cubic resolution (e.g. ``256``), or
    ``<Nx>x<Ny>x<Nz>_<linear|sinh>z`` (e.g. ``256x256x512_sinhz``).
``figures/timescales_snap_<NNNN>_<grid>.png``
    Six-panel raster: log10 dissipation column, Sigma, tau_R, tc/tv, tesc/tv and
    max(tc, tesc)/tv. A cyan contour marks the wedge. No data are encoded in the
    image beyond rendered pixels; use the NPZ for numerical analysis.
``summary_snap_<NNNN>.txt``
    Whitespace-delimited table with one header line and shape ``(3*G, 5)``,
    where G is the number of requested grids. Zero-based columns: 0 integer
    x-resolution, 1 string statistic (``median``, ``dissipation_weighted_mean``
    or ``max_dissipation_pixel``), 2 float tc/tv, 3 float tesc/tv, 4 float
    max(tc, tesc)/tv. All three ratios are dimensionless; ``nan``/``inf`` can
    represent undefined results. Column 0 alone does not identify an anisotropic
    grid: retain its NPZ filename. This CLI does not write the legacy
    ``summary.csv`` or ``convergence.csv`` files that may exist in old runs.

Each map is a compressed NumPy ``.npz`` archive loaded with ``np.load``;
there is no single rectangular table. ``Nx``, ``Ny``, ``Nz`` are grid-point
counts. Maps have axis order ``(x, y)``, with z integrated out; plot ``map.T``
against the one-dimensional x/y coordinates. Values are linear, not logarithms.
Empty columns can yield NaN or infinity in divisions; mask non-finite values
for statistics/plots. Zero integrated density/dissipation outside the aperture
is retained. The keys written by the current calculation are:

``run``, ``snapshot_path`` : Unicode arrays, shape ``()``
    Mass label (``1e4``, ``1e5`` or ``1e6``) and source HDF5 path. Use ``.item()``
    to retrieve a Python string.
``snapnum`` : int64 array, shape ``()``
    Source snapshot number.
``resolution_x``, ``resolution_y``, ``resolution_z`` : int64 arrays, shape ``()``
    ``Nx``, ``Ny``, ``Nz``; these describe the sampling grid, not native cells.
``z_spacing`` : Unicode array, shape ``()``
    ``linear`` or ``sinh``. The z integration uses ``Nz - 1`` left samples and
    the differences between consecutive z coordinates.
``sinh_scale_rp`` : float64 array, shape ``()``
    Sinh scale divided by pericentre radius; NaN for linear spacing.
``time_tfb``, ``time_days`` : float64 arrays, shape ``()``
    Snapshot time divided by fallback time, and snapshot time in days.
``x_rp``, ``y_rp`` : float64 arrays, shapes ``(Nx,)``, ``(Ny,)``
    BH-frame x/y sampling coordinates divided by pericentre radius.
``wedge_mask`` : bool array, shape ``(Nx, Ny)``
    True for wedge pixels with positive integrated dissipation. The other maps
    cover the full aperture; apply this mask when selecting nozzle statistics.
``dissipation_column_erg_s_cm2`` : float64 array, shape ``(Nx, Ny)``
    ``integral(dissipation dz)`` in erg/s/cm**2; this is power per projected
    area, not pixel power. Multiply by pixel area in cm**2 before summing power.
``sigma_g_cm2`` : float64 array, shape ``(Nx, Ny)``
    Surface density ``Sigma`` in g/cm**2.
``H_Rstar`` : float64 array, shape ``(Nx, Ny)``
    Density-weighted absolute height ``H`` divided by stellar radius.
``vzbar_cm_s`` : float64 array, shape ``(Nx, Ny)``
    Density-weighted absolute vertical velocity in cm/s.
``tau_R`` : float64 array, shape ``(Nx, Ny)``
    Dimensionless Rosseland optical depth integrated through the aperture.
``tc_tdyn``, ``tv_tdyn``, ``tdiff_tdyn``, ``tesc_tdyn`` : float64 arrays, shape ``(Nx, Ny)``
    The four times defined above divided by stellar dynamical time
    ``sqrt(R_star**3/(G M_star))``; all dimensionless.
``tc_over_tv``, ``tdiff_over_tv``, ``tesc_over_tv`` : float64 arrays, shape ``(Nx, Ny)``
    Emission, diffusion and escape time divided by vertical-flow time.
``effective_over_tv`` : float64 array, shape ``(Nx, Ny)``
    ``max(tc, tesc)/tv``, dimensionless.

Previously produced caches may additionally contain ``resolution``, physical
scales, direction metadata or dimensional time/height maps. These are optional
legacy keys and are not written by the current calculation; the keys above are
the current contract. Old cubic caches can use ``resolution`` instead of
``resolution_x``, ``resolution_y``, ``resolution_z``, and omit z-spacing metadata.
The cache reader treats those as linear cubic grids.

Usage
-----
Run from ``/home/hey4/rich_tde`` in the richanalysis environment::

    python works/cooling-checks/nozzle-timescale-validation.py --mode 1 --resolution 256 --resolution 384
    python works/cooling-checks/nozzle-timescale-validation.py --mode 1 --snapshot-number 108 --resolution-xy 256 --resolution-z 512 --z-spacing sinh

Complete map caches and existing figures are reused; ``--overwrite`` replaces
them. The summary is always rewritten for the requested grids. Cache validation
checks required fields and shape only. Use a new ``--output-root`` or overwrite
when changing source data, direction or sinh scale; those are not fully encoded
in filenames. Full grids can require a cluster job.

Loading examples
----------------
Inspect one map and its wedge median; ``map[i,j]`` is at ``(x[i],y[j])``::

    from pathlib import Path
    import numpy as np
    import dev
    import matplotlib.pyplot as plt

    root = Path("data/processed/CoolingChecks/nozzle-timescale-series")
    run_root = root / "stage2-timescales/1e4"
    with np.load(run_root / "maps/timescales_snap_0108_256.npz") as data:
        x, y = data["x_rp"], data["y_rp"]
        ratio = data["effective_over_tv"]
        wedge = data["wedge_mask"]
        print(data["time_tfb"].item())
    print(np.nanmedian(ratio[wedge]))
    fig, ax = plt.subplots()
    image = ax.pcolormesh(x, y, np.ma.masked_invalid(ratio.T), shading="nearest")
    ax.set(xlabel="x/r_p", ylabel="y/r_p", aspect="equal")
    fig.colorbar(image, ax=ax, label="max(tc, tesc)/tv")
    plt.show()

Load the mixed-type summary without guessing columns::

    table = np.loadtxt(run_root / "summary_snap_0108.txt", dtype=str,
                       skiprows=1, ndmin=2)
    resolution = table[:, 0].astype(int)
    statistic = table[:, 1]
    effective_ratio = table[:, 4].astype(float)
    print(effective_ratio[statistic == "median"])
"""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import nozzle_timescales as nt
import numpy as np
import typer
from loguru import logger

import dev  # noqa: F401

OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/nozzle-timescale-series/stage2-timescales"
)
DIRECTION_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/nozzle-timescale-series/stage1-wedge-selection"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
SNAPSHOT = {"1e4": 108, "1e5": 142, "1e6": 961}
RESOLUTIONS = (256, 384)


def render(cache, destination):
    with np.load(cache) as data:
        x, y, wedge = data["x_rp"], data["y_rp"], data["wedge_mask"]
        if "resolution_x" in data:
            resolution = tuple(int(data[f"resolution_{axis}"]) for axis in "xyz")
        else:
            resolution = (int(data["resolution"]),) * 3
        panels = (
            (
                data["dissipation_column_erg_s_cm2"],
                r"$D$ [erg s$^{-1}$ cm$^{-2}$]",
                "viridis",
                None,
            ),
            (data["sigma_g_cm2"], r"$\Sigma$ [g cm$^{-2}$]", "magma", None),
            (data["tau_R"], r"$\tau_R$", "magma", None),
            (data["tc_over_tv"], r"$t_c/t_v$", "coolwarm", (-1, 1)),
            (data["tesc_over_tv"], r"$t_{\rm esc}/t_v$", "coolwarm", (-1, 1)),
            (
                data["effective_over_tv"],
                r"$\max(t_c,t_{\rm esc})/t_v$",
                "coolwarm",
                (-1, 1),
            ),
        )
        logger.info(
            "{} snapshot {} at {:.3f} t_fb, {}x{}x{} {} grid",
            str(data["run"]),
            int(data["snapnum"]),
            float(data["time_tfb"]),
            *resolution,
            str(data["z_spacing"]) if "z_spacing" in data else "linear",
        )

    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, (values, label, cmap, limits) in zip(axes.flat, panels):
        values = np.log10(np.where(values > 0, values, np.nan))
        if limits is None:
            limits = np.nanpercentile(values[np.isfinite(values)], (1, 99))
        image = ax.pcolormesh(
            x,
            y,
            values.T,
            shading="nearest",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.contour(xgrid, ygrid, wedge, levels=(0.5,), colors="cyan", linewidths=0.8)
        ax.set_title(rf"$\log_{{10}}$({label})")
        ax.set(xlabel=r"$x/r_p$", ylabel=r"$y/r_p$", aspect="equal")
        fig.colorbar(image, ax=ax)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def write_summary(path, caches):
    rows = [row for cache in caches for row in nt.summarize_cache(cache)]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("resolution statistic tc/tv tesc/tv effective/tv\n")
        for row in rows:
            stream.write(
                f"{row['resolution']:10d} {row['statistic']:25s} "
                f"{row['tc_over_tv']:.6e} {row['tesc_over_tv']:.6e} "
                f"{row['effective_over_tv']:.6e}\n"
            )


def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6"),
    resolution: list[int] = typer.Option(list(RESOLUTIONS), "--resolution", "-r"),
    resolution_xy: int | None = typer.Option(None),
    resolution_z: int | None = typer.Option(None),
    z_spacing: str = typer.Option("linear"),
    sinh_scale_rp: float = typer.Option(0.1),
    snapshot_number: int | None = typer.Option(
        None, help="Override the representative snapshot"
    ),
    workers: int = typer.Option(8, min=1, help="Nearest-cell query threads"),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Root for maps, figures and grid summaries"
    ),
    direction_root: Path = typer.Option(
        DIRECTION_ROOT, help="Root containing wedge direction NPZ files"
    ),
    overwrite: bool = typer.Option(False, help="Recalculate maps and replace figures"),
):
    """Compare cooling maps across grids using cached native-peak wedge directions."""
    run = RUN_BY_MODE[mode]
    snapnum = SNAPSHOT[run] if snapshot_number is None else snapshot_number
    snapshot = nt.snapshot_path(run, snapnum)
    direction = nt.load_direction(run, snapnum, direction_root)
    output = output_root / run
    if resolution_xy or resolution_z or z_spacing != "linear":
        shapes = [(resolution_xy or 256, resolution_xy or 256, resolution_z or 512)]
    else:
        shapes = [(n, n, n) for n in resolution]

    caches = []
    for shape in shapes:
        nx, ny, nz = shape
        grid = (
            str(nx)
            if nx == ny == nz and z_spacing == "linear"
            else f"{nx}x{ny}x{nz}_{z_spacing}z"
        )
        stem = f"timescales_snap_{snapnum:04d}_{grid}"
        cache = output / "maps" / f"{stem}.npz"
        if overwrite or not nt.cache_complete(cache, shape):
            nt.calculate_snapshot(
                snapshot,
                cache,
                nt.config_for(run),
                shape,
                workers,
                direction,
                z_spacing,
                sinh_scale_rp,
            )
        figure = output / "figures" / f"{stem}.png"
        if overwrite or not figure.is_file():
            render(cache, figure)
        caches.append(cache)

    summary = output / f"summary_snap_{snapnum:04d}.txt"
    write_summary(summary, caches)
    logger.info("Summary -> {}", summary)


if __name__ == "__main__":
    typer.run(main)
