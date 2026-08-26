#!/usr/bin/env python3
"""Compare nozzle cooling timescales at two grid resolutions."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import nozzle_timescales as nt

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
    snapshot_number: int | None = typer.Option(None),
    workers: int = typer.Option(8),
    output_root: Path = typer.Option(OUTPUT_ROOT),
    direction_root: Path = typer.Option(DIRECTION_ROOT),
    overwrite: bool = typer.Option(False),
):
    run = RUN_BY_MODE[mode]
    snapnum = snapshot_number or SNAPSHOT[run]
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
