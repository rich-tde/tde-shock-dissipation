#!/usr/bin/env python3
"""Validate dissipation-percentile selections for nozzle timescale columns.

Stage 1 only: interpolate density and dissipation inside ``r < 3 r_p``,
integrate along z, compare candidate dissipation percentiles, and write the
selection-quality figures and tables needed for the first review gate.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nozzle-selection")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from loguru import logger
from scipy import ndimage

import richio
from dev import DATAPATHS, SNAPSHOT_TFB
from dev.datapaths import TDE_PARAMETERS
from richio.plots import scalar_map


OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/"
    "nozzle-timescale-series/stage1-selection"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
REQUESTED_TFBS = {
    "1e4": (0.5, 1.0, 1.5, 2.0),
    "1e5": (0.3, 0.5),
    "1e6": (1.0, 1.2, 1.4, 1.5),
}
PERCENTILES = (99.0, 99.5, 99.9, 99.95, 99.99)
APERTURE_RP = 3.0
RESOLUTIONS = (256, 384)


@dataclass(frozen=True)
class RunConfig:
    mode: int
    run: str
    m_bh: float
    m_star: float
    r_star: float

    @property
    def r_p_code(self) -> float:
        return self.r_star * (self.m_bh / self.m_star) ** (1.0 / 3.0)

    @property
    def t_fb_code(self) -> float:
        return (
            math.pi
            / math.sqrt(2.0)
            * math.sqrt(self.r_star**3 / self.m_star)
            * math.sqrt(self.m_bh / self.m_star)
        )


RUNS = {
    mode: RunConfig(mode, run, *TDE_PARAMETERS[run])
    for mode, run in RUN_BY_MODE.items()
}


def selected_snapshots(run: str) -> list[tuple[int, Path, bool]]:
    """Return the current NozzleZoomSlices epochs plus the last snapshot."""

    selected = [
        (*SNAPSHOT_TFB(run, requested_tfb), False)
        for requested_tfb in REQUESTED_TFBS[run]
    ]
    snapnums, paths = DATAPATHS(run)
    selected.append((snapnums[-1], paths[-1], True))

    deduplicated = {}
    for snapnum, path, is_last in selected:
        old = deduplicated.get(snapnum)
        deduplicated[snapnum] = (Path(path), is_last or (old[1] if old else False))
    return [
        (snapnum, path, is_last)
        for snapnum, (path, is_last) in deduplicated.items()
    ]


def needs_reference_frame(run: str, path: Path) -> bool:
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def scalar_time(snapshot):
    time = snapshot.time
    value = float(np.asarray(time.to_value("code_time")).squeeze())
    return u.unyt_quantity(value, "code_time", registry=time.units.registry)


def bh_frame_coordinates(snapshot, config: RunConfig, path: Path):
    x, y, z = snapshot.X, snapshot.Y, snapshot.Z
    if needs_reference_frame(config.run, path):
        offset = dev.reference_frame_offset(
            t=scalar_time(snapshot),
            Mbh=config.m_bh * richio.units.mscale,
            Mstar=config.m_star * richio.units.mscale,
            Rstar=config.r_star * richio.units.lscale,
            beta=1,
        )
        x, y = x + offset[0], y + offset[1]
    return x, y, z


def cache_complete(path: Path, resolution: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return (
                data["dissipation_column"].shape == (resolution, resolution)
                and data["surface_density"].shape == (resolution, resolution)
                and all(
                    name in data
                    for name in (
                        "native_peak_x_rp",
                        "native_peak_y_rp",
                        "native_peak_z_rp",
                    )
                )
            )
    except (KeyError, OSError, ValueError):
        return False


def build_column_cache(
    snapshot_path: Path,
    destination: Path,
    config: RunConfig,
    resolution: int,
    workers: int,
) -> None:
    """Grid one snapshot and atomically cache its two Stage-1 columns."""

    snapshot = richio.load(str(snapshot_path))
    time = scalar_time(snapshot)
    x, y, z = bh_frame_coordinates(snapshot, config, snapshot_path)
    r_p = config.r_p_code * richio.units.lscale
    radius = np.sqrt(x**2 + y**2 + z**2)
    source_selection = np.asarray(radius < APERTURE_RP * r_p, dtype=bool)
    if not np.any(source_selection):
        raise ValueError(f"No source cells inside {APERTURE_RP} r_p for {snapshot_path}")
    source_indices = np.flatnonzero(source_selection)
    native_peak_index = source_indices[
        int(np.nanargmax(np.asarray(snapshot.dissipation[source_selection])))
    ]

    bound = APERTURE_RP * r_p
    indices, xspace, yspace, zspace = snapshot.to_3dgrid(
        res=(resolution, resolution, resolution),
        X=x,
        Y=y,
        Z=z,
        box_size=(-bound, -bound, -bound, bound, bound, bound),
        selection=source_selection,
        workers=workers,
    )
    dx = xspace[1] - xspace[0]
    dy = yspace[1] - yspace[0]
    dz = zspace[1] - zspace[0]
    # Broadcast one-dimensional coordinates instead of materializing three
    # full coordinate cubes; this matters for the 384^3 convergence run.
    spherical_mask = (
        xspace[:, None, None] ** 2
        + yspace[None, :, None] ** 2
        + zspace[None, None, :] ** 2
        < (APERTURE_RP * r_p) ** 2
    )

    dissipation_density = snapshot.dissipation[indices].in_cgs()
    density = snapshot.rho[indices].in_cgs()
    dissipation_column = np.sum(
        np.where(spherical_mask, dissipation_density, 0 * dissipation_density.units),
        axis=-1,
    ) * dz.in_cgs()
    surface_density = np.sum(
        np.where(spherical_mask, density, 0 * density.units), axis=-1
    ) * dz.in_cgs()
    eligible_columns = np.any(spherical_mask, axis=-1)

    arrays = {
        "run": np.asarray(config.run),
        "snapshot_path": np.asarray(str(snapshot_path)),
        "resolution": np.asarray(resolution),
        "snapnum": np.asarray(
            int(re.search(r"(\d+)\.h5$", snapshot_path.name).group(1))
        ),
        "time_tfb": np.asarray(float(time.to_value("code_time")) / config.t_fb_code),
        "r_p_cm": np.asarray(float(r_p.to_value("cm"))),
        "native_peak_x_rp": np.asarray(float((x[native_peak_index] / r_p).value)),
        "native_peak_y_rp": np.asarray(float((y[native_peak_index] / r_p).value)),
        "native_peak_z_rp": np.asarray(float((z[native_peak_index] / r_p).value)),
        "dx_rp": np.asarray(float((dx / r_p).to_value("dimensionless"))),
        "dy_rp": np.asarray(float((dy / r_p).to_value("dimensionless"))),
        "dz_rp": np.asarray(float((dz / r_p).to_value("dimensionless"))),
        "x_rp": np.asarray(xspace / r_p, dtype="float64"),
        "y_rp": np.asarray(yspace / r_p, dtype="float64"),
        "eligible_columns": eligible_columns,
        "dissipation_column": np.asarray(dissipation_column, dtype="float64"),
        "dissipation_column_unit": np.asarray(str(dissipation_column.units)),
        "surface_density": np.asarray(surface_density, dtype="float64"),
        "surface_density_unit": np.asarray(str(surface_density.units)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def percentile_metrics(
    dissipation_column: np.ndarray,
    eligible_columns: np.ndarray,
    x_rp: np.ndarray,
    y_rp: np.ndarray,
    percentile: float,
    native_peak_x_rp: float,
    native_peak_y_rp: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a high-dissipation mask and its selection-quality metrics."""

    candidates = eligible_columns & np.isfinite(dissipation_column) & (
        dissipation_column > 0
    )
    if not np.any(candidates):
        raise ValueError("No finite positive dissipation columns")
    threshold = float(np.percentile(dissipation_column[candidates], percentile))
    selection = candidates & (dissipation_column >= threshold)

    labels, component_count = ndimage.label(
        selection, structure=np.ones((3, 3), dtype=int)
    )
    component_sizes = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(component_sizes)) + 1
    largest = labels == largest_label

    maximum_flat = int(
        np.nanargmax(np.where(candidates, dissipation_column, np.nan))
    )
    maximum_index = np.unravel_index(maximum_flat, dissipation_column.shape)
    maximum_label = int(labels[maximum_index])
    peak_component = labels == maximum_label
    selected_indices = np.argwhere(selection)
    selected_x = x_rp[selected_indices[:, 0]]
    selected_y = y_rp[selected_indices[:, 1]]
    anchor_distances = np.hypot(
        selected_x - native_peak_x_rp, selected_y - native_peak_y_rp
    )
    anchor_index = tuple(selected_indices[int(np.argmin(anchor_distances))])
    anchor_label = int(labels[anchor_index])
    anchor_component = labels == anchor_label
    native_grid_index = (
        int(np.argmin(np.abs(x_rp - native_peak_x_rp))),
        int(np.argmin(np.abs(y_rp - native_peak_y_rp))),
    )
    selected_power = float(np.sum(dissipation_column[selection]))
    candidate_power = float(np.sum(dissipation_column[candidates]))
    selected_pixels = int(np.count_nonzero(selection))

    metrics = {
        "percentile": percentile,
        "threshold": threshold,
        "candidate_pixels": int(np.count_nonzero(candidates)),
        "selected_pixels": selected_pixels,
        "component_count": int(component_count),
        "captured_dissipation_fraction": selected_power / candidate_power,
        "largest_component_pixel_fraction": float(np.count_nonzero(largest))
        / selected_pixels,
        "largest_component_dissipation_fraction": float(
            np.sum(dissipation_column[largest]) / selected_power
        ),
        "peak_component_pixel_fraction": float(np.count_nonzero(peak_component))
        / selected_pixels,
        "peak_component_dissipation_fraction": float(
            np.sum(dissipation_column[peak_component]) / selected_power
        ),
        "nozzle_anchor_component_pixel_fraction": float(
            np.count_nonzero(anchor_component)
        )
        / selected_pixels,
        "nozzle_anchor_component_dissipation_fraction": float(
            np.sum(dissipation_column[anchor_component]) / selected_power
        ),
        "column_peak_in_nozzle_anchor_component": int(maximum_label == anchor_label),
        "native_peak_grid_pixel_selected": int(selection[native_grid_index]),
        "nearest_selected_to_native_peak_rp": float(np.min(anchor_distances)),
        "column_peak_native_offset_rp": float(
            math.hypot(
                x_rp[maximum_index[0]] - native_peak_x_rp,
                y_rp[maximum_index[1]] - native_peak_y_rp,
            )
        ),
        "native_peak_x_rp": native_peak_x_rp,
        "native_peak_y_rp": native_peak_y_rp,
        "peak_x_rp": float(x_rp[maximum_index[0]]),
        "peak_y_rp": float(y_rp[maximum_index[1]]),
        "peak_dissipation_column": float(dissipation_column[maximum_index]),
    }
    return selection, metrics


def positive_log10(values):
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def render_validation(cache_path: Path, output_path: Path) -> list[dict]:
    with np.load(cache_path) as data:
        run = str(data["run"])
        snapnum = int(data["snapnum"])
        resolution = int(data["resolution"])
        time_tfb = float(data["time_tfb"])
        x_rp = np.array(data["x_rp"])
        y_rp = np.array(data["y_rp"])
        eligible = np.array(data["eligible_columns"], dtype=bool)
        dissipation = np.array(data["dissipation_column"])
        surface_density = np.array(data["surface_density"])
        dissipation_unit = str(data["dissipation_column_unit"])
        surface_density_unit = str(data["surface_density_unit"])
        native_peak_x_rp = float(data["native_peak_x_rp"])
        native_peak_y_rp = float(data["native_peak_y_rp"])

    metrics_rows = []
    selections = {}
    for percentile in PERCENTILES:
        selections[percentile], metrics = percentile_metrics(
            dissipation,
            eligible,
            x_rp,
            y_rp,
            percentile,
            native_peak_x_rp,
            native_peak_y_rp,
        )
        metrics_rows.append(
            {
                "run": run,
                "snapnum": snapnum,
                "resolution": resolution,
                "time_tfb": time_tfb,
                **metrics,
            }
        )

    log_dissipation = positive_log10(dissipation)
    log_surface_density = positive_log10(surface_density)
    finite_dissipation = log_dissipation[np.isfinite(log_dissipation) & eligible]
    finite_surface_density = log_surface_density[np.isfinite(log_surface_density) & eligible]
    diss_limits = tuple(np.nanpercentile(finite_dissipation, (1, 100)))
    sigma_limits = tuple(np.nanpercentile(finite_surface_density, (1, 100)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    scalar_map(
        log_surface_density,
        x_rp,
        y_rp,
        ax=axes[0, 0],
        log_scale=False,
        cmap="magma",
        vmin=sigma_limits[0],
        vmax=sigma_limits[1],
        colorbar_label=rf"$\log_{{10}}(\Sigma/[{surface_density_unit}])$",
    )
    axes[0, 0].set_title("Surface density")
    scalar_map(
        log_dissipation,
        x_rp,
        y_rp,
        ax=axes[0, 1],
        log_scale=False,
        cmap="viridis",
        vmin=diss_limits[0],
        vmax=diss_limits[1],
        colorbar_label=rf"$\log_{{10}}(D/[{dissipation_unit}])$",
    )
    axes[0, 1].set_title("Integrated dissipation")

    overlay_axes = list(axes.flat[2:7])
    for ax, percentile, row in zip(overlay_axes, PERCENTILES, metrics_rows):
        scalar_map(
            log_dissipation,
            x_rp,
            y_rp,
            ax=ax,
            log_scale=False,
            cmap="Greys",
            vmin=diss_limits[0],
            vmax=diss_limits[1],
            colorbar_label=r"$\log_{10}D$",
        )
        xgrid, ygrid = np.meshgrid(x_rp, y_rp, indexing="ij")
        labels, _ = ndimage.label(
            selections[percentile], structure=np.ones((3, 3), dtype=int)
        )
        peak_index = np.unravel_index(
            int(np.nanargmax(np.where(eligible, dissipation, np.nan))),
            dissipation.shape,
        )
        peak_component = labels == labels[peak_index]
        selected_indices = np.argwhere(selections[percentile])
        anchor_index = tuple(
            selected_indices[
                int(
                    np.argmin(
                        np.hypot(
                            x_rp[selected_indices[:, 0]] - native_peak_x_rp,
                            y_rp[selected_indices[:, 1]] - native_peak_y_rp,
                        )
                    )
                )
            ]
        )
        anchor_component = labels == labels[anchor_index]
        ax.contourf(
            xgrid,
            ygrid,
            np.ma.masked_where(labels == 0, labels),
            levels=np.arange(0.5, labels.max() + 1.5),
            cmap="tab20",
            alpha=0.45,
        )
        ax.contour(
            xgrid,
            ygrid,
            selections[percentile].astype(int),
            levels=[0.5],
            colors=["C3"],
            linewidths=1.2,
        )
        ax.contour(
            xgrid,
            ygrid,
            peak_component.astype(int),
            levels=[0.5],
            colors=["C1"],
            linewidths=2.0,
        )
        ax.contour(
            xgrid,
            ygrid,
            anchor_component.astype(int),
            levels=[0.5],
            colors=["C0"],
            linewidths=1.6,
            linestyles="--",
        )
        ax.plot(row["peak_x_rp"], row["peak_y_rp"], "x", color="C1", ms=8, mew=2)
        ax.plot(native_peak_x_rp, native_peak_y_rp, "+", color="C0", ms=10, mew=2)
        ax.set_title(
            f"p{percentile:g}: {row['selected_pixels']} px, "
            f"{row['captured_dissipation_fraction']:.1%} D\n"
            f"{row['component_count']} components; peak component "
            f"{row['peak_component_pixel_fraction']:.0%}\n"
            rf"$\Delta_{{\rm native}}={row['column_peak_native_offset_rp']:.2f}r_p$; "
            rf"anchor distance $={row['nearest_selected_to_native_peak_rp']:.2f}r_p$"
        )

    table_ax = axes[1, 3]
    table_ax.axis("off")
    table_data = [
        [
            f"{row['percentile']:g}",
            str(row["selected_pixels"]),
            f"{row['captured_dissipation_fraction']:.3f}",
            str(row["component_count"]),
            f"{row['nozzle_anchor_component_dissipation_fraction']:.3f}",
            "yes" if row["column_peak_in_nozzle_anchor_component"] else "no",
        ]
        for row in metrics_rows
    ]
    table_ax.table(
        cellText=table_data,
        colLabels=("pct", "pixels", "D frac", "N comp", "anchor D", "peak=anchor"),
        loc="center",
        cellLoc="center",
    )
    table_ax.set_title("Selection metrics")

    for ax in axes.flat[:7]:
        ax.set_xlim(-APERTURE_RP, APERTURE_RP)
        ax.set_ylim(-APERTURE_RP, APERTURE_RP)
        ax.set_xlabel(r"$x/r_p$")
        ax.set_ylabel(r"$y/r_p$")
    fig.suptitle(
        rf"{run}, snapshot {snapnum}, $t/t_{{\rm fb}}={time_tfb:.3f}$, "
        rf"${resolution}^3$, $r<3r_p$"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return metrics_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def convergence_rows(rows: list[dict], resolutions: tuple[int, ...]) -> list[dict]:
    by_key = {(row["snapnum"], row["percentile"], row["resolution"]): row for row in rows}
    output = []
    for snapnum in sorted({row["snapnum"] for row in rows}):
        for percentile in PERCENTILES:
            low = by_key[(snapnum, percentile, resolutions[0])]
            high = by_key[(snapnum, percentile, resolutions[1])]
            peak_shift = math.hypot(
                high["peak_x_rp"] - low["peak_x_rp"],
                high["peak_y_rp"] - low["peak_y_rp"],
            )
            output.append(
                {
                    "run": low["run"],
                    "snapnum": snapnum,
                    "percentile": percentile,
                    "low_resolution": resolutions[0],
                    "high_resolution": resolutions[1],
                    "captured_dissipation_fraction_low": low[
                        "captured_dissipation_fraction"
                    ],
                    "captured_dissipation_fraction_high": high[
                        "captured_dissipation_fraction"
                    ],
                    "captured_dissipation_fraction_difference": high[
                        "captured_dissipation_fraction"
                    ]
                    - low["captured_dissipation_fraction"],
                    "peak_shift_rp": peak_shift,
                    "peak_component_pixel_fraction_low": low[
                        "peak_component_pixel_fraction"
                    ],
                    "peak_component_pixel_fraction_high": high[
                        "peak_component_pixel_fraction"
                    ],
                    "nozzle_anchor_component_pixel_fraction_low": low[
                        "nozzle_anchor_component_pixel_fraction"
                    ],
                    "nozzle_anchor_component_pixel_fraction_high": high[
                        "nozzle_anchor_component_pixel_fraction"
                    ],
                    "nozzle_anchor_component_dissipation_fraction_low": low[
                        "nozzle_anchor_component_dissipation_fraction"
                    ],
                    "nozzle_anchor_component_dissipation_fraction_high": high[
                        "nozzle_anchor_component_dissipation_fraction"
                    ],
                    "column_peak_in_nozzle_anchor_component_low": low[
                        "column_peak_in_nozzle_anchor_component"
                    ],
                    "column_peak_in_nozzle_anchor_component_high": high[
                        "column_peak_in_nozzle_anchor_component"
                    ],
                    "native_peak_grid_pixel_selected_low": low[
                        "native_peak_grid_pixel_selected"
                    ],
                    "native_peak_grid_pixel_selected_high": high[
                        "native_peak_grid_pixel_selected"
                    ],
                    "nearest_selected_to_native_peak_rp_low": low[
                        "nearest_selected_to_native_peak_rp"
                    ],
                    "nearest_selected_to_native_peak_rp_high": high[
                        "nearest_selected_to_native_peak_rp"
                    ],
                }
            )
    return output


def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6"),
    workers: int = typer.Option(8, min=1, help="KD-tree query threads"),
    output_root: Path = typer.Option(OUTPUT_ROOT, help="Stage-1 output directory"),
    overwrite: bool = typer.Option(False, help="Rebuild existing column caches"),
    rerender: bool = typer.Option(False, help="Redraw figures from valid caches"),
    list_only: bool = typer.Option(False, help="List work without loading snapshots"),
    snapshot_index: int | None = typer.Option(
        None, min=0, help="Only process this zero-based representative-snapshot index"
    ),
    resolutions: str = typer.Option(
        "256,384", help="Comma-separated grid resolutions (small values are for smoke tests)"
    ),
) -> None:
    config = RUNS[mode]
    selected = selected_snapshots(config.run)
    if snapshot_index is not None:
        if snapshot_index >= len(selected):
            raise typer.BadParameter(
                f"snapshot-index must be smaller than {len(selected)} for {config.run}"
            )
        selected = [selected[snapshot_index]]
    try:
        requested_resolutions = tuple(int(value) for value in resolutions.split(","))
    except ValueError as error:
        raise typer.BadParameter("resolutions must be comma-separated integers") from error
    if not requested_resolutions or any(value < 2 for value in requested_resolutions):
        raise typer.BadParameter("every resolution must be at least 2")
    for snapnum, path, is_last in selected:
        logger.info("{} snapshot {}: {}{}", config.run, snapnum, path, " (last)" if is_last else "")
    if list_only:
        return

    run_root = output_root / config.run
    all_rows = []
    for resolution in requested_resolutions:
        for snapnum, snapshot_path, _ in selected:
            cache_path = run_root / "columns" / (
                f"selection_snap_{snapnum:04d}_{resolution}.npz"
            )
            figure_path = run_root / "figures" / (
                f"selection_snap_{snapnum:04d}_{resolution}.png"
            )
            if overwrite or not cache_complete(cache_path, resolution):
                logger.info("Gridding {} snapshot {} at {}^3", config.run, snapnum, resolution)
                build_column_cache(snapshot_path, cache_path, config, resolution, workers)
            else:
                logger.info("Using cached {}", cache_path)
            if rerender or not figure_path.is_file() or figure_path.stat().st_size == 0:
                logger.info("Rendering {}", figure_path)
            rows = render_validation(cache_path, figure_path)
            all_rows.extend(rows)

    write_csv(run_root / "selection-metrics.csv", all_rows)
    if len(requested_resolutions) >= 2:
        write_csv(
            run_root / "selection-convergence.csv",
            convergence_rows(all_rows, requested_resolutions),
        )
    logger.info("Stage 1 complete for {}: {} metric rows", config.run, len(all_rows))


if __name__ == "__main__":
    typer.run(main)
