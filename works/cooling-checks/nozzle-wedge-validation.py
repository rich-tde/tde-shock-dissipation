#!/usr/bin/env python3
"""Validate annular-maximum-centred nozzle wedges on cached column maps.

The direction is found from the maximum integrated-dissipation column in either
0.6 < R/r_p < 1.75 or 0.8 < R/r_p < 1.75.  In both cases the final wedge keeps
0.6 < R/r_p < 1.75 and |Delta phi| < 4.5 degrees.  This is a Stage-1 selection
test only; no timescales are calculated.
"""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nozzle-wedge")

import dev
import matplotlib.pyplot as plt
import numpy as np
import richio
import typer
import unyt as u
from loguru import logger

from dev.datapaths import TDE_PARAMETERS
from richio.plots import scalar_map


INPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/"
    "nozzle-timescale-series/stage1-selection"
)
OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/"
    "nozzle-timescale-series/stage1-wedge-selection"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
RESOLUTIONS = (256, 384)
WEDGE_RADIUS_MIN_RP = 0.6
DIRECTION_RADIUS_MINIMA_RP = (0.6, 0.8)
RADIUS_MAX_RP = 1.75
ANGULAR_HALF_WIDTH_DEG = 4.5


def wrapped_angular_distance(angle, center):
    return (angle - center + np.pi) % (2 * np.pi) - np.pi


def positive_log10(values):
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def needs_reference_frame(run: str, path: Path) -> bool:
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def scalar_time(snapshot):
    time = snapshot.time
    value = float(np.asarray(time.to_value("code_time")).squeeze())
    return u.unyt_quantity(value, "code_time", registry=time.units.registry)


def direction_cache_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return all(
                f"direction_peak_{axis}_rp_dirmin_{str(radius).replace('.', 'p')}"
                in data
                for radius in DIRECTION_RADIUS_MINIMA_RP
                for axis in ("x", "y", "z")
            )
    except (OSError, ValueError):
        return False


def build_direction_cache(column_cache: Path, destination: Path, run: str) -> None:
    """Find native-cell dissipation maxima inside both direction shells."""

    with np.load(column_cache) as data:
        snapshot_path = Path(str(data["snapshot_path"]))
        snapnum = int(data["snapnum"])
    snapshot = richio.load(str(snapshot_path))
    x, y, z = snapshot.X, snapshot.Y, snapshot.Z
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    if needs_reference_frame(run, snapshot_path):
        offset = dev.reference_frame_offset(
            t=scalar_time(snapshot),
            Mbh=m_bh * richio.units.mscale,
            Mstar=m_star * richio.units.mscale,
            Rstar=r_star * richio.units.lscale,
            beta=1,
        )
        x, y = x + offset[0], y + offset[1]
    r_p = r_star * (m_bh / m_star) ** (1.0 / 3.0) * richio.units.lscale
    radius = np.sqrt(x**2 + y**2 + z**2)

    arrays = {
        "run": np.asarray(run),
        "snapnum": np.asarray(snapnum),
        "snapshot_path": np.asarray(str(snapshot_path)),
    }
    for radius_min_rp in DIRECTION_RADIUS_MINIMA_RP:
        shell = (radius >= radius_min_rp * r_p) & (radius <= RADIUS_MAX_RP * r_p)
        shell_indices = np.flatnonzero(np.asarray(shell, dtype=bool))
        if not len(shell_indices):
            raise ValueError(
                f"No native cells in {radius_min_rp}<r/r_p<{RADIUS_MAX_RP} "
                f"for {snapshot_path}"
            )
        local_index = int(
            np.nanargmax(np.asarray(snapshot.dissipation[shell_indices]))
        )
        peak_index = int(shell_indices[local_index])
        label = str(radius_min_rp).replace(".", "p")
        arrays[f"direction_peak_x_rp_dirmin_{label}"] = np.asarray(
            float((x[peak_index] / r_p).value)
        )
        arrays[f"direction_peak_y_rp_dirmin_{label}"] = np.asarray(
            float((y[peak_index] / r_p).value)
        )
        arrays[f"direction_peak_z_rp_dirmin_{label}"] = np.asarray(
            float((z[peak_index] / r_p).value)
        )
        arrays[f"direction_peak_dissipation_cgs_dirmin_{label}"] = np.asarray(
            float(snapshot.dissipation[peak_index].in_cgs().value)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def load_direction(path: Path, radius_min_rp: float) -> dict[str, float]:
    label = str(radius_min_rp).replace(".", "p")
    with np.load(path) as data:
        x = float(data[f"direction_peak_x_rp_dirmin_{label}"])
        y = float(data[f"direction_peak_y_rp_dirmin_{label}"])
        z = float(data[f"direction_peak_z_rp_dirmin_{label}"])
        dissipation = float(
            data[f"direction_peak_dissipation_cgs_dirmin_{label}"]
        )
    return {"x": x, "y": y, "z": z, "dissipation_cgs": dissipation}


def cache_paths(input_root: Path, run: str) -> dict[tuple[int, int], Path]:
    output = {}
    pattern = re.compile(r"selection_snap_(\d+)_(\d+)\.npz$")
    for path in sorted((input_root / run / "columns").glob("*.npz")):
        match = pattern.fullmatch(path.name)
        if match:
            output[(int(match.group(1)), int(match.group(2)))] = path
    return output


def wedge_metrics(
    cache_path: Path,
    direction_radius_min_rp: float,
    direction_peak: dict[str, float],
) -> tuple[dict, dict[str, np.ndarray]]:
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
        native_x = float(data["native_peak_x_rp"])
        native_y = float(data["native_peak_y_rp"])

    xgrid, ygrid = np.meshgrid(x_rp, y_rp, indexing="ij")
    radius = np.hypot(xgrid, ygrid)
    azimuth = np.arctan2(ygrid, xgrid)
    positive = eligible & np.isfinite(dissipation) & (dissipation > 0)
    direction = math.atan2(direction_peak["y"], direction_peak["x"])
    angular_distance = wrapped_angular_distance(azimuth, direction)
    final_annulus = (
        positive
        & (radius >= WEDGE_RADIUS_MIN_RP)
        & (radius <= RADIUS_MAX_RP)
    )
    wedge = final_annulus & (
        np.abs(angular_distance) <= math.radians(ANGULAR_HALF_WIDTH_DEG)
    )
    if not np.any(wedge):
        raise ValueError(f"Empty wedge in {cache_path}")

    wedge_indices = np.argwhere(wedge)
    wedge_values = dissipation[wedge]
    local_peak = tuple(wedge_indices[int(np.argmax(wedge_values))])
    global_flat = int(np.nanargmax(np.where(positive, dissipation, np.nan)))
    global_peak = np.unravel_index(global_flat, dissipation.shape)
    total_power = float(np.sum(dissipation[positive]))
    annulus_power = float(np.sum(dissipation[final_annulus]))
    wedge_power = float(np.sum(wedge_values))
    weights = wedge_values / wedge_power
    weighted_radius = float(np.sum(radius[wedge] * weights))
    weighted_angle_offset = math.degrees(
        math.atan2(
            float(np.sum(np.sin(angular_distance[wedge]) * weights)),
            float(np.sum(np.cos(angular_distance[wedge]) * weights)),
        )
    )

    row = {
        "run": run,
        "snapnum": snapnum,
        "resolution": resolution,
        "time_tfb": time_tfb,
        "direction_source": "maximum native-cell dissipation density in shell",
        "direction_radius_min_rp": direction_radius_min_rp,
        "wedge_radius_min_rp": WEDGE_RADIUS_MIN_RP,
        "radius_max_rp": RADIUS_MAX_RP,
        "angular_half_width_deg": ANGULAR_HALF_WIDTH_DEG,
        "direction_deg": math.degrees(direction),
        "direction_peak_x_rp": direction_peak["x"],
        "direction_peak_y_rp": direction_peak["y"],
        "direction_peak_z_rp": direction_peak["z"],
        "direction_peak_radius_rp": math.sqrt(
            direction_peak["x"] ** 2
            + direction_peak["y"] ** 2
            + direction_peak["z"] ** 2
        ),
        "direction_peak_projected_radius_rp": math.hypot(
            direction_peak["x"], direction_peak["y"]
        ),
        "direction_peak_dissipation_cgs": direction_peak["dissipation_cgs"],
        "selected_pixels": int(np.count_nonzero(wedge)),
        "captured_total_dissipation_fraction": wedge_power / total_power,
        "captured_annulus_dissipation_fraction": wedge_power / annulus_power,
        "global_column_peak_in_wedge": int(wedge[global_peak]),
        "wedge_peak_x_rp": float(x_rp[local_peak[0]]),
        "wedge_peak_y_rp": float(y_rp[local_peak[1]]),
        "wedge_peak_radius_rp": float(radius[local_peak]),
        "wedge_peak_angle_offset_deg": math.degrees(
            float(angular_distance[local_peak])
        ),
        "weighted_radius_rp": weighted_radius,
        "weighted_angle_offset_deg": weighted_angle_offset,
        "native_peak_x_rp": native_x,
        "native_peak_y_rp": native_y,
        "native_peak_projected_radius_rp": math.hypot(native_x, native_y),
        "wedge_peak_dissipation_column": float(dissipation[local_peak]),
    }
    arrays = {
        "x_rp": x_rp,
        "y_rp": y_rp,
        "dissipation": dissipation,
        "surface_density": surface_density,
        "eligible": eligible,
        "wedge": wedge,
        "native_x": native_x,
        "native_y": native_y,
        "dissipation_unit": dissipation_unit,
        "surface_density_unit": surface_density_unit,
    }
    return row, arrays


def render(row: dict, arrays: dict, destination: Path) -> None:
    x_rp = arrays["x_rp"]
    y_rp = arrays["y_rp"]
    dissipation = arrays["dissipation"]
    surface_density = arrays["surface_density"]
    eligible = arrays["eligible"]
    wedge = arrays["wedge"]
    log_dissipation = positive_log10(dissipation)
    log_surface_density = positive_log10(surface_density)
    diss_limits = np.nanpercentile(log_dissipation[np.isfinite(log_dissipation) & eligible], (1, 100))
    sigma_limits = np.nanpercentile(log_surface_density[np.isfinite(log_surface_density) & eligible], (1, 100))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
    scalar_map(
        log_surface_density,
        x_rp,
        y_rp,
        ax=axes[0],
        log_scale=False,
        cmap="magma",
        vmin=sigma_limits[0],
        vmax=sigma_limits[1],
        colorbar_label=rf"$\log_{{10}}(\Sigma/[{arrays['surface_density_unit']}])$",
    )
    axes[0].set_title("Surface density")
    for ax, title in zip(axes[1:], ("Integrated dissipation", "Wedge selection")):
        scalar_map(
            log_dissipation,
            x_rp,
            y_rp,
            ax=ax,
            log_scale=False,
            cmap="viridis" if ax is axes[1] else "Greys",
            vmin=diss_limits[0],
            vmax=diss_limits[1],
            colorbar_label=rf"$\log_{{10}}(D/[{arrays['dissipation_unit']}])$",
        )
        ax.set_title(title)

    xgrid, ygrid = np.meshgrid(x_rp, y_rp, indexing="ij")
    axes[2].contourf(
        xgrid,
        ygrid,
        np.ma.masked_where(~wedge, wedge.astype(float)),
        levels=(0.5, 1.5),
        colors=("C0",),
        alpha=0.35,
    )
    axes[2].contour(
        xgrid,
        ygrid,
        wedge.astype(int),
        levels=(0.5,),
        colors=("C0",),
        linewidths=1.5,
    )
    direction = math.radians(row["direction_deg"])
    axes[2].plot(
        np.array((WEDGE_RADIUS_MIN_RP, RADIUS_MAX_RP)) * math.cos(direction),
        np.array((WEDGE_RADIUS_MIN_RP, RADIUS_MAX_RP)) * math.sin(direction),
        color="C0",
        linestyle="--",
        linewidth=1.2,
    )
    axes[2].plot(
        arrays["native_x"],
        arrays["native_y"],
        "+",
        color="C0",
        ms=11,
        mew=2,
        label="unrestricted native max",
    )
    axes[2].plot(
        row["direction_peak_x_rp"],
        row["direction_peak_y_rp"],
        "D",
        color="C3",
        markerfacecolor="none",
        ms=8,
        mew=1.8,
        label="shell native max",
    )
    axes[2].plot(
        row["wedge_peak_x_rp"],
        row["wedge_peak_y_rp"],
        "x",
        color="C1",
        ms=9,
        mew=2,
        label="wedge column max",
    )
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.8)
    axes[2].text(
        0.02,
        0.02,
        f"{row['selected_pixels']} pixels\n"
        f"{row['captured_total_dissipation_fraction']:.1%} of total D\n"
        f"{row['captured_annulus_dissipation_fraction']:.1%} of annulus D\n"
        rf"direction peak: {row['direction_peak_radius_rp']:.2f}$r_p$"
        "\n"
        rf"wedge peak: {row['wedge_peak_radius_rp']:.2f}$r_p$, "
        rf"$\Delta\phi={row['wedge_peak_angle_offset_deg']:+.1f}^\circ$",
        transform=axes[2].transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(r"$x/r_p$")
        ax.set_ylabel(r"$y/r_p$")
    fig.suptitle(
        rf"{row['run']}, snapshot {row['snapnum']}, "
        rf"$t/t_{{\rm fb}}={row['time_tfb']:.3f}$, ${row['resolution']}^3$; "
        rf"direction search: ${row['direction_radius_min_rp']:.1f}<R/r_p<1.75$; "
        rf"wedge: $0.6<R/r_p<1.75$, $|\Delta\phi|<4.5^\circ$"
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


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


def convergence_rows(rows: list[dict]) -> list[dict]:
    by_key = {
        (row["snapnum"], row["direction_radius_min_rp"], row["resolution"]): row
        for row in rows
    }
    output = []
    for snapnum in sorted({row["snapnum"] for row in rows}):
        for direction_radius_min_rp in DIRECTION_RADIUS_MINIMA_RP:
            low = by_key[(snapnum, direction_radius_min_rp, RESOLUTIONS[0])]
            high = by_key[(snapnum, direction_radius_min_rp, RESOLUTIONS[1])]
            output.append(
                {
                    "run": low["run"],
                    "snapnum": snapnum,
                    "direction_radius_min_rp": direction_radius_min_rp,
                    "wedge_radius_min_rp": WEDGE_RADIUS_MIN_RP,
                    "low_resolution": RESOLUTIONS[0],
                    "high_resolution": RESOLUTIONS[1],
                    "selected_pixels_low": low["selected_pixels"],
                    "selected_pixels_high": high["selected_pixels"],
                    "captured_total_dissipation_fraction_low": low[
                        "captured_total_dissipation_fraction"
                    ],
                    "captured_total_dissipation_fraction_high": high[
                        "captured_total_dissipation_fraction"
                    ],
                    "captured_annulus_dissipation_fraction_low": low[
                        "captured_annulus_dissipation_fraction"
                    ],
                    "captured_annulus_dissipation_fraction_high": high[
                        "captured_annulus_dissipation_fraction"
                    ],
                    "wedge_peak_shift_rp": math.hypot(
                        high["wedge_peak_x_rp"] - low["wedge_peak_x_rp"],
                        high["wedge_peak_y_rp"] - low["wedge_peak_y_rp"],
                    ),
                    "wedge_peak_radius_rp_low": low["wedge_peak_radius_rp"],
                    "wedge_peak_radius_rp_high": high["wedge_peak_radius_rp"],
                    "direction_difference_deg": abs(
                        math.degrees(
                            wrapped_angular_distance(
                                math.radians(high["direction_deg"]),
                                math.radians(low["direction_deg"]),
                            )
                        )
                    ),
                }
            )
    return output


def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6"),
    input_root: Path = typer.Option(INPUT_ROOT, help="Stage-1 column-cache root"),
    output_root: Path = typer.Option(OUTPUT_ROOT, help="Wedge-validation output root"),
    overwrite: bool = typer.Option(False, help="Redraw existing figures"),
    snapshot_index: int | None = typer.Option(
        None, min=0, help="Only process this zero-based representative-snapshot index"
    ),
) -> None:
    run = RUN_BY_MODE[mode]
    paths = cache_paths(input_root, run)
    snapnums = sorted({snapnum for snapnum, _ in paths})
    missing = [
        (snapnum, resolution)
        for snapnum in snapnums
        for resolution in RESOLUTIONS
        if (snapnum, resolution) not in paths
    ]
    if not snapnums or missing:
        raise ValueError(f"Missing input caches for {run}: {missing}")
    if snapshot_index is not None:
        if snapshot_index >= len(snapnums):
            raise typer.BadParameter(
                f"snapshot-index must be smaller than {len(snapnums)} for {run}"
            )
        snapnums = [snapnums[snapshot_index]]

    run_root = output_root / run
    rows = []
    for snapnum in snapnums:
        direction_path = run_root / "directions" / f"direction_snap_{snapnum:04d}.npz"
        if not direction_cache_complete(direction_path):
            logger.info("Finding native shell maxima for {} snapshot {}", run, snapnum)
            build_direction_cache(
                paths[(snapnum, RESOLUTIONS[0])], direction_path, run
            )
        for resolution in RESOLUTIONS:
            for direction_radius_min_rp in DIRECTION_RADIUS_MINIMA_RP:
                direction_peak = load_direction(
                    direction_path, direction_radius_min_rp
                )
                row, arrays = wedge_metrics(
                    paths[(snapnum, resolution)],
                    direction_radius_min_rp,
                    direction_peak,
                )
                rows.append(row)
                radius_label = str(direction_radius_min_rp).replace(".", "p")
                figure_path = run_root / "figures" / (
                    f"wedge_snap_{snapnum:04d}_{resolution}_dirmin_{radius_label}.png"
                )
                if (
                    overwrite
                    or not figure_path.is_file()
                    or figure_path.stat().st_size == 0
                ):
                    logger.info("Rendering {}", figure_path)
                    render(row, arrays, figure_path)

    write_csv(run_root / "wedge-metrics.csv", rows)
    write_csv(run_root / "wedge-convergence.csv", convergence_rows(rows))
    logger.info("Wedge validation complete for {}: {} metric rows", run, len(rows))


if __name__ == "__main__":
    typer.run(main)
