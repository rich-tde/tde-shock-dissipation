"""Stage-3 restartable nozzle-cooling timeseries worker and aggregator."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nozzle-series")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import nozzle_timescales as VALIDATION
import numpy as np
import typer
from dev.datapaths import DATAPATHS
from loguru import logger

import richio

OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/CoolingChecks/"
    "nozzle-timescale-series/production-sinh"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
STATISTICS = VALIDATION.STATISTICS
PLOTTED_STATISTICS = (
    "median",
    "dissipation_weighted_mean",
    "max_dissipation_pixel",
)


def result_path(
    root: Path, run: str, snapnum: int, resolution: int, resolution_z: int
) -> Path:
    return (
        root
        / run
        / f"snap_{snapnum:04d}_{resolution}x{resolution}x{resolution_z}_sinhz.npz"
    )


def unavailable_complete(path: Path, run: str, snapnum: int) -> bool:
    try:
        with np.load(path) as data:
            return (
                str(data["run"]) == run
                and int(data["snapnum"]) == snapnum
                and str(data["status"]) == "no_aperture_material"
            )
    except (KeyError, OSError, ValueError):
        return False


def result_complete(
    path: Path, run: str, snapnum: int, shape: tuple[int, int, int]
) -> bool:
    return VALIDATION.cache_complete(path, shape) or unavailable_complete(
        path, run, snapnum
    )


def write_unavailable(
    path: Path, run: str, snapnum: int, snapshot_path: Path, reason: str
) -> None:
    snapshot = richio.load(str(snapshot_path))
    config = VALIDATION.config_for(run)
    time = VALIDATION.scalar_time(snapshot)
    VALIDATION.atomic_npz(
        path,
        {
            "run": np.asarray(run),
            "snapnum": np.asarray(snapnum),
            "snapshot_path": np.asarray(str(snapshot_path)),
            "time_tfb": np.asarray(float(time.to_value("code_time")) / config["t_fb"]),
            "time_days": np.asarray(float(time.to_value("day"))),
            "status": np.asarray("no_aperture_material"),
            "status_reason": np.asarray(reason),
        },
    )


def run_worker(
    mode: int,
    snapshot_index: int,
    resolution: int,
    resolution_z: int,
    workers: int,
    output_root: Path,
    overwrite: bool,
) -> None:
    run = RUN_BY_MODE[mode]
    snapnums, paths = DATAPATHS(run)
    if snapshot_index < 0 or snapshot_index >= len(snapnums):
        raise typer.BadParameter(
            f"snapshot-index {snapshot_index} outside 0..{len(snapnums) - 1} for {run}"
        )
    snapnum = snapnums[snapshot_index]
    snapshot_path = Path(paths[snapshot_index])
    shape = (resolution, resolution, resolution_z)
    destination = result_path(output_root, run, snapnum, resolution, resolution_z)
    if not overwrite and result_complete(destination, run, snapnum, shape):
        logger.info("Reusing valid result {}", destination)
        return
    try:
        VALIDATION.calculate_snapshot(
            snapshot_path,
            destination,
            VALIDATION.config_for(run),
            shape,
            workers,
            direction=None,
            z_spacing="sinh",
            sinh_scale_rp=0.1,
        )
    except ValueError as exc:
        message = str(exc)
        expected = (
            "No source cells inside",
            "Direction shell is empty",
            "Accepted wedge is empty",
        )
        if not any(token in message for token in expected):
            raise
        logger.warning("Recording unavailable snapshot: {}", message)
        write_unavailable(destination, run, snapnum, snapshot_path, message)


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def unavailable_rows(path: Path) -> list[dict]:
    with np.load(path) as data:
        common = {
            "run": str(data["run"]),
            "snapnum": int(data["snapnum"]),
            "snapshot_path": str(data["snapshot_path"]),
            "time_tfb": float(data["time_tfb"]),
            "time_days": float(data["time_days"]),
            "status": str(data["status"]),
            "status_reason": str(data["status_reason"]),
        }
    return [{**common, "statistic": statistic} for statistic in STATISTICS]


def classify(row: dict) -> None:
    tau = float(row.get("tau_R", np.nan))
    escape = float(row.get("tesc_over_tv", np.nan))
    emission = float(row.get("tc_over_tv", np.nan))
    effective = float(row.get("effective_over_tv", np.nan))
    if not (np.isfinite(tau) and np.isfinite(escape)):
        epoch = "unavailable"
    elif tau < 1:
        epoch = "optically_thin"
    elif escape < 1:
        epoch = "optically_thick_escape_efficient"
    else:
        epoch = "photon_trapped"
    row["epoch_class"] = epoch
    row["emission_limited"] = bool(emission >= 1) if np.isfinite(emission) else ""
    row["effectively_cooled"] = bool(effective < 1) if np.isfinite(effective) else ""


def aggregate(output_root: Path, resolution: int, resolution_z: int) -> list[dict]:
    rows: list[dict] = []
    missing = []
    invalid = []
    for run in RUN_BY_MODE.values():
        snapnums, _ = DATAPATHS(run)
        expected_names = {
            result_path(output_root, run, snapnum, resolution, resolution_z).name
            for snapnum in snapnums
        }
        actual_names = {path.name for path in (output_root / run).glob("*.npz")}
        extras = sorted(actual_names - expected_names)
        if extras:
            raise ValueError(
                f"Unexpected/duplicate-style outputs for {run}: {extras[:8]}"
            )
        for snapnum in snapnums:
            path = result_path(output_root, run, snapnum, resolution, resolution_z)
            if not path.is_file():
                missing.append(str(path))
                continue
            if unavailable_complete(path, run, snapnum):
                snapshot_rows = unavailable_rows(path)
            elif VALIDATION.cache_complete(
                path, (resolution, resolution, resolution_z)
            ):
                snapshot_rows = VALIDATION.summarize_cache(path)
                for row in snapshot_rows:
                    row["status"] = "ok"
                    row["status_reason"] = ""
                    row["snapshot_path"] = ""
            else:
                invalid.append(str(path))
                continue
            for row in snapshot_rows:
                classify(row)
                rows.append({key: scalar(value) for key, value in row.items()})
    if missing or invalid:
        raise ValueError(f"Missing {len(missing)} and invalid {len(invalid)} results")
    expected_rows = sum(len(DATAPATHS(run)[0]) for run in RUN_BY_MODE.values()) * len(
        STATISTICS
    )
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summary rows, found {len(rows)}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    destination = output_root / "nozzle_timescale_series.csv"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return rows


def rows_for(rows: list[dict], run: str, statistic: str) -> list[dict]:
    selected = [
        row for row in rows if row["run"] == run and row["statistic"] == statistic
    ]
    return sorted(selected, key=lambda row: float(row["time_tfb"]))


def series(items: list[dict], field: str):
    return np.asarray([float(row.get(field, np.nan)) for row in items])


def render_timescales(rows: list[dict], output_root: Path, statistic: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex="col")
    for column, run in enumerate(RUN_BY_MODE.values()):
        items = rows_for(rows, run, statistic)
        time = series(items, "time_tfb")
        ax = axes[0, column]
        for field, label in (
            ("tc_tdyn", r"$t_c$"),
            ("tv_tdyn", r"$t_v$"),
            ("tdiff_tdyn", r"$t_{\rm diff}$"),
            ("tesc_tdyn", r"$t_{\rm esc}$"),
        ):
            ax.plot(time, series(items, field), lw=1, label=label)
        ax.set_yscale("log")
        ax.set_title(run)
        ax.set_ylabel(r"timescale / $t_{\rm dyn,*}$")
        ax.legend(fontsize=8, ncol=2)
        ax = axes[1, column]
        for field, label in (
            ("tc_over_tv", r"$t_c/t_v$"),
            ("tdiff_over_tv", r"$t_{\rm diff}/t_v$"),
            ("tesc_over_tv", r"$t_{\rm esc}/t_v$"),
            ("effective_over_tv", r"$\max(t_c,t_{\rm esc})/t_v$"),
        ):
            ax.plot(time, series(items, field), lw=1, label=label)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"$t/t_{\rm fb}$")
        ax.set_ylabel("timescale ratio")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    destination = output_root / "figures" / f"timescales_{statistic}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def render_components(rows: list[dict], output_root: Path, statistic: str) -> None:
    fields = (
        ("tau_R", r"$\tau_R$", True),
        ("H_Rstar", r"$H/R_*$", True),
        ("vzbar_cm_s", r"$\langle |v_z|\rangle_\rho$ [cm s$^{-1}$]", True),
        ("sigma_g_cm2", r"$\Sigma$ [g cm$^{-2}$]", True),
    )
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 9), sharex="col")
    for column, run in enumerate(RUN_BY_MODE.values()):
        items = rows_for(rows, run, statistic)
        time = series(items, "time_tfb")
        for row_index, (field, label, logarithmic) in enumerate(fields):
            ax = axes[row_index, column]
            ax.plot(time, series(items, field), lw=1)
            if logarithmic:
                ax.set_yscale("log")
            ax.set_ylabel(label)
            if row_index == 0:
                ax.set_title(run)
            if row_index == len(fields) - 1:
                ax.set_xlabel(r"$t/t_{\rm fb}$")
    fig.tight_layout()
    destination = output_root / "figures" / f"components_{statistic}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def render_selection(rows: list[dict], output_root: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 7.5), sharex="col")
    for column, run in enumerate(RUN_BY_MODE.values()):
        items = rows_for(rows, run, "max_dissipation_pixel")
        time = series(items, "time_tfb")
        axes[0, column].plot(time, series(items, "selected_pixels"), lw=1)
        axes[0, column].set_ylabel("selected pixels")
        axes[0, column].set_title(run)
        axes[1, column].plot(
            time, series(items, "captured_total_dissipation_fraction"), lw=1
        )
        axes[1, column].set_ylabel("captured D fraction")
        x = series(items, "max_dissipation_x_rp")
        y = series(items, "max_dissipation_y_rp")
        axes[2, column].plot(time, np.hypot(x, y), lw=1, label=r"$R_{\rm peak}/r_p$")
        axes[2, column].axhline(0.6, color="k", ls=":", lw=0.8)
        axes[2, column].axhline(1.75, color="k", ls=":", lw=0.8)
        axes[2, column].set_ylabel(r"$R_{\rm peak}/r_p$")
        axes[2, column].set_xlabel(r"$t/t_{\rm fb}$")
    fig.tight_layout()
    destination = output_root / "figures" / "selection_quality.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run_aggregate(output_root: Path, resolution: int, resolution_z: int) -> None:
    rows = aggregate(output_root, resolution, resolution_z)
    for statistic in PLOTTED_STATISTICS:
        render_timescales(rows, output_root, statistic)
        render_components(rows, output_root, statistic)
    render_selection(rows, output_root)
    logger.info("Aggregated {} rows and rendered timeseries figures", len(rows))


def main(
    action: str = typer.Option("worker", help="worker or aggregate"),
    mode: int = typer.Option(1, min=1, max=3),
    snapshot_index: int = typer.Option(0, min=0),
    percentile: float | None = typer.Option(
        None, help="Deprecated compatibility option; wedge selection is used"
    ),
    resolution: int = typer.Option(256, min=8, help="x/y resolution"),
    resolution_z: int = typer.Option(512, min=8),
    workers: int = typer.Option(8, min=1),
    output_root: Path = typer.Option(  # noqa: B008 - Typer declares options in defaults.
        OUTPUT_ROOT
    ),
    overwrite: bool = typer.Option(False),
) -> None:
    if percentile is not None:
        logger.warning(
            "Ignoring --percentile={} because Stage 3 uses the accepted wedge",
            percentile,
        )
    if action == "worker":
        run_worker(
            mode,
            snapshot_index,
            resolution,
            resolution_z,
            workers,
            output_root,
            overwrite,
        )
    elif action == "aggregate":
        run_aggregate(output_root, resolution, resolution_z)
    else:
        raise typer.BadParameter("action must be worker or aggregate")


if __name__ == "__main__":
    typer.run(main)
