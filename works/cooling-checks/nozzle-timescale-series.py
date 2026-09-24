"""Calculate restartable nozzle cooling maps and assemble their time series.

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
Workers use linear x/y and sinh z sampling (scale ``0.1 r_p``), finding the native
peak direction in ``0.6 <= r/r_p <= 1.75`` (inner radius 0.8 for ``1e6``).

Input files
-----------
``--mode 1/2/3`` selects ``1e4/1e5/1e6``. ``dev.datapaths.DATAPATHS`` supplies
ordered ``snap_full_<n>.h5`` or ``snap_<n>.h5`` paths with restart exclusions.
A worker selects zero-based ``--snapshot-index`` (default 0), or exact
``--snapshot-number``. Required ``richio`` fields are time, X/Y/Z, rho, T, vz, sie
and dissipation. No prior direction cache is required. ``--action aggregate``
requires a valid worker NPZ for every registered snapshot of all three runs.

Output files
------------
Default ``--output-root`` is
``data/processed/CoolingChecks/nozzle-timescale-series/production-sinh``.
Workers write ``<output-root>/<run>/snap_<NNNN>_<Nx>x<Nx>x<Nz>_sinhz.npz``;
default grid-point counts are ``Nx=256`` and ``Nz=512`` (``Ny=Nx``).

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

A worker can instead write an unavailable-result NPZ for specifically handled
empty-region errors. This smaller archive has only ``run``, ``snapshot_path``
(Unicode scalars), ``snapnum`` (int64 scalar), ``time_tfb``, ``time_days``
(float64 scalars, fallback units and days), ``status`` (Unicode scalar,
``no_aperture_material``) and ``status_reason`` (Unicode scalar, error message).
All have shape ``()``; no map or wedge is present. Check ``status`` before
loading maps. Other calculation errors propagate rather than creating a marker.

Aggregation replaces ``nozzle_timescale_series.csv`` and seven PNGs under
``<output-root>``: ``figures/selection_quality.png`` plus
``figures/timescales_<statistic>.png`` and ``figures/components_<statistic>.png``
for the three statistics. PNGs are raster plots; load the CSV for numerical
values. The headered CSV has three rows per registered snapshot. Column order
is determined by first appearance, so use column names with ``csv.DictReader``.
All CSV cells load as strings; convert numeric fields explicitly. Valid rows
contain these summary fields:

The helper creates one row for each
``statistic``: ``median``, ``dissipation_weighted_mean`` and
``max_dissipation_pixel``. The median ignores NaNs; the weighted mean does not
filter NaNs or infinities. Weights are positive column dissipation in the wedge;
the peak pixel is the largest column dissipation there. Each row has:

``run``, ``z_spacing``, ``statistic`` : str
    Mass label, vertical grid spacing and aggregation method.
``snapnum``, ``resolution``, ``resolution_x``, ``resolution_y``, ``resolution_z`` : int
    Snapshot and sampling-grid sizes; ``resolution`` repeats ``resolution_x``.
``sinh_scale_rp``, ``time_tfb``, ``time_days`` : float
    Same units/meaning as the NPZ scalar metadata; linear-grid sinh scale is NaN.
``selected_pixels`` : int
    Number of True wedge pixels.
``captured_total_dissipation_fraction`` : float
    Wedge dissipation sum divided by the sum over all positive map pixels.
``max_dissipation_x_rp``, ``max_dissipation_y_rp`` : float
    Coordinates of the maximum-dissipation wedge pixel in pericentre units.
``sigma_g_cm2``, ``H_Rstar``, ``vzbar_cm_s``, ``tau_R`` : float
    Statistic of each map, in g/cm**2, stellar radii, cm/s and dimensionless
    optical depth, respectively.
``tc_tdyn``, ``tv_tdyn``, ``tdiff_tdyn``, ``tesc_tdyn`` : float
    Statistic of each dimensionless time map.
``tc_over_tv``, ``tdiff_over_tv``, ``tesc_over_tv``, ``effective_over_tv`` : float
    Statistic of each dimensionless ratio map. These are statistics of ratios,
    not ratios computed from separately aggregated times.

Additional CSV fields are:

``status``, ``status_reason``, ``snapshot_path`` : str
    Valid rows use ``ok``, empty reason and empty source path (read the path from
    their NPZ). Unavailable rows contain the marker status/reason/source path;
    their physical/grid/statistic-result fields are empty, not zeros.
``epoch_class`` : str
    ``unavailable`` if tau_R or tesc/tv is non-finite; otherwise
    ``optically_thin`` for tau_R < 1, ``optically_thick_escape_efficient`` for
    tau_R >= 1 and tesc/tv < 1, or ``photon_trapped`` for the remaining rows.
``emission_limited``, ``effectively_cooled`` : bool serialized as text, or empty
    ``True``/``False`` for tc/tv >= 1 and max(tc,tesc)/tv < 1 respectively;
    empty when the relevant ratio is non-finite. These are timescale labels.

Floating fields may contain ``nan`` or ``inf``; missing fields are empty CSV
cells. Older aggregate files may include extra fields, but those are not written
by the current aggregator.

Usage
-----
Run from ``/home/hey4/rich_tde`` in the richanalysis environment::

    python works/cooling-checks/nozzle-timescale-series.py --action worker --mode 1 --snapshot-number 108
    python works/cooling-checks/nozzle-timescale-series.py --action aggregate

Workers reuse accepted caches unless ``--overwrite``. Normal cache checks test
fields/grid shape, not source changes; unavailable markers are checked by
run/snapshot/status only. Aggregate rejects missing, invalid or unexpected NPZ
files. Keep experiments in separate output roots. The full production sequence
uses ``jobs/submit-nozzle-timescale-series-1e4.sh`` (and ``-1e5.sh``, ``-1e6.sh``),
then the aggregate job. A production snapshot can take substantial time/memory.
The deprecated ``--percentile`` option is ignored: the wedge is always used.

Loading examples
----------------
Inspect the result type, then select the finite wedge values::

    from pathlib import Path
    import numpy as np

    root = Path("data/processed/CoolingChecks/nozzle-timescale-series/production-sinh")
    with np.load(root / "1e4/snap_0108_256x256x512_sinhz.npz") as data:
        if "status" in data:
            print(data["status"].item(), data["status_reason"].item())
        else:
            ratio = data["effective_over_tv"]
            selected = data["wedge_mask"] & np.isfinite(ratio)
            print(data["time_tfb"].item(), np.median(ratio[selected]))
            # Maps have (x,y) order: pcolormesh(x_rp, y_rp, ratio.T).

Load one physical time series from the CSV, ignoring unavailable/undefined rows::

    import csv
    import dev
    import matplotlib.pyplot as plt

    with (root / "nozzle_timescale_series.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["run"] == "1e4" and row["statistic"] == "median"
                and row["status"] == "ok"]
    rows.sort(key=lambda row: float(row["time_tfb"]))
    time = np.array([float(row["time_tfb"]) for row in rows])
    ratio = np.array([float(row["effective_over_tv"]) for row in rows])
    finite = np.isfinite(ratio)
    fig, ax = plt.subplots()
    ax.plot(time[finite], ratio[finite])
    ax.set(xlabel="t/t_fb", ylabel="median max(tc,tesc)/tv")
    plt.show()
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev  # noqa: F401  # isort: skip  # Configure style before pyplot.
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
    mode: int = typer.Option(1, min=1, max=3, help="Worker run: 1=1e4, 2=1e5, 3=1e6"),
    snapshot_index: int = typer.Option(
        0, min=0, help="Zero-based index in DATAPATHS order"
    ),
    snapshot_number: int | None = typer.Option(
        None, help="Exact snapshot number, overriding index"
    ),
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
    """Compute one cached map, or aggregate all registered snapshots of all runs."""
    if percentile is not None:
        logger.warning(
            "Ignoring --percentile={} because Stage 3 uses the accepted wedge",
            percentile,
        )
    if action == "worker":
        if snapshot_number is not None:
            numbers, _ = DATAPATHS(RUN_BY_MODE[mode])
            snapshot_index = numbers.index(snapshot_number)
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
