#!/usr/bin/env python
"""Find persistent moving-mesh cell IDs and extract their thermodynamic histories.

``scan`` compares sorted IDs in successive selected snapshots. It measures every
contiguous ID lifetime in that selection, keeps bounded pools of long/variable
stellar candidates, and never allocates an array indexed by the numerical ID.
``extract`` searches every selected snapshot for explicit IDs and saves matched
rows. ``select`` ranks complete extracted histories by late thermal activity and
chooses a few distinct density-temperature paths. Persistent IDs follow mesh
cells, not exact Lagrangian fluid parcels.

Input files
-----------
Ranked RICH HDF5 snapshots from ``dev.DATAPATHS(run)`` (including its restart
exclusions and preference for ``snap_full`` files). ``--start-snapshot`` and
``--stop-snapshot`` are inclusive. ``--snapshot-step`` subsamples this catalogue;
its default is 1. A sparse scan establishes presence only at sampled snapshots,
not continuous survival between them. All IDs are read as uint64 integers.
``extract --ids-file`` reads a plain text file containing one decimal ID per
line, with optional ``#`` comments. IDs must be unique.
``select --input-dir`` reads completed ``extract`` outputs covering every
catalogue snapshot. It requires presence throughout, finite positive density,
temperature, pressure and specific internal energy, finite dissipation, stellar
fraction >= ``--min-star`` (0.99), abs(WasRemoved) <= ``--max-was-removed`` (1e-3),
and temperature < ``--max-temperature`` (1e9 K), at every saved sample.
Selection measures reheating, sampled normalized dissipation exposure, and warm
temperature after ``--late-fraction`` (0.1) of the elapsed time. It averages
their percentile ranks, shortlists the top 100 (or ``--keep`` if larger), and
greedily combines 70% activity score with 30% trajectory diversity. This is an
exploratory ranking, not a physical shock classification or resolved heating
budget. Sparse snapshot sampling cannot establish survival between saved times.

``census`` first reads only IDs to find every maximal presence interval lasting
at least ``--min-duration-tfb`` (default 1). It then reads candidate fields,
splits intervals at invalid samples, and keeps every valid stellar subinterval
that still meets that elapsed-time threshold. There is no ranking or top-K
limit. The physical, stellar, drain-tracer, and temperature filters match
``select`` and must hold at every retained sample. Snapshot gaps are rejected;
reappearing IDs start separate intervals. Fallback time uses
``pi/sqrt(2) * sqrt(Rstar**3/Mstar) * sqrt(Mbh/Mstar)`` in RICH's G=1 code units,
with stellar/BH parameters from ``dev.datapaths.TDE_PARAMETERS``.

``nozzle`` reads an existing census, first selects contiguous intervals by
elapsed lifetime, then examines only those histories. Near-nozzle samples have
``t > --after-tfb * tfb``, three-dimensional ``r < --radius-rp * rp`` and
BH-frame ``x > 0`` (the pericentre side, a geometric proxy, not a confirmed
shock). At least ``--min-points`` (2 or 3) samples must meet all three conditions;
they need not be consecutive. The default lifetime is 2 tfb and the default
late-time cut is 0.5 tfb. This currently supports only run 1e4, beta=1, and time
cuts excluding all pre-switch snapshots (before snapshot 21). It never joins
separate intervals or copies histories. All source-census physical filters and
its minimum lifetime remain in force. ``--count-only`` creates no output;
writes are refused above 5,000 qualifying intervals, so increase the lifetime
cut if needed. This geometric catalogue is not a shock classification.

Output files
------------
``metadata.json``
    JSON settings, absolute source paths and size/mtime signatures, completion
    status, and incremental progress. Existing results are skipped only when a
    completed manifest matches exactly; interrupted runs require ``--overwrite``.
    Runs do not resume from partial lifetime state.
``candidates.tsv`` (scan)
    Tab-separated table with a named header. Each row is one contiguous sampled
    ID interval. Columns: ``id`` (uint64); ``start_snapshot``, ``end_snapshot``,
    ``n_samples`` (integers); ``time_start_code``, ``time_end_code`` (code time);
    ``logrho_span``, ``logT_span`` (base-10 dex); ``max_dissipation_code`` (maximum
    positive volumetric dissipation in RICH code units); ``min_star``,
    ``max_star`` (dimensionless stellar mass fraction); ``max_was_removed``
    (maximum absolute dimensionless drain-history tracer);
    ``score=logT_span+0.5*logrho_span``.
    Temperature is in K; density is in RICH code units. Span differences are
    independent of the density unit. Eligibility requires finite positive
    density/temperature at every sample, max_star >= min_star, and the requested
    minimum sample count. The optional removed-tracer limit applies to its
    maximum over the entire interval (default limit 1e-3). Candidate pools prioritize lifetime or
    thermal score within minimum lengths 30, 60, 100, and the full selection.
    The table is bounded, not a census of all eligible intervals; exact counts
    and maximum observed lifetimes are recorded in metadata.
``history_ID.txt`` (extract)
    One whitespace-separated float64 table per requested uint64 ID, with a
    comment header. Load with ``np.loadtxt(..., ndmin=2)``. Columns (zero-based):
    0 snapshot number (integer-valued); 1 time [code_time]; 2 density
    [code density]; 3 temperature [K]; 4 volumetric dissipation [code units];
    5 stellar fraction; 6 WasRemoved tracer; 7/8/9 X/Y/Z [code_length];
    10 pressure [code pressure]; 11 specific internal energy [code specific energy].
    Coordinates use the snapshot's stored frame without any frame correction.
    Only present rows are written, chronologically; missing snapshots are not
    interpolated. Values are linear, unmasked, and may include nonfinite data.
``histories.npz`` (extract)
    ``ids`` uint64 (N,), ``snapshots`` int32 (S,), ``time_code`` float64 (S,),
    ``present`` bool (S,N), ``values`` float64 (S,N,10), and ``fields`` Unicode
    (10,). The last axis is Density, Temperature, Dissipation, tracers/Star,
    tracers/WasRemoved, X, Y, Z, Pressure, and InternalEnergy, with the same
    units/order as text columns 2..11. Missing rows are NaN. Duplicate ID
    matches raise an error.
``selected_ids.txt`` and ``histories/history_ID.txt`` (select)
    Decimal uint64 IDs in plotting order, one per line with ``#`` comments;
    load with ``np.loadtxt(..., dtype=np.uint64, ndmin=1)``. The text histories
    are unchanged copies of the extract tables documented above.
``candidates_summary.tsv`` (select)
    One row per fully eligible ID, sorted by decreasing score, then ID. Named
    columns: ``id`` (uint64); ``score`` (mean percentile, dimensionless);
    ``late_reheating_dex`` (largest log10 T increase above its prior late-window
    minimum); ``late_heating_exposure`` (dimensionless trapezoidal integral of
    max(Dissipation,0)/(Density*InternalEnergy) dt over the late window);
    ``late_temperature_p90_K`` (sampled late-window 90th percentile);
    ``late_warm_duration_days`` and ``late_warm_fraction`` (trapezoidal integral
    of indicator(T > 1100 K), in days and divided by the sampled late-window
    duration); ``logrho_span_dex``, ``logT_span_dex`` (whole-history log10 spans);
    ``min_temperature_K``, ``max_temperature_K`` (whole-history extrema);
    ``min_star``, ``max_abs_was_removed`` (whole-history dimensionless extrema).
    All metrics are finite float64; IDs remain exact integers.
``selection.json`` (select)
    JSON source signatures, thresholds, snapshot/time coverage, independent and
    cumulative filter counts, explicit score/diversity formulas and curve scales
    in dex, selected IDs and their metrics, and interpretation limits. Existing
    complete matching outputs are skipped; changed settings require overwrite.
``histories.h5`` (census)
    One compressed HDF5 archive, readable with ``h5py.File``. ``ids`` is uint64
    (K,), one ID per qualifying interval; IDs may repeat after a gap or invalid
    sample. ``offsets`` is int64 (K+1,), defining each interval's contiguous row
    slice. ``snapshot_index`` is int32 (R,), indexing the shared ``snapshots``
    int32 (S,) and ``time_code`` float64 (S,) catalogues. ``values`` is float64
    (R,10), with columns named by ``fields``, a fixed-width byte-string array
    (10,); order, units and stored-coordinate frame match ``histories.npz``.
    ``start_index`` and ``end_index`` are inclusive int32 (K,) catalogue indices.
    K is the number of qualifying intervals, R the total retained sample count,
    and S the full catalogue size. Values are linear, with no missing rows or
    interpolation within a retained interval. Positive rho/T/pressure/energy
    and finite dissipation/star/drain tracer are guaranteed; coordinates are
    copied without a separate finiteness filter. Empty results have K=R=0 and
    offsets=[0]. Attributes include ``schema_version`` (integer),
    ``fallback_time_code`` and ``min_duration_tfb`` (floats), and ``frame`` and
    ``units`` (descriptive strings).
``ids.txt`` (census)
    Sorted unique decimal uint64 IDs, one per line with a comment header; load
    with ``np.loadtxt(..., dtype=np.uint64, ndmin=1)``. No per-cell text files
    are created. ``metadata.json`` records exact filters, source signatures,
    candidate/final counts, fallback time, and per-snapshot progress. Temporary
    candidate HDF5 and target mmap files are removed after successful completion.

``ids.txt``, ``catalogue.tsv``, ``metadata.json`` (nozzle)
    Only these three small files are written. IDs are sorted unique uint64.
    The TSV has one row per qualifying source interval, ordered by descending
    duration, then ID. ``id`` is an exact uint64 integer; ``interval_index`` and
    ``row_start``/``row_stop`` are source-archive indices (stop exclusive).
    ``first_snapshot``, ``last_snapshot``, ``n_samples``, ``near_points`` and
    ``max_consecutive_near_points`` are integers. ``duration_tfb``,
    ``first_near_time_tfb``, ``last_near_time_tfb``,
    ``time_before_first_near_tfb`` and ``time_after_last_near_tfb`` use tfb;
    first/last near times are absolute simulation times, before/after are
    differences from the interval endpoints. ``min_late_radius_rp`` uses rp
    and covers all late samples, irrespective of x. Near-sample maxima are
    ``max_near_temperature_K`` [K] and
    ``max_positive_near_dissipation_code`` [code volumetric power, clipped at
    zero]. Metadata records criteria, source signature, units, counts before
    the point-count cut, and inherited physical filters. Use the stored row
    slice to inspect the original history without searching for its ID.

Usage
-----
Run from ``/home/hey4/rich_tde`` using the richanalysis environment::

    python works/shock-tde/trace-cell-histories.py scan --run 1e4 \
        --output-dir data/processed/ThermodynamicTracks/1e4/global --workers 8
    python works/shock-tde/trace-cell-histories.py scan --run 1e4 \
        --snapshot-step 151 --min-snapshots 2 --workers 8 \
        --output-dir data/processed/ThermodynamicTracks/1e4/endpoints
    python works/shock-tde/trace-cell-histories.py extract --run 1e4 \
        --ids-file data/processed/ThermodynamicTracks/1e4/selected_ids.txt \
        --output-dir data/processed/ThermodynamicTracks/1e4/histories --workers 8
    python works/shock-tde/trace-cell-histories.py select \
        --input-dir data/processed/ThermodynamicTracks/1e4/verification \
        --output-dir data/processed/ThermodynamicTracks/1e4 --keep 6
    python works/shock-tde/trace-cell-histories.py census --run 1e4 \
        --min-duration-tfb 1 --workers 24 \
        --output-dir data/processed/ThermodynamicTracks/1e4/one-tfb

    python works/shock-tde/trace-cell-histories.py nozzle --run 1e4 \
        --input-file data/processed/ThermodynamicTracks/1e4/one-tfb/histories.h5 \
        --min-duration-tfb 2 --after-tfb 0.5 --min-points 3 --count-only
    # Once the reported sample size is useful, replace --count-only with:
    # --output-dir data/processed/ThermodynamicTracks/1e4/nozzle

``--overwrite`` explicitly replaces same-named outputs. Use a distinct output
directory for changed settings. Process workers decompress separate rank chunks;
the parent bounds pending reads and assembles arrays in original rank order.

Loading examples
----------------
Load the compact nozzle catalogue with inferred column types::

    import os
    import h5py
    import numpy as np
    root = "data/processed/ThermodynamicTracks/1e4"
    catalogue = np.atleast_1d(np.genfromtxt(
        os.path.join(root, "nozzle", "catalogue.tsv"),
        names=True, dtype=None, encoding=None,
    ))
    with h5py.File(os.path.join(root, "one-tfb", "histories.h5")) as archive:
        row = catalogue[0]
        history = archive["values"][row["row_start"]:row["row_stop"]]
        print(int(row["id"]), history.shape)

The archive avoids rescanning the large raw snapshots in a notebook::

    import numpy as np
    import richio
    import unyt as u
    with np.load("data/processed/ThermodynamicTracks/1e4/histories/histories.npz") as f:
        ids, time, present = f["ids"], f["time_code"], f["present"]
        values = f["values"]
    keep = present[:, 0]
    rho = u.unyt_array(values[keep, 0, 0], richio.units.get_unit("Density"))
    temperature = u.unyt_array(values[keep, 0, 1], "K")
    print(int(ids[0]), rho.to("g/cm**3"), temperature)

Load the smaller selected histories directly::

    from pathlib import Path
    root = Path("data/processed/ThermodynamicTracks/1e4")
    ids = np.loadtxt(root / "selected_ids.txt", dtype=np.uint64, ndmin=1)
    history = np.loadtxt(root / "histories" / f"history_{int(ids[0])}.txt", ndmin=2)
    time = u.unyt_array(history[:, 1], richio.units.get_unit("Time")).to("day")
    rho = u.unyt_array(history[:, 2], richio.units.get_unit("Density")).to("g/cm**3")
    temperature = u.unyt_array(history[:, 3], "K")
    print(int(ids[0]), len(time), time[-1] - time[0])

Read one of the unranked, one-fallback-time intervals::

    import h5py
    with h5py.File(root / "one-tfb/histories.h5") as archive:
        start, stop = archive["offsets"][:2]
        rows = archive["values"][start:stop]
        indices = archive["snapshot_index"][start:stop]
        snapshot = archive["snapshots"][:][indices]
        time = u.unyt_array(archive["time_code"][:][indices], richio.units.tscale)
        density = u.unyt_array(rows[:, 0], richio.units.get_unit("Density"))
        temperature = u.unyt_array(rows[:, 1], "K")
        print(int(archive["ids"][0]), snapshot, density.to("g/cm**3"))
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
from numba import njit

SCAN_FIELDS = (
    "Density",
    "Temperature",
    "Dissipation",
    "tracers/Star",
    "tracers/WasRemoved",
)
EXTRACT_FIELDS = (*SCAN_FIELDS, "X", "Y", "Z", "Pressure", "InternalEnergy")
COLUMNS = (
    "id",
    "start_snapshot",
    "end_snapshot",
    "n_samples",
    "time_start_code",
    "time_end_code",
    "logrho_span",
    "logT_span",
    "max_dissipation_code",
    "min_star",
    "max_star",
    "max_was_removed",
    "score",
)
_WORKER_FILE = None


def _open_worker(path):
    global _WORKER_FILE
    if _WORKER_FILE is None or _WORKER_FILE.filename != path:
        if _WORKER_FILE is not None:
            _WORKER_FILE.close()
        _WORKER_FILE = h5py.File(path, "r")
    return _WORKER_FILE


def _read_rank(job):
    path, rank, targets, fields = job
    if isinstance(targets, tuple):
        target_path, target_count = targets
        targets = np.memmap(target_path, mode="r", dtype=np.uint64, shape=target_count)
    group = _open_worker(path)[rank]
    ids = np.asarray(group["ID"], dtype=np.uint64)
    if targets is None:
        values = (
            np.column_stack([group[field][()] for field in fields])
            if fields
            else np.empty((len(ids), 0))
        )
        return ids, values
    rows = np.flatnonzero((ids >= targets[0]) & (ids <= targets[-1]))
    positions = np.searchsorted(targets, ids[rows])
    rows = rows[targets[positions] == ids[rows]]
    if not len(rows):
        return np.empty(0, dtype=np.uint64), np.empty((0, len(fields)))
    # Large HDF5 point selections are expensive, especially for compressed ranks.
    # A dense selection is faster as one rank read followed by NumPy indexing.
    dense = len(rows) >= min(1024, max(1, len(ids) // 20))
    values = np.column_stack(
        [group[field][()][rows] if dense else group[field][rows] for field in fields]
    )
    return ids[rows], values


def _rank_layout(path):
    with h5py.File(path, "r") as source:
        ranks = sorted(
            (name for name in source if name.startswith("rank") and name[4:].isdigit()),
            key=lambda name: int(name[4:]),
        ) or ["/"]
        counts = [len(source[rank]["ID"]) for rank in ranks]
        time_code = float(np.asarray(source["Time"]).squeeze())
    return ranks, counts, time_code


def _rank_results(path, ranks, targets, executor, workers, fields=None):
    if fields is None:
        fields = SCAN_FIELDS if targets is None else EXTRACT_FIELDS
    jobs = iter(enumerate((str(path), rank, targets, fields) for rank in ranks))
    if executor is None:
        for index, job in jobs:
            yield index, _read_rank(job)
        return
    pending = {}
    for _ in range(2 * workers):
        item = next(jobs, None)
        if item is not None:
            index, job = item
            pending[executor.submit(_read_rank, job)] = index
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            index = pending.pop(future)
            yield index, future.result()
            item = next(jobs, None)
            if item is not None:
                next_index, job = item
                pending[executor.submit(_read_rank, job)] = next_index


def _load_snapshot(path, executor, workers, fields=SCAN_FIELDS):
    ranks, counts, time_code = _rank_layout(path)
    offsets = np.r_[0, np.cumsum(counts)]
    ids = np.empty(offsets[-1], dtype=np.uint64)
    values = np.empty((offsets[-1], len(fields)), dtype=np.float64)
    for index, (rank_ids, rank_values) in _rank_results(
        path, ranks, None, executor, workers, fields
    ):
        ids[offsets[index] : offsets[index + 1]] = rank_ids
        values[offsets[index] : offsets[index + 1]] = rank_values
    if not fields:
        ids.sort()
        if np.any(ids[1:] == ids[:-1]):
            raise ValueError(f"Duplicate IDs in {path}")
        return ids, values, time_code
    order = np.argsort(ids)
    ids = ids[order]
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids, values[order], time_code


@njit(cache=True)
def _advance(
    old_ids,
    old_start,
    old_stats,
    old_valid,
    ids,
    values,
    index,
    minimum,
    star_limit,
    removed_limit,
):
    """Merge sorted live IDs; return bounded-state summaries and eligible deaths."""
    size = len(ids)
    starts = np.empty(size, dtype=np.int32)
    stats = np.empty((size, 8), dtype=np.float64)
    valid = np.empty(size, dtype=np.bool_)
    dead = np.empty(len(old_ids), dtype=np.int64)
    n_dead = 0
    previous = 0
    max_any = 0
    max_stellar = 0
    max_finished_stellar = 0
    for row in range(size):
        while previous < len(old_ids) and old_ids[previous] < ids[row]:
            length = index - old_start[previous]
            if (
                old_valid[previous]
                and old_stats[previous, 6] >= star_limit
                and old_stats[previous, 7] <= removed_limit
            ):
                max_finished_stellar = max(max_finished_stellar, length)
                if length >= minimum:
                    dead[n_dead] = previous
                    n_dead += 1
            previous += 1
        rho, temperature, diss, star, removed = values[row]
        good = (
            np.isfinite(rho)
            and rho > 0
            and np.isfinite(temperature)
            and temperature > 0
        )
        logrho = np.log10(rho) if good else np.nan
        logtemp = np.log10(temperature) if good else np.nan
        starts[row] = index
        stats[row, 0] = logrho
        stats[row, 1] = logrho
        stats[row, 2] = logtemp
        stats[row, 3] = logtemp
        stats[row, 4] = max(0.0, diss) if np.isfinite(diss) else 0.0
        stats[row, 5] = star
        stats[row, 6] = star
        stats[row, 7] = abs(removed)
        valid[row] = good and np.isfinite(star) and np.isfinite(removed)
        if previous < len(old_ids) and old_ids[previous] == ids[row]:
            starts[row] = old_start[previous]
            valid[row] = valid[row] and old_valid[previous]
            stats[row, 0] = min(old_stats[previous, 0], logrho)
            stats[row, 1] = max(old_stats[previous, 1], logrho)
            stats[row, 2] = min(old_stats[previous, 2], logtemp)
            stats[row, 3] = max(old_stats[previous, 3], logtemp)
            stats[row, 4] = max(old_stats[previous, 4], stats[row, 4])
            stats[row, 5] = min(old_stats[previous, 5], star)
            stats[row, 6] = max(old_stats[previous, 6], star)
            stats[row, 7] = max(old_stats[previous, 7], abs(removed))
            previous += 1
        length = index - starts[row] + 1
        max_any = max(max_any, length)
        if (
            valid[row]
            and stats[row, 6] >= star_limit
            and stats[row, 7] <= removed_limit
        ):
            max_stellar = max(max_stellar, length)
    while previous < len(old_ids):
        length = index - old_start[previous]
        if (
            old_valid[previous]
            and old_stats[previous, 6] >= star_limit
            and old_stats[previous, 7] <= removed_limit
        ):
            max_finished_stellar = max(max_finished_stellar, length)
            if length >= minimum:
                dead[n_dead] = previous
                n_dead += 1
        previous += 1
    return (
        starts,
        stats,
        valid,
        dead[:n_dead],
        max_any,
        max_stellar,
        max_finished_stellar,
    )


def _record(ids, starts, stats, row, end, snapshots, times):
    start = int(starts[row])
    rho_span = float(stats[row, 1] - stats[row, 0])
    temp_span = float(stats[row, 3] - stats[row, 2])
    return dict(
        zip(
            COLUMNS,
            (
                int(ids[row]),
                snapshots[start],
                snapshots[end],
                end - start + 1,
                times[start],
                times[end],
                rho_span,
                temp_span,
                float(stats[row, 4]),
                float(stats[row, 5]),
                float(stats[row, 6]),
                float(stats[row, 7]),
                temp_span + 0.5 * rho_span,
            ),
        )
    )


class CandidatePools:
    def __init__(self, keep, minimum, total):
        self.keep = keep
        self.minimum = minimum
        self.limits = sorted(set((minimum, 30, 60, 100, total)))
        self.pools = {"longest": []} | {f"thermal_{limit}": [] for limit in self.limits}
        self.counts = {str(limit): 0 for limit in self.limits}
        self.total = 0

    def add(self, ids, starts, stats, rows, end, snapshots, times):
        if not len(rows):
            return
        lengths = end - starts[rows] + 1
        scores = (
            stats[rows, 3] - stats[rows, 2] + 0.5 * (stats[rows, 1] - stats[rows, 0])
        )
        diss = stats[rows, 4]
        self.total += len(rows)
        for limit in self.limits:
            self.counts[str(limit)] += int(np.count_nonzero(lengths >= limit))
        for name, existing in self.pools.items():
            eligible = (
                np.arange(len(rows))
                if name == "longest"
                else np.flatnonzero(lengths >= int(name.split("_")[1]))
            )
            if not len(eligible):
                continue
            primary, secondary = (
                (lengths, scores) if name == "longest" else (scores, lengths)
            )
            order = np.lexsort((diss[eligible], secondary[eligible], primary[eligible]))
            incoming = [
                _record(ids, starts, stats, rows[eligible[k]], end, snapshots, times)
                for k in order[-self.keep :]
            ]
            key = (
                (
                    lambda r: (
                        r["n_samples"],
                        r["score"],
                        r["max_dissipation_code"],
                        -r["id"],
                    )
                )
                if name == "longest"
                else (
                    lambda r: (
                        r["score"],
                        r["n_samples"],
                        r["max_dissipation_code"],
                        -r["id"],
                    )
                )
            )
            self.pools[name] = sorted(existing + incoming, key=key, reverse=True)[
                : self.keep
            ]

    def rows(self):
        unique = {
            (row["id"], row["start_snapshot"], row["end_snapshot"]): row
            for pool in self.pools.values()
            for row in pool
        }
        return sorted(
            unique.values(),
            key=lambda row: (-row["n_samples"], -row["score"], row["id"]),
        )


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_candidates(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _catalogue(args):
    from dev.datapaths import DATAPATHS

    paths = [
        path
        for path in DATAPATHS(args.run)
        if args.start_snapshot
        <= int(path.stem.rsplit("_", 1)[-1])
        <= args.stop_snapshot
    ][:: args.snapshot_step]
    if not paths:
        raise ValueError("No snapshots match the requested selection")
    snapshots = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
    return paths, snapshots


def _manifest(args, paths, snapshots, ids=None):
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in ("overwrite", "output_dir", "workers", "ids_file")
    }
    settings["sources"] = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    if ids is not None:
        settings["ids"] = [int(value) for value in ids]
        settings["ids_file"] = str(args.ids_file.resolve())
    metadata = {
        "schema_version": 1,
        "settings": settings,
        "status": "running",
        "snapshots": snapshots,
        "selection_is_every_catalogue_snapshot": args.snapshot_step == 1,
        "workers": args.workers,
        "progress": [],
    }
    target = args.output_dir / "metadata.json"
    if target.exists() and not args.overwrite:
        previous = json.loads(target.read_text())
        if (
            previous.get("settings") == settings
            and previous.get("status") == "complete"
        ):
            if args.command == "scan":
                expected = [args.output_dir / "candidates.tsv"]
            elif args.command == "census":
                expected = [
                    args.output_dir / name for name in ("histories.h5", "ids.txt")
                ]
            else:
                expected = [
                    args.output_dir / "histories.npz",
                    *(args.output_dir / f"history_{int(value)}.txt" for value in ids),
                ]
            if all(path.exists() for path in expected):
                print(
                    f"Matching completed results exist in {args.output_dir}; skipping",
                    flush=True,
                )
                return None
        raise FileExistsError(
            f"Existing/incomplete results differ; use --overwrite: {target}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target, metadata)
    return metadata


def scan(args, executor):
    paths, snapshots = _catalogue(args)
    metadata = _manifest(args, paths, snapshots)
    if metadata is None:
        return
    pools = CandidatePools(args.keep, args.min_snapshots, len(paths))
    old_ids = np.empty(0, dtype=np.uint64)
    starts = np.empty(0, dtype=np.int32)
    stats = np.empty((0, 8))
    valid = np.empty(0, dtype=bool)
    times = []
    maximum_any = maximum_stellar = 0
    removed_limit = np.inf if args.max_was_removed is None else args.max_was_removed
    started = time.monotonic()
    for index, path in enumerate(paths):
        tick = time.monotonic()
        ids, values, time_code = _load_snapshot(path, executor, args.workers)
        times.append(time_code)
        new_start, new_stats, new_valid, dead, max_any, max_prefix, max_finished = (
            _advance(
                old_ids,
                starts,
                stats,
                valid,
                ids,
                values,
                index,
                args.min_snapshots,
                args.min_star,
                removed_limit,
            )
        )
        del values
        pools.add(old_ids, starts, stats, dead, index - 1, snapshots, times)
        old_ids, starts, stats, valid = ids, new_start, new_stats, new_valid
        maximum_any = max(maximum_any, int(max_any))
        maximum_stellar = max(maximum_stellar, int(max_finished))
        progress = {
            "snapshot": snapshots[index],
            "n_cells": len(ids),
            "time_code": time_code,
            "seconds": time.monotonic() - tick,
            "max_any_samples_so_far": maximum_any,
            "max_eligible_completed_stellar_samples": maximum_stellar,
            "max_currently_eligible_stellar_prefix_samples": int(max_prefix),
            "eligible_completed_intervals": pools.total,
        }
        metadata["progress"].append(progress)
        _write_json(args.output_dir / "metadata.json", metadata)
        _write_candidates(args.output_dir / "candidates.tsv", pools.rows())
        print(json.dumps(progress), flush=True)
    end = len(paths) - 1
    stellar = valid & (stats[:, 6] >= args.min_star) & (stats[:, 7] <= removed_limit)
    if np.any(stellar):
        maximum_stellar = max(maximum_stellar, int(np.max(end - starts[stellar] + 1)))
    rows = np.flatnonzero((end - starts + 1 >= args.min_snapshots) & stellar)
    pools.add(old_ids, starts, stats, rows, end, snapshots, times)
    candidates = pools.rows()
    _write_candidates(args.output_dir / "candidates.tsv", candidates)
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "max_any_samples": maximum_any,
            "max_eligible_stellar_samples": maximum_stellar,
            "eligible_intervals": pools.total,
            "eligible_intervals_by_min_samples": pools.counts,
            "retained_candidates": len(candidates),
            "times_code": times,
        }
    )
    _write_json(args.output_dir / "metadata.json", metadata)
    print(
        f"Completed scan: {len(candidates)} candidates; output {args.output_dir}",
        flush=True,
    )


@njit(cache=True)
def _advance_presence(old_ids, old_start, ids, index, times, minimum_duration):
    """Merge sorted uint64 IDs and finish intervals using actual elapsed time."""
    starts = np.full(len(ids), index, dtype=np.int32)
    dead = np.empty(len(old_ids), dtype=np.int64)
    previous = 0
    n_dead = 0
    for row in range(len(ids)):
        while previous < len(old_ids) and old_ids[previous] < ids[row]:
            if times[index - 1] - times[old_start[previous]] >= minimum_duration:
                dead[n_dead] = previous
                n_dead += 1
            previous += 1
        if previous < len(old_ids) and old_ids[previous] == ids[row]:
            starts[row] = old_start[previous]
            previous += 1
    while previous < len(old_ids):
        if times[index - 1] - times[old_start[previous]] >= minimum_duration:
            dead[n_dead] = previous
            n_dead += 1
        previous += 1
    return starts, dead[:n_dead]


def _physical_rows(values, min_star, max_was_removed, max_temperature):
    positive = values[:, (0, 1, 8, 9)]
    return (
        (np.isfinite(positive) & (positive > 0)).all(axis=1)
        & np.isfinite(values[:, 2])
        & np.isfinite(values[:, 3])
        & (values[:, 3] >= min_star)
        & np.isfinite(values[:, 4])
        & (np.abs(values[:, 4]) <= max_was_removed)
        & (values[:, 1] < max_temperature)
    )


def _advance_valid(starts, good, index, times, minimum_duration):
    """Finish valid segments at missing/invalid samples, and begin new segments."""
    ended = np.flatnonzero((starts >= 0) & ~good)
    eligible = ended[times[index - 1] - times[starts[ended]] >= minimum_duration]
    intervals = np.column_stack(
        (eligible, starts[eligible], np.full(len(eligible), index - 1))
    ).astype(np.int64)
    starts[~good] = -1
    starts[(starts < 0) & good] = index
    return intervals


def _compact_census(source, path, ids, intervals, snapshots, times, tfb, minimum):
    """Compact candidate columns to contiguous interval records in bounded blocks."""
    if len(intervals):
        intervals = intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]
    lengths = intervals[:, 2] - intervals[:, 1] + 1
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    with h5py.File(path, "w") as target:
        target.attrs["schema_version"] = 1
        target.attrs["fallback_time_code"] = tfb
        target.attrs["min_duration_tfb"] = minimum
        target.attrs["frame"] = "Snapshot stored coordinates; no frame correction"
        target.attrs["units"] = (
            "Density:code density; Temperature:K; Dissipation:code volumetric power; "
            "Star/WasRemoved:dimensionless; XYZ:code length; "
            "Pressure:code pressure; InternalEnergy:code specific energy"
        )
        target.create_dataset("ids", data=ids[intervals[:, 0]])
        target.create_dataset("offsets", data=offsets)
        target.create_dataset("start_index", data=intervals[:, 1].astype(np.int32))
        target.create_dataset("end_index", data=intervals[:, 2].astype(np.int32))
        target.create_dataset("snapshots", data=np.asarray(snapshots, dtype=np.int32))
        target.create_dataset("time_code", data=times)
        target.create_dataset("fields", data=np.asarray(EXTRACT_FIELDS, dtype="S"))
        total = int(offsets[-1])
        if not total:
            target.create_dataset("values", shape=(0, len(EXTRACT_FIELDS)), dtype="f8")
            target.create_dataset("snapshot_index", shape=(0,), dtype="i4")
        else:
            data = target.create_dataset(
                "values",
                shape=(total, len(EXTRACT_FIELDS)),
                dtype="f8",
                chunks=(min(total, 4096), len(EXTRACT_FIELDS)),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            sample_index = target.create_dataset(
                "snapshot_index",
                shape=(total,),
                dtype="i4",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            begin = 0
            while begin < len(intervals):
                first = int(intervals[begin, 0])
                end = np.searchsorted(intervals[:, 0], first + 1024)
                last = int(intervals[end - 1, 0])
                block = source[:, first : last + 1, :]
                values = []
                indices = []
                for column, start, stop in intervals[begin:end]:
                    values.append(block[start : stop + 1, column - first])
                    indices.append(np.arange(start, stop + 1, dtype=np.int32))
                data[offsets[begin] : offsets[end]] = np.concatenate(values)
                sample_index[offsets[begin] : offsets[end]] = np.concatenate(indices)
                begin = end
    return intervals, offsets


def census(args, executor):
    """Find every sufficiently long valid stellar interval without a top-K pool."""
    from dev.datapaths import TDE_PARAMETERS
    import unyt as u

    import richio

    paths, snapshots = _catalogue(args)
    if args.snapshot_step != 1 or np.any(np.diff(snapshots) != 1):
        raise ValueError("Census requires every consecutive snapshot, with no gaps")
    layouts = [_rank_layout(path) for path in paths]
    times = np.asarray([layout[2] for layout in layouts])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Census times must be finite and strictly increasing")
    mbh, mstar, rstar = TDE_PARAMETERS[args.run]
    tfb = np.pi / np.sqrt(2) * np.sqrt(rstar**3 / mstar) * np.sqrt(mbh / mstar)
    duration = args.min_duration_tfb * tfb
    metadata = _manifest(args, paths, snapshots)
    if metadata is None:
        return
    started = time.monotonic()
    metadata.update(
        {
            "fallback_time_code": float(tfb),
            "fallback_time_days": float(
                u.unyt_quantity(tfb, richio.units.tscale).to_value("day")
            ),
            "min_duration_code": float(duration),
            "times_code": times.tolist(),
            "filter_definitions": {
                "duration": "time_end - time_start >= min_duration_tfb * fallback_time",
                "continuity": "Presence at every consecutive saved snapshot; gaps split intervals",
                "physical": "Finite positive Density, Temperature, Pressure, InternalEnergy; finite Dissipation",
                "stellar": "Finite Star >= min_star at every sample",
                "undrained": "Finite abs(WasRemoved) <= max_was_removed at every sample",
                "temperature_limit": "Temperature < max_temperature at every sample",
                "intervals": "Invalid rows split otherwise continuous ID histories; every qualifying segment retained",
            },
            "limitations": "Persistent mesh IDs are not conserved Lagrangian fluid parcels; "
            "continuity and shock passages are sampled only at saved times. No thermal "
            "ranking or top-K limit is applied.",
        }
    )
    old_ids = np.empty(0, dtype=np.uint64)
    starts = np.empty(0, dtype=np.int32)
    candidate_ids, candidate_starts, candidate_ends = [], [], []
    candidate_count = 0
    for index, path in enumerate(paths):
        tick = time.monotonic()
        ids, _, _ = _load_snapshot(path, executor, args.workers, fields=())
        new_starts, dead = _advance_presence(
            old_ids, starts, ids, index, times, duration
        )
        if len(dead):
            candidate_ids.append(old_ids[dead])
            candidate_starts.append(starts[dead])
            candidate_ends.append(np.full(len(dead), index - 1, dtype=np.int32))
            candidate_count += len(dead)
        old_ids, starts = ids, new_starts
        progress = {
            "stage": "presence",
            "snapshot": snapshots[index],
            "n_cells": len(ids),
            "finished_candidate_intervals": candidate_count,
            "seconds": time.monotonic() - tick,
        }
        metadata["progress"].append(progress)
        _write_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(progress), flush=True)
    rows = np.flatnonzero(times[-1] - times[starts] >= duration)
    candidate_ids.append(old_ids[rows])
    candidate_starts.append(starts[rows])
    candidate_ends.append(np.full(len(rows), len(times) - 1, dtype=np.int32))
    ids = np.concatenate(candidate_ids)
    interval_start = np.concatenate(candidate_starts)
    interval_end = np.concatenate(candidate_ends)
    order = np.lexsort((interval_start, ids))
    ids, interval_start, interval_end = (
        values[order] for values in (ids, interval_start, interval_end)
    )
    del old_ids, starts, candidate_ids, candidate_starts, candidate_ends, order
    metadata["candidate_intervals"] = len(ids)
    metadata["candidate_unique_ids"] = len(np.unique(ids))
    metadata["presence_elapsed_seconds"] = time.monotonic() - started
    print(f"Presence census complete: {len(ids)} candidate intervals", flush=True)
    _write_json(args.output_dir / "metadata.json", metadata)

    work_path = args.output_dir / ".census-candidates.h5.tmp"
    targets_path = args.output_dir / ".census-targets.u64.tmp"
    archive_path = args.output_dir / "histories.h5.tmp"
    valid_starts = np.full(len(ids), -1, dtype=np.int32)
    finished = []
    with h5py.File(work_path, "w") as work:
        if len(ids):
            stored = work.create_dataset(
                "values",
                shape=(len(paths), len(ids), len(EXTRACT_FIELDS)),
                dtype="f8",
                chunks=(1, min(1024, len(ids)), len(EXTRACT_FIELDS)),
                fillvalue=np.nan,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            for index, path in enumerate(paths):
                tick = time.monotonic()
                active = np.flatnonzero(
                    (interval_start <= index) & (interval_end >= index)
                )
                targets = ids[active]
                # One read-only mmap is shared by workers instead of pickling a
                # potentially million-ID target array once for each rank.
                targets.tofile(targets_path)
                values = np.full((len(ids), len(EXTRACT_FIELDS)), np.nan)
                seen = np.zeros(len(ids), dtype=bool)
                if len(active):
                    for _, (found_ids, found_values) in _rank_results(
                        path,
                        layouts[index][0],
                        (str(targets_path), len(targets)),
                        executor,
                        args.workers,
                    ):
                        columns = active[np.searchsorted(targets, found_ids)]
                        if len(np.unique(columns)) != len(columns) or np.any(
                            seen[columns]
                        ):
                            raise ValueError(f"Duplicate selected ID in {path}")
                        seen[columns] = True
                        values[columns] = found_values
                    if not seen[active].all():
                        raise ValueError(f"Presence/extraction disagree in {path}")
                good = seen & _physical_rows(
                    values, args.min_star, args.max_was_removed, args.max_temperature
                )
                ended = _advance_valid(valid_starts, good, index, times, duration)
                if len(ended):
                    finished.append(ended)
                values[~good] = np.nan
                # Leave empty chunks unallocated. The rest contain only valid
                # stellar samples; all rejected rows remain HDF5 fill values.
                chunks = np.unique(np.flatnonzero(good) // 1024)
                for chunk in chunks:
                    begin = int(chunk) * 1024
                    end = min(begin + 1024, len(ids))
                    stored[index, begin:end] = values[begin:end]
                progress = {
                    "stage": "physical",
                    "snapshot": snapshots[index],
                    "candidate_cells": len(active),
                    "valid_stellar_cells": int(good.sum()),
                    "seconds": time.monotonic() - tick,
                }
                metadata["progress"].append(progress)
                _write_json(args.output_dir / "metadata.json", metadata)
                print(json.dumps(progress), flush=True)
            ended = _advance_valid(
                valid_starts,
                np.zeros(len(ids), dtype=bool),
                len(paths),
                times,
                duration,
            )
            if len(ended):
                finished.append(ended)
        else:
            stored = work.create_dataset(
                "values", shape=(len(paths), 0, len(EXTRACT_FIELDS))
            )
        intervals = (
            np.concatenate(finished) if finished else np.empty((0, 3), dtype=np.int64)
        )
        intervals, offsets = _compact_census(
            stored,
            archive_path,
            ids,
            intervals,
            snapshots,
            times,
            tfb,
            args.min_duration_tfb,
        )
    archive_path.replace(args.output_dir / "histories.h5")
    work_path.unlink()
    targets_path.unlink(missing_ok=True)
    unique_ids = np.unique(ids[intervals[:, 0]])
    np.savetxt(
        args.output_dir / "ids.txt",
        unique_ids,
        fmt="%d",
        header="Every ID with a valid stellar interval lasting at least the requested fallback-time duration",
    )
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "qualifying_intervals": len(intervals),
            "qualifying_unique_ids": len(unique_ids),
            "samples": int(offsets[-1]),
        }
    )
    _write_json(args.output_dir / "metadata.json", metadata)
    print(
        f"Completed census: {len(intervals)} qualifying intervals, {len(unique_ids)} IDs; "
        f"output {args.output_dir}",
        flush=True,
    )


def extract(args, executor):
    paths, snapshots = _catalogue(args)
    ids = np.loadtxt(args.ids_file, dtype=np.uint64, ndmin=1)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("ID file must contain one unique uint64 ID per line")
    metadata = _manifest(args, paths, snapshots, ids)
    if metadata is None:
        return
    order = np.argsort(ids)
    targets = ids[order]
    data = np.full((len(paths), len(ids), len(EXTRACT_FIELDS)), np.nan)
    present = np.zeros((len(paths), len(ids)), dtype=bool)
    times = np.empty(len(paths))
    started = time.monotonic()
    for index, path in enumerate(paths):
        tick = time.monotonic()
        ranks, _, times[index] = _rank_layout(path)
        for _, (found_ids, values) in _rank_results(
            path, ranks, targets, executor, args.workers
        ):
            columns = order[np.searchsorted(targets, found_ids)]
            if len(np.unique(columns)) != len(columns) or np.any(
                present[index, columns]
            ):
                raise ValueError(f"Duplicate selected ID in {path}")
            data[index, columns] = values
            present[index, columns] = True
        progress = {
            "snapshot": snapshots[index],
            "found": int(present[index].sum()),
            "requested": len(ids),
            "seconds": time.monotonic() - tick,
        }
        metadata["progress"].append(progress)
        _write_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(progress), flush=True)
    header = (
        "snapshot time_code rho_code T_K dissipation_code star was_removed "
        "X_code Y_code Z_code pressure_code specific_internal_energy_code"
    )
    for column, particle_id in enumerate(ids):
        mask = present[:, column]
        values = np.column_stack(
            (np.asarray(snapshots)[mask], times[mask], data[mask, column])
        )
        np.savetxt(
            args.output_dir / f"history_{int(particle_id)}.txt",
            values,
            header=header,
            fmt=["%d"] + ["%.17g"] * (len(EXTRACT_FIELDS) + 1),
        )
    np.savez_compressed(
        args.output_dir / "histories.npz",
        ids=ids,
        snapshots=np.asarray(snapshots, dtype=np.int32),
        time_code=times,
        present=present,
        values=data,
        fields=np.asarray(EXTRACT_FIELDS),
    )
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "samples_per_id": {
                str(int(value)): int(present[:, i].sum()) for i, value in enumerate(ids)
            },
        }
    )
    _write_json(args.output_dir / "metadata.json", metadata)
    print(f"Completed extraction: {len(ids)} IDs; output {args.output_dir}", flush=True)


def select(args):
    """Rank complete histories by late thermal activity, then diversify their curves."""
    import unyt as u
    from scipy.stats import rankdata

    import richio

    source_metadata = args.input_dir / "metadata.json"
    archive_path = args.input_dir / "histories.npz"
    source = json.loads(source_metadata.read_text())
    if (
        source["status"] != "complete"
        or not source["selection_is_every_catalogue_snapshot"]
    ):
        raise ValueError(
            "Selection requires a completed extraction of every catalogue snapshot"
        )
    with np.load(archive_path) as archive:
        ids, snapshots = archive["ids"], archive["snapshots"]
        times, present, values = (
            archive["time_code"],
            archive["present"],
            archive["values"],
        )
        if tuple(archive["fields"]) != EXTRACT_FIELDS:
            raise ValueError("Unexpected history field order")
    if not np.array_equal(snapshots, source["snapshots"]):
        raise ValueError("Archive and extraction catalogue disagree")
    if (
        len(times) < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("History times must be finite and strictly increasing")
    if ids.dtype != np.uint64 or len(np.unique(ids)) != len(ids):
        raise ValueError("History IDs must be unique uint64 integers")

    positive = values[:, :, [0, 1, 8, 9]]
    masks = {
        "complete": present.all(axis=0),
        "physical": (np.isfinite(positive) & (positive > 0)).all(axis=(0, 2))
        & np.isfinite(values[:, :, 2]).all(axis=0),
        "stellar": (
            np.isfinite(values[:, :, 3]) & (values[:, :, 3] >= args.min_star)
        ).all(axis=0),
        "undrained": (
            np.isfinite(values[:, :, 4])
            & (np.abs(values[:, :, 4]) <= args.max_was_removed)
        ).all(axis=0),
        "temperature_limit": (values[:, :, 1] < args.max_temperature).all(axis=0),
    }
    eligible = np.logical_and.reduce(list(masks.values()))
    columns = np.flatnonzero(eligible)
    if not len(columns):
        raise ValueError("No complete histories satisfy the physical selection")
    late = times >= times[0] + args.late_fraction * (times[-1] - times[0])
    if np.count_nonzero(late) < 2:
        raise ValueError("The late-time window needs at least two samples")

    data = values[:, columns]
    time_quantity = u.unyt_array(times, richio.units.get_unit("Time"))
    density = u.unyt_array(data[:, :, 0], richio.units.get_unit("Density"))
    temperature = u.unyt_array(data[:, :, 1], "K")
    dissipation = u.unyt_array(data[:, :, 2], richio.units.get_unit("Dissipation"))
    energy = u.unyt_array(data[:, :, 9], richio.units.get_unit("InternalEnergy"))
    heating_rate = np.maximum(dissipation, 0) / (density * energy)
    exposure = np.trapezoid(heating_rate[late], x=time_quantity[late], axis=0).to_value(
        ""
    )
    log_density = np.log10(density.to_value("g/cm**3"))
    log_temperature = np.log10(temperature.to_value("K"))
    late_log_temperature = log_temperature[late]
    reheating = np.max(
        late_log_temperature - np.minimum.accumulate(late_log_temperature, axis=0),
        axis=0,
    )
    warm = u.unyt_array(
        (temperature[late] > u.unyt_quantity(1100, "K")).astype(float), "dimensionless"
    )
    warm_duration = np.trapezoid(warm, x=time_quantity[late], axis=0)
    late_temperature_p90 = np.percentile(temperature[late].to_value("K"), 90, axis=0)
    # Tied values receive the same percentile rank; the three activity terms have equal weight.
    scores = np.mean(
        [
            (rankdata(metric) - 0.5) / len(columns)
            for metric in (reheating, np.log1p(exposure), late_temperature_p90)
        ],
        axis=0,
    )
    metrics = {
        "score": scores,
        "late_reheating_dex": reheating,
        "late_heating_exposure": exposure,
        "late_temperature_p90_K": late_temperature_p90,
        "late_warm_duration_days": warm_duration.to_value("day"),
        "late_warm_fraction": (
            warm_duration / (time_quantity[late][-1] - time_quantity[late][0])
        ).to_value(""),
        "logrho_span_dex": np.ptp(log_density, axis=0),
        "logT_span_dex": np.ptp(log_temperature, axis=0),
        "min_temperature_K": temperature.min(axis=0).to_value("K"),
        "max_temperature_K": temperature.max(axis=0).to_value("K"),
        "min_star": data[:, :, 3].min(axis=0),
        "max_abs_was_removed": np.abs(data[:, :, 4]).max(axis=0),
    }
    if not all(np.isfinite(metric).all() for metric in metrics.values()):
        raise ValueError("Nonfinite selection metrics; inspect the extracted histories")
    order = np.lexsort((ids[columns], -scores))
    shortlist = order[: max(100, args.keep)]
    # Compare shapes and locations in the rho-T plane, using robust scales in dex.
    scales = np.array(
        [
            max(float(np.diff(np.percentile(curves, [10, 90]))[0]), 1e-12)
            for curves in (log_density, log_temperature)
        ]
    )
    curves = np.stack((log_density, log_temperature), axis=-1) / scales
    chosen = [int(shortlist[0])]
    distance = np.full(len(columns), np.inf)
    while len(chosen) < min(args.keep, len(shortlist)):
        new_distance = np.sqrt(
            np.mean((curves - curves[:, chosen[-1], None]) ** 2, axis=(0, 2))
        )
        distance = np.minimum(distance, new_distance)
        remaining = shortlist[~np.isin(shortlist, chosen)]
        utility = 0.7 * scores[remaining] + 0.3 * distance[remaining] / (
            1 + distance[remaining]
        )
        chosen.append(int(remaining[np.argmax(utility)]))

    rows = [
        {
            "id": int(ids[columns[column]]),
            **{key: float(value[column]) for key, value in metrics.items()},
        }
        for column in order
    ]
    rows_by_id = {row["id"]: row for row in rows}
    selected_ids = [int(ids[columns[column]]) for column in chosen]
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in ("input_dir", "output_dir", "overwrite")
    }
    settings["sources"] = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in (source_metadata, archive_path)
    ]
    destination = args.output_dir / "selection.json"
    if destination.exists() and not args.overwrite:
        previous = json.loads(destination.read_text())
        expected = [
            args.output_dir / "selected_ids.txt",
            args.output_dir / "candidates_summary.tsv",
            *(
                args.output_dir / "histories" / f"history_{value}.txt"
                for value in selected_ids
            ),
        ]
        if previous.get("settings") == settings and all(
            path.exists() for path in expected
        ):
            print(
                f"Matching selection exists in {args.output_dir}; skipping", flush=True
            )
            return
        raise FileExistsError(
            f"Existing selection differs; use --overwrite: {destination}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = args.output_dir / "histories"
    histories_dir.mkdir(exist_ok=True)
    for value in selected_ids:
        shutil.copy2(
            args.input_dir / f"history_{value}.txt",
            histories_dir / f"history_{value}.txt",
        )
    (args.output_dir / "selected_ids.txt").write_text(
        f"# Complete cell histories: snapshots {snapshots[0]}..{snapshots[-1]}, {len(snapshots)} samples\n"
        "# Ranked by late thermal activity, with diversity in density-temperature paths.\n"
        "# See selection.json for physical filters, scores, and source provenance.\n"
        + "".join(f"{value}\n" for value in selected_ids)
    )
    with (args.output_dir / "candidates_summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    cumulative = np.ones(len(ids), dtype=bool)
    filter_counts = {"input_ids": len(ids)}
    for name, mask in masks.items():
        cumulative &= mask
        filter_counts[name] = {
            "passing_independently": int(mask.sum()),
            "passing_cumulatively": int(cumulative.sum()),
        }
    _write_json(
        destination,
        {
            "schema_version": 1,
            "status": "complete",
            "settings": settings,
            "source_metadata": str(source_metadata.resolve()),
            "snapshots": snapshots.tolist(),
            "time_start_code": float(times[0]),
            "time_end_code": float(times[-1]),
            "duration_days": float(
                (time_quantity[-1] - time_quantity[0]).to_value("day")
            ),
            "late_start_snapshot": int(snapshots[late][0]),
            "late_start_time_code": float(times[late][0]),
            "filter_counts": filter_counts,
            "filter_definitions": {
                "complete": "ID is present at every saved catalogue snapshot",
                "physical": "Density, Temperature, Pressure and InternalEnergy are finite and "
                "positive, and Dissipation is finite, at every sample",
                "stellar": "finite Star >= min_star at every sample",
                "undrained": "finite abs(WasRemoved) <= max_was_removed at every sample",
                "temperature_limit": "Temperature < max_temperature at every sample",
            },
            "eligible_ids": len(columns),
            "selection_method": {
                "activity_score": "mean percentile rank of late reheating, log1p(exposure), and T90",
                "reheating": "max(log10(T) - running_min(log10(T))) within the late window",
                "exposure": "trapezoidal integral of max(Dissipation,0)/(Density*InternalEnergy) dt; "
                "a sampled ranking proxy, not a resolved energy budget",
                "warm_duration": "trapezoidal integral of indicator(T > 1100 K) in the late window",
                "shortlist_size": len(shortlist),
                "seed": "highest activity score",
                "diversity": "minimum RMS distance in (log10 rho, log10 T) from chosen full curves, "
                "scaled by the eligible ensemble's 10th-to-90th percentile spans",
                "curve_scales_dex": scales.tolist(),
                "subsequent_utility": "0.7*activity_score + 0.3*distance/(1+distance)",
            },
            "selected": [rows_by_id[value] for value in selected_ids],
            "limitations": "Continuous presence is verified only at saved catalogue snapshots. "
            "Persistent mesh-cell IDs do not identify conserved Lagrangian material parcels.",
        },
    )
    print(
        f"Selected {len(selected_ids)} of {len(columns)} eligible complete histories: {selected_ids}",
        flush=True,
    )


NOZZLE_COLUMNS = (
    "id",
    "interval_index",
    "row_start",
    "row_stop",
    "first_snapshot",
    "last_snapshot",
    "n_samples",
    "duration_tfb",
    "near_points",
    "max_consecutive_near_points",
    "first_near_time_tfb",
    "last_near_time_tfb",
    "time_before_first_near_tfb",
    "time_after_last_near_tfb",
    "min_late_radius_rp",
    "max_near_temperature_K",
    "max_positive_near_dissipation_code",
)


def nozzle(args):
    """Write a small index into census histories passing a late nozzle-region cut."""
    from dev.datapaths import TDE_PARAMETERS
    import unyt as u

    import richio

    if args.run != "1e4":
        raise ValueError("Nozzle selection currently supports only run 1e4")
    for name in ("min_duration_tfb", "radius_rp"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.after_tfb) or args.after_tfb < 0:
        raise ValueError("after_tfb must be finite and nonnegative")
    if args.min_points not in (2, 3):
        raise ValueError("min_points must be 2 or 3")
    if not args.count_only and not args.output_dir:
        raise ValueError("--output-dir is required unless --count-only is used")
    input_file = os.path.realpath(args.input_file)
    source_meta_path = os.path.join(os.path.dirname(input_file), "metadata.json")
    with open(source_meta_path) as stream:
        source_meta = json.load(stream)
    source_settings = source_meta.get("settings", {})
    if (
        source_meta.get("status") != "complete"
        or source_settings.get("command") != "census"
        or source_settings.get("run") != args.run
    ):
        raise ValueError(
            "Source metadata must describe a completed census for this run"
        )
    if not source_meta.get("filter_definitions"):
        raise ValueError("Source metadata must document the inherited census filters")
    output_paths = {}
    if not args.count_only:
        output_dir = os.path.realpath(args.output_dir)
        if output_dir == os.path.dirname(input_file):
            raise ValueError(
                "Use a separate output directory; preserve the source census"
            )
        output_paths = {
            name: os.path.join(output_dir, name)
            for name in ("ids.txt", "catalogue.tsv", "metadata.json")
        }
        if not args.overwrite:
            for path in output_paths.values():
                if os.path.exists(path):
                    raise FileExistsError(
                        f"Existing output requires --overwrite: {path}"
                    )

    mbh, mstar, rstar = TDE_PARAMETERS[args.run]
    expected_tfb = np.pi / np.sqrt(2) * np.sqrt(rstar**3 / mstar) * np.sqrt(mbh / mstar)
    rp = rstar * (mbh / mstar) ** (1 / 3)  # beta = 1 for this run.
    source_stat = os.stat(input_file)
    started = time.monotonic()
    selected_rows = []
    at_least_two = at_least_three = selected_count = 0
    with h5py.File(input_file, "r", rdcc_nbytes=32 * 1024**2) as archive:
        if archive.attrs.get("schema_version") != 1:
            raise ValueError("Unsupported census schema_version; expected 1")
        tfb = float(archive.attrs["fallback_time_code"])
        source_floor = float(archive.attrs["min_duration_tfb"])
        if not np.isfinite(tfb) or not np.isclose(tfb, expected_tfb, rtol=1e-10):
            raise ValueError(
                "Source fallback time disagrees with this run's parameters"
            )
        if not np.isfinite(source_floor) or source_floor <= 0:
            raise ValueError("Source minimum duration must be finite and positive")
        if args.min_duration_tfb < source_floor:
            raise ValueError(
                f"Requested lifetime is below the source archive floor of {source_floor:g} tfb"
            )
        if source_settings.get("min_duration_tfb") != source_floor:
            raise ValueError(
                "Source metadata and archive disagree about minimum duration"
            )
        fields = archive["fields"].asstr()[:].tolist()
        if len(fields) != len(EXTRACT_FIELDS) or set(fields) != set(EXTRACT_FIELDS):
            raise ValueError("Source census fields do not match the documented schema")
        field = {name: index for index, name in enumerate(fields)}
        snapshots = archive["snapshots"][:]
        times = archive["time_code"][:]
        if (
            snapshots.ndim != 1
            or len(snapshots) < 2
            or times.shape != snapshots.shape
            or not np.issubdtype(snapshots.dtype, np.integer)
            or np.any(np.diff(snapshots) != 1)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "Source snapshot/time catalogue must be consecutive and increasing"
            )
        late_catalogue = times > args.after_tfb * tfb
        if np.any(late_catalogue & (snapshots < 21)):
            raise ValueError(
                "This time cut includes snapshots before BH-frame switch 21; "
                "increase --after-tfb (no earlier frame correction is implemented)"
            )
        n_intervals = len(archive["ids"])
        n_rows = archive["values"].shape[0]
        if (
            archive["ids"].shape != (n_intervals,)
            or archive["ids"].dtype != np.dtype("uint64")
            or archive["values"].shape != (n_rows, len(fields))
            or archive["snapshot_index"].shape != (n_rows,)
            or archive["offsets"].shape != (n_intervals + 1,)
            or archive["start_index"].shape != (n_intervals,)
            or archive["end_index"].shape != (n_intervals,)
            or archive["offsets"][0] != 0
            or archive["offsets"][-1] != n_rows
        ):
            raise ValueError("Invalid census array shapes, ID dtype or offsets")

        # These two integer metadata arrays are small compared with the histories.
        starts = archive["start_index"][:]
        ends = archive["end_index"][:]
        if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= len(times)):
            raise ValueError("Source interval endpoints are out of bounds")
        candidates = []
        for begin in range(0, n_intervals, 1_000_000):
            stop = min(begin + 1_000_000, n_intervals)
            duration = times[ends[begin:stop]] - times[starts[begin:stop]]
            candidates.append(
                np.flatnonzero(duration >= args.min_duration_tfb * tfb) + begin
            )
        candidates = (
            np.concatenate(candidates) if candidates else np.empty(0, dtype=int)
        )
        print(
            f"Lifetime cut: {len(candidates):,}/{n_intervals:,} intervals >= "
            f"{args.min_duration_tfb:g} tfb; inspecting only these histories",
            flush=True,
        )
        offset_dataset = archive["offsets"]
        index_dataset = archive["snapshot_index"]
        value_dataset = archive["values"]
        id_dataset = archive["ids"]
        for number, interval in enumerate(candidates, 1):
            first, last = int(starts[interval]), int(ends[interval])
            row_start, row_stop = map(int, offset_dataset[interval : interval + 2])
            if not 0 <= row_start < row_stop <= n_rows:
                raise ValueError(f"Invalid row bounds for source interval {interval}")
            indices = index_dataset[row_start:row_stop]
            if not np.array_equal(indices, np.arange(first, last + 1)):
                raise ValueError(
                    f"Source interval {interval} is not consecutive; refusing to join gaps"
                )
            late = late_catalogue[indices]
            if np.count_nonzero(late) < 2:
                continue
            values = value_dataset[row_start:row_stop]
            xyz = values[:, [field["X"], field["Y"], field["Z"]]]
            radius = np.linalg.norm(xyz, axis=1)
            near = late & (radius < args.radius_rp * rp) & (xyz[:, 0] > 0)
            near_points = int(np.count_nonzero(near))
            at_least_two += near_points >= 2
            at_least_three += near_points >= 3
            if near_points >= args.min_points:
                selected_count += 1
                if len(selected_rows) < 5000:
                    near_indices = np.flatnonzero(near)
                    near_times = times[indices[near]] / tfb
                    sample_times = times[indices] / tfb
                    # Runs are counted along consecutive saved snapshots, not ID matches.
                    breaks = np.flatnonzero(np.diff(near_indices) > 1) + 1
                    consecutive = max(map(len, np.split(near_indices, breaks)))
                    finite_late_radii = radius[late & np.isfinite(radius)]
                    selected_rows.append(
                        dict(
                            zip(
                                NOZZLE_COLUMNS,
                                (
                                    int(id_dataset[interval]),
                                    int(interval),
                                    row_start,
                                    row_stop,
                                    int(snapshots[first]),
                                    int(snapshots[last]),
                                    len(indices),
                                    float(sample_times[-1] - sample_times[0]),
                                    near_points,
                                    consecutive,
                                    float(near_times[0]),
                                    float(near_times[-1]),
                                    float(near_times[0] - sample_times[0]),
                                    float(sample_times[-1] - near_times[-1]),
                                    float(np.min(finite_late_radii) / rp),
                                    float(np.max(values[near, field["Temperature"]])),
                                    float(
                                        max(
                                            0,
                                            np.max(values[near, field["Dissipation"]]),
                                        )
                                    ),
                                ),
                                strict=True,
                            )
                        )
                    )
            if number % 10000 == 0:
                print(
                    f"Inspected {number:,}/{len(candidates):,}; selected {selected_count:,}",
                    flush=True,
                )

    print(
        f"Late near-nozzle counts: >=2 points: {at_least_two:,}; "
        f">=3 points: {at_least_three:,}; selected: {selected_count:,} intervals",
        flush=True,
    )
    if args.count_only:
        return
    if selected_count > 5000:
        raise ValueError(
            f"{selected_count:,} intervals exceed the 5,000-row output limit; "
            "increase --min-duration-tfb and use --count-only first. No files written."
        )
    selected_rows.sort(
        key=lambda row: (-row["duration_tfb"], row["id"], row["interval_index"])
    )
    unique_ids = np.unique(
        np.asarray([row["id"] for row in selected_rows], dtype=np.uint64)
    )
    current_stat = os.stat(input_file)
    if (current_stat.st_size, current_stat.st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise ValueError("Source archive changed during selection; no files written")
    metadata = {
        "schema_version": 1,
        "command": "nozzle",
        "status": "complete",
        "run": args.run,
        "source": {
            "path": input_file,
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "metadata_path": source_meta_path,
            "min_duration_tfb": source_floor,
        },
        "criteria": {
            "min_duration_tfb": args.min_duration_tfb,
            "duration_comparison": ">=",
            "after_tfb": args.after_tfb,
            "time_comparison": "> (absolute simulation time)",
            "radius_rp": args.radius_rp,
            "radius_comparison": "sqrt(X**2 + Y**2 + Z**2) < radius_rp * rp",
            "side": "X > 0 in BH frame (pericentre side)",
            "min_points": args.min_points,
            "point_comparison": ">=, not necessarily consecutive",
            "all_sample_conditions_simultaneous": True,
            "separate_intervals_are_never_joined": True,
            "frame_switch_snapshot": 21,
        },
        "parameters": {"mbh": mbh, "mstar": mstar, "rstar": rstar, "beta": 1},
        "rp_code_length": rp,
        "rp_cm": float(u.unyt_quantity(rp, richio.units.lscale).to_value("cm")),
        "fallback_time_code": tfb,
        "fallback_time_days": float(
            u.unyt_quantity(tfb, richio.units.tscale).to_value("day")
        ),
        "code_units_cgs": {
            "length_cm": float(u.unyt_quantity(1, richio.units.lscale).to_value("cm")),
            "time_s": float(u.unyt_quantity(1, richio.units.tscale).to_value("s")),
            "dissipation_erg_cm3_s": float(
                u.unyt_quantity(1, richio.units.get_unit("Dissipation")).to_value(
                    "erg/cm**3/s"
                )
            ),
        },
        "inherited_physical_filters": {
            "min_star": source_settings.get("min_star"),
            "max_was_removed": source_settings.get("max_was_removed"),
            "max_temperature": source_settings.get("max_temperature"),
            "definitions": source_meta["filter_definitions"],
        },
        "counts": {
            "source_intervals": n_intervals,
            "lifetime_eligible_intervals": len(candidates),
            "near_points_ge_2": at_least_two,
            "near_points_ge_3": at_least_three,
            "selected_intervals": selected_count,
            "selected_unique_ids": len(unique_ids),
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": (
            "Geometric nozzle-region candidates, not confirmed shocks. Persistent mesh "
            "IDs are not conserved Lagrangian fluid parcels. Samples resolve only saved "
            "times; two or three near samples need not belong to one passage. Source "
            "census filters and minimum lifetime exclude other cells. No histories copied."
        ),
    }
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        output_paths["ids.txt"],
        unique_ids,
        fmt="%d",
        header="Unique nozzle-region candidate IDs; see catalogue.tsv and metadata.json",
    )
    with open(output_paths["catalogue.tsv"], "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=NOZZLE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)
    with open(output_paths["metadata.json"], "w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(
        f"Wrote {selected_count:,} intervals ({len(unique_ids):,} unique IDs) to {output_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("scan", "extract", "census"):
        child = commands.add_parser(command)
        child.add_argument("--run", choices=("1e4", "1e5", "1e6"), default="1e4")
        child.add_argument("--start-snapshot", type=int, default=0)
        child.add_argument("--stop-snapshot", type=int, default=151)
        child.add_argument("--snapshot-step", type=int, default=1)
        child.add_argument("--workers", type=int, default=1)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--overwrite", action="store_true")
        if command == "scan":
            child.add_argument("--min-snapshots", type=int, default=30)
            child.add_argument(
                "--min-star",
                type=float,
                default=0.99,
                help="Minimum maximum stellar fraction over each lifetime",
            )
            child.add_argument(
                "--max-was-removed",
                "--max-removed",
                type=float,
                default=1e-3,
                help="Upper limit on lifetime max(abs(WasRemoved)); default 1e-3",
            )
            child.add_argument(
                "--keep",
                type=int,
                default=100,
                help="Maximum candidates retained in each lifetime/thermal pool",
            )
        elif command == "extract":
            child.add_argument("--ids-file", type=Path, required=True)
        else:
            child.add_argument(
                "--min-duration-tfb",
                type=float,
                default=1.0,
                help="Minimum elapsed time per contiguous valid interval in fallback times",
            )
            child.add_argument("--min-star", type=float, default=0.99)
            child.add_argument("--max-was-removed", type=float, default=1e-3)
            child.add_argument("--max-temperature", type=float, default=1e9)
    child = commands.add_parser(
        "select", help="Select interesting complete extracted histories"
    )
    child.add_argument("--input-dir", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child.add_argument("--keep", type=int, default=6)
    child.add_argument("--min-star", type=float, default=0.99)
    child.add_argument("--max-was-removed", type=float, default=1e-3)
    child.add_argument(
        "--max-temperature", type=float, default=1e9, help="Strict upper limit in K"
    )
    child.add_argument(
        "--late-fraction",
        type=float,
        default=0.1,
        help="Fraction of the total elapsed time excluded from activity ranking",
    )
    child.add_argument("--overwrite", action="store_true")
    child = commands.add_parser(
        "nozzle", help="Index late nozzle-region histories in an existing census"
    )
    child.add_argument("--input-file", required=True)
    child.add_argument("--output-dir")
    child.add_argument("--run", choices=("1e4",), default="1e4")
    child.add_argument("--min-duration-tfb", type=float, default=2.0)
    child.add_argument("--after-tfb", type=float, default=0.5)
    child.add_argument("--radius-rp", type=float, default=2.0)
    child.add_argument("--min-points", type=int, choices=(2, 3), default=3)
    child.add_argument("--count-only", action="store_true")
    child.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "nozzle":
        if not args.count_only and not args.output_dir:
            parser.error("--output-dir is required unless --count-only is used")
        nozzle(args)
        return
    if args.command == "select":
        if args.keep < 1 or not 0 <= args.late_fraction < 1:
            parser.error("--keep must be positive and --late-fraction must be in [0,1)")
        select(args)
        return
    if args.workers < 1 or args.snapshot_step < 1:
        parser.error("--workers and --snapshot-step must be positive")
    if args.command == "scan" and (args.min_snapshots < 1 or args.keep < 1):
        parser.error("--min-snapshots and --keep must be positive")
    if args.command == "census" and (
        not np.isfinite(args.min_duration_tfb)
        or args.min_duration_tfb <= 0
        or not 0 <= args.min_star <= 1
        or not np.isfinite(args.max_was_removed)
        or args.max_was_removed < 0
        or not np.isfinite(args.max_temperature)
        or args.max_temperature <= 0
    ):
        parser.error(
            "Census needs a positive finite duration, min-star in [0,1], and a finite nonnegative removed limit"
        )
    if args.start_snapshot > args.stop_snapshot:
        parser.error("--start-snapshot must not exceed --stop-snapshot")
    if args.workers == 1:
        globals()[args.command](args, None)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            globals()[args.command](args, executor)


if __name__ == "__main__":
    main()
