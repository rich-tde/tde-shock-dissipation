#!/usr/bin/env python
"""Find persistent moving-mesh cell IDs and extract their thermodynamic histories.

Persistent IDs follow mesh cells, not conserved Lagrangian fluid parcels, and a
history is only sampled at saved snapshots.

Subcommands
-----------
``census``
    Every run of consecutive snapshots in which an ID is present and valid,
    lasting at least ``--min-duration-tfb`` fallback times. No ranking or top-K.
``extract``
    Full histories of the IDs in ``--ids-file`` (one decimal ID per line).
``select``
    Ranks complete, valid ``extract`` histories by late thermal activity
    (reheating, heating exposure and 90th-percentile temperature after
    ``--late-fraction`` of the elapsed time) and picks ``--keep`` diverse
    density-temperature paths. Exploratory, not a shock classification.
``nozzle``
    Indexes ``census`` intervals lasting ``--min-duration-tfb`` with at least
    ``--min-points`` samples at ``t > --after-tfb t_fb``, ``r < --radius-rp r_p``
    and ``x > 0``. A geometric proxy, not a confirmed shock. The time cut must
    exclude star-frame snapshots, because coordinates are stored uncorrected.

A sample is valid when density, temperature, pressure and internal energy are
finite and positive, dissipation is finite, ``Star >= --min-star`` (0.99),
``|WasRemoved| <= --max-was-removed`` (1e-3) and ``T < --max-temperature`` (1e9 K).
Snapshots come from ``dev.DATAPATHS(run)``; ``--start/--stop-snapshot`` are inclusive.

Outputs
-------
Each command writes ``metadata.json`` with its settings, source signatures and
progress. A completed run with identical settings is skipped; anything else at
the destination needs ``--overwrite``. Fields always come in the order of
``FIELDS``: Density [code], Temperature [K], Dissipation [code volumetric power],
Star, WasRemoved, X, Y, Z [code length, snapshot frame], Pressure [code],
InternalEnergy [code specific energy].

``histories.h5``, ``ids.txt`` (census)
    ``ids`` (K,) uint64, one per interval (an ID may repeat). Interval k is
    ``values[offsets[k]:offsets[k+1]]`` (rows, 10), sampled at catalogue indices
    ``snapshot_index`` into ``snapshots`` and ``time_code``; ``start_index`` and
    ``end_index`` (K,) are inclusive catalogue indices.
``history_ID.txt``, ``histories.npz`` (extract)
    Text columns: snapshot, time [code], then the ten fields. The npz holds
    ``ids``, ``snapshots``, ``time_code``, ``present`` (S, N) and ``values``
    (S, N, 10), NaN where absent.
``selected_ids.txt``, ``histories/``, ``candidates_summary.tsv``, ``selection.json`` (select)
    Chosen IDs in plotting order, copies of their histories, metrics of every
    eligible ID, and the filters, formulas and counts behind the choice.
``ids.txt``, ``catalogue.tsv`` (nozzle)
    One row per interval, sorted by decreasing duration; ``row_start`` and
    ``row_stop`` slice the census ``values``. Times in t_fb, radii in r_p.

Usage
-----
::

    python works/shock-tde/trace-cell-histories.py census --run 1e4 \\
        --min-duration-tfb 1 --workers 24 \\
        --output-dir data/processed/ThermodynamicTracks/1e4/one-tfb
    python works/shock-tde/trace-cell-histories.py extract --run 1e4 --workers 8 \\
        --ids-file data/processed/ThermodynamicTracks/1e4/selected_ids.txt \\
        --output-dir data/processed/ThermodynamicTracks/1e4/histories
    python works/shock-tde/trace-cell-histories.py select --keep 6 \\
        --input-dir data/processed/ThermodynamicTracks/1e4/verification \\
        --output-dir data/processed/ThermodynamicTracks/1e4
    python works/shock-tde/trace-cell-histories.py nozzle --run 1e4 \\
        --input-file data/processed/ThermodynamicTracks/1e4/one-tfb/histories.h5 \\
        --min-duration-tfb 1.5 --after-tfb 0.5 --radius-rp 2 --min-points 2 \\
        --output-dir data/processed/ThermodynamicTracks/1e4/nozzle

Loading example
---------------
::

    root = "data/processed/ThermodynamicTracks/1e4"
    catalogue = np.genfromtxt(f"{root}/nozzle/catalogue.tsv", names=True, dtype=None, encoding=None)
    with h5py.File(f"{root}/one-tfb/histories.h5") as archive:
        row = catalogue[0]
        history = archive["values"][row["row_start"] : row["row_stop"]]
"""

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

import h5py
import numpy as np
from numba import njit

FIELDS = (
    "Density",
    "Temperature",
    "Dissipation",
    "tracers/Star",
    "tracers/WasRemoved",
    "X",
    "Y",
    "Z",
    "Pressure",
    "InternalEnergy",
)
(
    DENSITY,
    TEMPERATURE,
    DISSIPATION,
    STAR,
    WAS_REMOVED,
    X,
    Y,
    Z,
    PRESSURE,
    INTERNAL_ENERGY,
) = range(len(FIELDS))
FILTER_DEFINITIONS = {
    "physical": "Density, Temperature, Pressure and InternalEnergy finite and positive; "
    "Dissipation finite",
    "stellar": "finite Star >= min_star",
    "undrained": "finite abs(WasRemoved) <= max_was_removed",
    "temperature_limit": "Temperature < max_temperature",
}
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
MAX_NOZZLE_ROWS = 5000
GZIP = {"compression": "gzip", "compression_opts": 1, "shuffle": True}

_open_file = None


def _read_rank(job):
    """IDs and field values of one rank, optionally only for the sorted ``targets``."""
    global _open_file
    path, rank, targets, fields = job
    if _open_file is None or _open_file.filename != path:
        if _open_file is not None:
            _open_file.close()
        _open_file = h5py.File(path, "r")
    group = _open_file[rank]
    ids = np.asarray(group["ID"], dtype=np.uint64)
    if isinstance(targets, tuple):
        targets = np.memmap(targets[0], mode="r", dtype=np.uint64, shape=targets[1])
    if targets is None:
        return ids, np.column_stack(
            [group[field][()] for field in fields]
        ) if fields else None
    rows = np.flatnonzero((ids >= targets[0]) & (ids <= targets[-1]))
    rows = rows[targets[np.searchsorted(targets, ids[rows])] == ids[rows]]
    if not len(rows):
        return np.empty(0, dtype=np.uint64), np.empty((0, len(fields)))
    # Reading the whole rank beats a large HDF5 point selection.
    dense = len(rows) >= min(1024, max(1, len(ids) // 20))
    values = np.column_stack(
        [group[field][()][rows] if dense else group[field][rows] for field in fields]
    )
    return ids[rows], values


def read_ranks(path, ranks, targets, executor, workers, fields=FIELDS):
    """Yield ``(rank_index, (ids, values))`` for each rank of a snapshot.

    ``targets`` is None (every cell), a sorted uint64 array, or ``(memmap_path, count)``.
    With an executor, up to ``2 * workers`` ranks are read in parallel.
    """
    jobs = ((index, (path, rank, targets, fields)) for index, rank in enumerate(ranks))
    if executor is None:
        for index, job in jobs:
            yield index, _read_rank(job)
        return
    pending = {}

    def submit_next():
        item = next(jobs, None)
        if item is not None:
            pending[executor.submit(_read_rank, item[1])] = item[0]

    for _ in range(2 * workers):
        submit_next()
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            yield index, future.result()
            submit_next()


def rank_layout(path):
    """Rank group names in numerical order, and the snapshot's code time."""
    with h5py.File(path, "r") as source:
        ranks = sorted(
            (name for name in source if name.startswith("rank") and name[4:].isdigit()),
            key=lambda name: int(name[4:]),
        ) or ["/"]
        return ranks, float(np.asarray(source["Time"]).squeeze())


def snapshot_ids(path, ranks, executor, workers):
    """Sorted IDs of every cell in a snapshot."""
    ids = np.sort(
        np.concatenate(
            [
                rank_ids
                for _, (rank_ids, _) in read_ranks(
                    path, ranks, None, executor, workers, ()
                )
            ]
        )
    )
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def sample_filters(values, min_star, max_was_removed, max_temperature):
    """Per-sample validity tests on ``values[..., field]``, keyed as ``FILTER_DEFINITIONS``."""
    positive = values[..., [DENSITY, TEMPERATURE, PRESSURE, INTERNAL_ENERGY]]
    star, removed = values[..., STAR], values[..., WAS_REMOVED]
    return {
        "physical": (np.isfinite(positive) & (positive > 0)).all(axis=-1)
        & np.isfinite(values[..., DISSIPATION]),
        "stellar": np.isfinite(star) & (star >= min_star),
        "undrained": np.isfinite(removed) & (np.abs(removed) <= max_was_removed),
        "temperature_limit": values[..., TEMPERATURE] < max_temperature,
    }


def valid_samples(values, args):
    filters = sample_filters(
        values, args.min_star, args.max_was_removed, args.max_temperature
    )
    return np.logical_and.reduce(list(filters.values()))


@dataclass
class Catalogue:
    paths: list
    snapshots: list
    ranks: list
    times: np.ndarray


def make_catalogue(args):
    from dev.datapaths import DATAPATHS, snapshot_number

    paths = [
        path
        for path in DATAPATHS(args.run)
        if args.start_snapshot <= snapshot_number(path) <= args.stop_snapshot
    ][:: args.snapshot_step]
    if not paths:
        raise ValueError("No snapshots match the requested selection")
    ranks, times = zip(*(rank_layout(path) for path in paths))
    return Catalogue(
        paths, [snapshot_number(path) for path in paths], list(ranks), np.asarray(times)
    )


def signatures(paths):
    return [
        {
            "path": os.path.realpath(path),
            "size": os.stat(path).st_size,
            "mtime_ns": os.stat(path).st_mtime_ns,
        }
        for path in paths
    ]


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def start_run(args, catalogue, outputs, ids=None):
    """Fresh metadata for a run, or None when identical completed results already exist."""
    excluded = ("overwrite", "output_dir", "workers", "ids_file")
    settings = {key: value for key, value in vars(args).items() if key not in excluded}
    settings["sources"] = signatures(catalogue.paths)
    if ids is not None:
        settings["ids"] = [int(value) for value in ids]
        settings["ids_file"] = os.path.realpath(args.ids_file)
    target = os.path.join(args.output_dir, "metadata.json")
    if os.path.exists(target) and not args.overwrite:
        with open(target) as stream:
            previous = json.load(stream)
        complete = (
            previous.get("settings") == settings
            and previous.get("status") == "complete"
        )
        if complete and all(
            os.path.exists(os.path.join(args.output_dir, name)) for name in outputs
        ):
            print(
                f"Matching completed results exist in {args.output_dir}; skipping",
                flush=True,
            )
            return None
        raise FileExistsError(
            f"Existing/incomplete results differ; use --overwrite: {target}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "settings": settings,
        "status": "running",
        "snapshots": catalogue.snapshots,
        "selection_is_every_catalogue_snapshot": args.snapshot_step == 1,
        "workers": args.workers,
        "progress": [],
    }
    write_json(target, metadata)
    return metadata


def log_progress(metadata, output_dir, **progress):
    metadata["progress"].append(progress)
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    print(json.dumps(progress), flush=True)


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


def presence_intervals(args, catalogue, duration, executor, metadata):
    """Runs of consecutive presence lasting ``duration``, as (ids, first, last) sorted by ID."""
    times = catalogue.times
    old_ids = np.empty(0, dtype=np.uint64)
    starts = np.empty(0, dtype=np.int32)
    found = []
    finished_count = 0
    for index, path in enumerate(catalogue.paths):
        tick = time.monotonic()
        ids = snapshot_ids(path, catalogue.ranks[index], executor, args.workers)
        new_starts, dead = _advance_presence(
            old_ids, starts, ids, index, times, duration
        )
        found.append(
            (old_ids[dead], starts[dead], np.full(len(dead), index - 1, dtype=np.int32))
        )
        finished_count += len(dead)
        old_ids, starts = ids, new_starts
        log_progress(
            metadata,
            args.output_dir,
            stage="presence",
            snapshot=catalogue.snapshots[index],
            n_cells=len(ids),
            finished_candidate_intervals=finished_count,
            seconds=time.monotonic() - tick,
        )
    alive = np.flatnonzero(times[-1] - times[starts] >= duration)
    found.append(
        (
            old_ids[alive],
            starts[alive],
            np.full(len(alive), len(times) - 1, dtype=np.int32),
        )
    )
    ids, first, last = (np.concatenate(column) for column in zip(*found))
    order = np.lexsort((first, ids))
    return ids[order], first[order], last[order]


def store_valid_samples(
    args, catalogue, candidates, duration, stored, executor, metadata
):
    """Store the valid samples of each candidate and split candidates at invalid ones.

    Returns ``(candidate, first, last)`` rows of every valid run lasting ``duration``.
    """
    ids, first, last = candidates
    targets_path = os.path.join(args.output_dir, ".census-targets.u64.tmp")
    valid_starts = np.full(len(ids), -1, dtype=np.int32)
    finished = []
    for index, path in enumerate(catalogue.paths):
        tick = time.monotonic()
        active = np.flatnonzero((first <= index) & (last >= index))
        targets = ids[active]
        targets.tofile(targets_path)
        values = np.full((len(ids), len(FIELDS)), np.nan)
        seen = np.zeros(len(ids), dtype=bool)
        if len(active):
            for _, (found_ids, found_values) in read_ranks(
                path,
                catalogue.ranks[index],
                (targets_path, len(targets)),
                executor,
                args.workers,
            ):
                columns = active[np.searchsorted(targets, found_ids)]
                seen[columns] = True
                values[columns] = found_values
        good = seen & valid_samples(values, args)
        finished.append(
            _advance_valid(valid_starts, good, index, catalogue.times, duration)
        )
        values[~good] = np.nan
        for chunk in np.unique(np.flatnonzero(good) // 1024):
            begin = int(chunk) * 1024
            stored[index, begin : begin + 1024] = values[begin : begin + 1024]
        log_progress(
            metadata,
            args.output_dir,
            stage="physical",
            snapshot=catalogue.snapshots[index],
            candidate_cells=len(active),
            valid_stellar_cells=int(good.sum()),
            seconds=time.monotonic() - tick,
        )
    no_more = np.zeros(len(ids), dtype=bool)
    finished.append(
        _advance_valid(
            valid_starts, no_more, len(catalogue.paths), catalogue.times, duration
        )
    )
    os.remove(targets_path)
    return np.concatenate(finished)


def write_census_archive(
    path, stored, ids, intervals, catalogue, tfb, min_duration_tfb
):
    """Copy each interval's samples into one contiguous, compressed HDF5 archive."""
    intervals = intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]
    offsets = np.r_[0, np.cumsum(intervals[:, 2] - intervals[:, 1] + 1, dtype=np.int64)]
    total = int(offsets[-1])
    compressed = GZIP if total else {}
    with h5py.File(path, "w") as target:
        target.attrs.update(
            schema_version=1,
            fallback_time_code=tfb,
            min_duration_tfb=min_duration_tfb,
            frame="Snapshot stored coordinates; no frame correction",
            units="Density:code density; Temperature:K; Dissipation:code volumetric power; "
            "Star/WasRemoved:dimensionless; XYZ:code length; "
            "Pressure:code pressure; InternalEnergy:code specific energy",
        )
        target["ids"] = ids[intervals[:, 0]]
        target["offsets"] = offsets
        target["start_index"] = intervals[:, 1].astype(np.int32)
        target["end_index"] = intervals[:, 2].astype(np.int32)
        target["snapshots"] = np.asarray(catalogue.snapshots, dtype=np.int32)
        target["time_code"] = catalogue.times
        target["fields"] = np.asarray(FIELDS, dtype="S")
        chunks = {"chunks": (min(total, 4096), len(FIELDS))} if total else {}
        data = target.create_dataset(
            "values", (total, len(FIELDS)), "f8", **chunks, **compressed
        )
        sample_index = target.create_dataset(
            "snapshot_index", (total,), "i4", **compressed
        )
        begin = 0
        while begin < len(intervals):
            first_column = int(intervals[begin, 0])
            end = np.searchsorted(intervals[:, 0], first_column + 1024)
            block = stored[:, first_column : int(intervals[end - 1, 0]) + 1, :]
            rows = intervals[begin:end]
            data[offsets[begin] : offsets[end]] = np.concatenate(
                [
                    block[start : stop + 1, column - first_column]
                    for column, start, stop in rows
                ]
            )
            sample_index[offsets[begin] : offsets[end]] = np.concatenate(
                [np.arange(start, stop + 1, dtype=np.int32) for _, start, stop in rows]
            )
            begin = end
    return intervals, offsets


def build_census_archive(
    args, catalogue, candidates, duration, tfb, executor, metadata
):
    """Write ``histories.h5`` from the candidates, through a temporary per-snapshot work file."""
    ids = candidates[0]
    work_path = os.path.join(args.output_dir, ".census-candidates.h5.tmp")
    archive_path = os.path.join(args.output_dir, "histories.h5.tmp")
    shape = (len(catalogue.paths), len(ids), len(FIELDS))
    storage = {
        "chunks": (1, min(1024, len(ids)), len(FIELDS)),
        "fillvalue": np.nan,
        **GZIP,
    }
    with h5py.File(work_path, "w") as work:
        stored = work.create_dataset(
            "values", shape, "f8", **(storage if len(ids) else {})
        )
        intervals = (
            store_valid_samples(
                args, catalogue, candidates, duration, stored, executor, metadata
            )
            if len(ids)
            else np.empty((0, 3), dtype=np.int64)
        )
        intervals, offsets = write_census_archive(
            archive_path, stored, ids, intervals, catalogue, tfb, args.min_duration_tfb
        )
    os.replace(archive_path, os.path.join(args.output_dir, "histories.h5"))
    os.remove(work_path)
    return intervals, offsets


def census(args, executor):
    """Archive every sufficiently long valid stellar interval, without ranking."""
    from dev.tde import tde_basics

    import richio

    catalogue = make_catalogue(args)
    if args.snapshot_step != 1 or np.any(np.diff(catalogue.snapshots) != 1):
        raise ValueError("Census requires every consecutive snapshot, with no gaps")
    tfb = tde_basics(args.run).t_fb.to_value(richio.units.tscale)
    duration = args.min_duration_tfb * tfb
    metadata = start_run(args, catalogue, ("histories.h5", "ids.txt"))
    if metadata is None:
        return
    started = time.monotonic()
    metadata.update(
        fallback_time_code=float(tfb),
        min_duration_code=float(duration),
        times_code=catalogue.times.tolist(),
        filter_definitions=FILTER_DEFINITIONS,
    )
    candidates = presence_intervals(args, catalogue, duration, executor, metadata)
    ids = candidates[0]
    metadata.update(
        candidate_intervals=len(ids), candidate_unique_ids=len(np.unique(ids))
    )
    print(f"Presence census complete: {len(ids)} candidate intervals", flush=True)
    intervals, offsets = build_census_archive(
        args, catalogue, candidates, duration, tfb, executor, metadata
    )

    unique_ids = np.unique(ids[intervals[:, 0]])
    header = "Every ID with a valid stellar interval lasting at least the requested fallback-time duration"
    np.savetxt(
        os.path.join(args.output_dir, "ids.txt"), unique_ids, fmt="%d", header=header
    )
    metadata.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        qualifying_intervals=len(intervals),
        qualifying_unique_ids=len(unique_ids),
        samples=int(offsets[-1]),
    )
    write_json(os.path.join(args.output_dir, "metadata.json"), metadata)
    print(
        f"Completed census: {len(intervals)} intervals, {len(unique_ids)} IDs; {args.output_dir}"
    )


def write_histories(output_dir, ids, catalogue, present, data):
    header = (
        "snapshot time_code rho_code T_K dissipation_code star was_removed "
        "X_code Y_code Z_code pressure_code specific_internal_energy_code"
    )
    snapshots = np.asarray(catalogue.snapshots)
    for column, particle_id in enumerate(ids):
        mask = present[:, column]
        np.savetxt(
            os.path.join(output_dir, f"history_{int(particle_id)}.txt"),
            np.column_stack(
                (snapshots[mask], catalogue.times[mask], data[mask, column])
            ),
            header=header,
            fmt=["%d"] + ["%.17g"] * (len(FIELDS) + 1),
        )
    np.savez_compressed(
        os.path.join(output_dir, "histories.npz"),
        ids=ids,
        snapshots=snapshots.astype(np.int32),
        time_code=catalogue.times,
        present=present,
        values=data,
        fields=np.asarray(FIELDS),
    )


def extract(args, executor):
    """Save the full histories of explicitly listed IDs."""
    catalogue = make_catalogue(args)
    ids = np.loadtxt(args.ids_file, dtype=np.uint64, ndmin=1)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("ID file must contain one unique uint64 ID per line")
    outputs = ["histories.npz", *(f"history_{int(value)}.txt" for value in ids)]
    metadata = start_run(args, catalogue, outputs, ids)
    if metadata is None:
        return
    started = time.monotonic()
    order = np.argsort(ids)
    targets = ids[order]
    data = np.full((len(catalogue.paths), len(ids), len(FIELDS)), np.nan)
    present = np.zeros((len(catalogue.paths), len(ids)), dtype=bool)
    for index, path in enumerate(catalogue.paths):
        tick = time.monotonic()
        for _, (found_ids, values) in read_ranks(
            path, catalogue.ranks[index], targets, executor, args.workers
        ):
            columns = order[np.searchsorted(targets, found_ids)]
            data[index, columns] = values
            present[index, columns] = True
        log_progress(
            metadata,
            args.output_dir,
            snapshot=catalogue.snapshots[index],
            found=int(present[index].sum()),
            requested=len(ids),
            seconds=time.monotonic() - tick,
        )
    write_histories(args.output_dir, ids, catalogue, present, data)
    metadata.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        samples_per_id={
            str(int(value)): int(present[:, i].sum()) for i, value in enumerate(ids)
        },
    )
    write_json(os.path.join(args.output_dir, "metadata.json"), metadata)
    print(f"Completed extraction: {len(ids)} IDs; output {args.output_dir}", flush=True)


def activity_metrics(times, data, late):
    """Late-activity metrics and their mean percentile ``score`` for histories ``data[time, id]``."""
    import unyt as u
    from scipy.stats import rankdata

    import richio

    unit = richio.units.get_unit
    time_quantity = u.unyt_array(times, unit("Time"))
    density = u.unyt_array(data[:, :, DENSITY], unit("Density"))
    temperature = u.unyt_array(data[:, :, TEMPERATURE], "K")
    dissipation = u.unyt_array(data[:, :, DISSIPATION], unit("Dissipation"))
    internal_energy = u.unyt_array(data[:, :, INTERNAL_ENERGY], unit("InternalEnergy"))

    heating_rate = np.maximum(dissipation, 0) / (density * internal_energy)
    exposure = np.trapezoid(heating_rate[late], x=time_quantity[late], axis=0).to_value(
        "dimensionless"
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
    activity = (reheating, np.log1p(exposure), late_temperature_p90)
    metrics = {
        "score": np.mean(
            [(rankdata(metric) - 0.5) / len(reheating) for metric in activity], axis=0
        ),
        "late_reheating_dex": reheating,
        "late_heating_exposure": exposure,
        "late_temperature_p90_K": late_temperature_p90,
        "late_warm_duration_days": warm_duration.to_value("day"),
        "late_warm_fraction": (
            warm_duration / (time_quantity[late][-1] - time_quantity[late][0])
        ).to_value("dimensionless"),
        "logrho_span_dex": np.ptp(log_density, axis=0),
        "logT_span_dex": np.ptp(log_temperature, axis=0),
        "min_temperature_K": temperature.min(axis=0).to_value("K"),
        "max_temperature_K": temperature.max(axis=0).to_value("K"),
        "min_star": data[:, :, STAR].min(axis=0),
        "max_abs_was_removed": np.abs(data[:, :, WAS_REMOVED]).max(axis=0),
    }
    if not all(np.isfinite(metric).all() for metric in metrics.values()):
        raise ValueError("Nonfinite selection metrics; inspect the extracted histories")
    return metrics, log_density, log_temperature


def diverse_choice(scores, log_density, log_temperature, shortlist, keep):
    """Seed with the best score, then maximise ``0.7 score + 0.3 d/(1 + d)``.

    ``d`` is the smallest RMS distance in (log rho, log T), scaled by each
    quantity's 10th-90th percentile span, to any history chosen so far.
    """
    scales = np.array(
        [
            max(float(np.diff(np.percentile(curves, [10, 90]))[0]), 1e-12)
            for curves in (log_density, log_temperature)
        ]
    )
    curves = np.stack((log_density, log_temperature), axis=-1) / scales
    chosen = [int(shortlist[0])]
    distance = np.full(len(scores), np.inf)
    while len(chosen) < min(keep, len(shortlist)):
        latest = np.sqrt(
            np.mean((curves - curves[:, chosen[-1], None]) ** 2, axis=(0, 2))
        )
        distance = np.minimum(distance, latest)
        remaining = shortlist[~np.isin(shortlist, chosen)]
        utility = 0.7 * scores[remaining] + 0.3 * distance[remaining] / (
            1 + distance[remaining]
        )
        chosen.append(int(remaining[np.argmax(utility)]))
    return chosen, scales


def write_selection(args, sources, snapshots, times, late, selected_ids, rows, summary):
    """Write the chosen IDs, their histories and the selection record, unless already done."""
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in ("input_dir", "output_dir", "overwrite")
    }
    settings["sources"] = signatures(sources)
    destination = os.path.join(args.output_dir, "selection.json")
    histories_dir = os.path.join(args.output_dir, "histories")
    if os.path.exists(destination) and not args.overwrite:
        with open(destination) as stream:
            previous = json.load(stream)
        outputs = ["selected_ids.txt", "candidates_summary.tsv"]
        outputs += [
            os.path.join("histories", f"history_{value}.txt") for value in selected_ids
        ]
        if previous.get("settings") == settings and all(
            os.path.exists(os.path.join(args.output_dir, name)) for name in outputs
        ):
            print(
                f"Matching selection exists in {args.output_dir}; skipping", flush=True
            )
            return
        raise FileExistsError(
            f"Existing selection differs; use --overwrite: {destination}"
        )
    os.makedirs(histories_dir, exist_ok=True)
    for value in selected_ids:
        name = f"history_{value}.txt"
        shutil.copy2(
            os.path.join(args.input_dir, name), os.path.join(histories_dir, name)
        )
    with open(os.path.join(args.output_dir, "selected_ids.txt"), "w") as stream:
        stream.write(
            f"# Complete cell histories: snapshots {snapshots[0]}..{snapshots[-1]}, {len(snapshots)} samples\n"
            "# Ranked by late thermal activity, with diversity in density-temperature paths.\n"
            "# See selection.json for physical filters, scores, and source provenance.\n"
            + "".join(f"{value}\n" for value in selected_ids)
        )
    with open(
        os.path.join(args.output_dir, "candidates_summary.tsv"), "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        destination,
        {
            "schema_version": 1,
            "status": "complete",
            "settings": settings,
            "snapshots": snapshots.tolist(),
            "time_start_code": float(times[0]),
            "time_end_code": float(times[-1]),
            "late_start_snapshot": int(snapshots[late][0]),
            "late_start_time_code": float(times[late][0]),
            **summary,
        },
    )


def filter_counts(masks):
    """How many IDs pass each filter on its own and together with the ones before it."""
    cumulative = np.ones(len(next(iter(masks.values()))), dtype=bool)
    counts = {"input_ids": len(cumulative)}
    for name, mask in masks.items():
        cumulative &= mask
        counts[name] = {
            "passing_independently": int(mask.sum()),
            "passing_cumulatively": int(cumulative.sum()),
        }
    return counts


def selection_summary(masks, eligible, shortlist, scales, selected_rows):
    return {
        "filter_counts": filter_counts(masks),
        "filter_definitions": {
            "complete": "ID present at every catalogue snapshot",
            **FILTER_DEFINITIONS,
        },
        "eligible_ids": eligible,
        "selection_method": {
            "activity_score": "mean percentile rank of late reheating, log1p(exposure), and T90",
            "reheating": "max(log10(T) - running_min(log10(T))) within the late window",
            "exposure": "trapezoidal integral of max(Dissipation,0)/(Density*InternalEnergy) dt",
            "warm_duration": "trapezoidal integral of indicator(T > 1100 K) in the late window",
            "shortlist_size": len(shortlist),
            "diversity": "0.7*activity_score + 0.3*d/(1+d); d = scaled RMS (log rho, log T) distance",
            "curve_scales_dex": scales.tolist(),
        },
        "selected": selected_rows,
    }


def select(args):
    """Rank complete histories by late thermal activity, then diversify their curves."""
    sources = [
        os.path.join(args.input_dir, name)
        for name in ("metadata.json", "histories.npz")
    ]
    with open(sources[0]) as stream:
        source = json.load(stream)
    if (
        source["status"] != "complete"
        or not source["selection_is_every_catalogue_snapshot"]
    ):
        raise ValueError(
            "Selection requires a completed extraction of every catalogue snapshot"
        )
    with np.load(sources[1]) as archive:
        ids, snapshots, times = (
            archive["ids"],
            archive["snapshots"],
            archive["time_code"],
        )
        present, values = archive["present"], archive["values"]

    filters = sample_filters(
        values, args.min_star, args.max_was_removed, args.max_temperature
    )
    masks = {"complete": present.all(axis=0)}
    masks |= {name: mask.all(axis=0) for name, mask in filters.items()}
    columns = np.flatnonzero(np.logical_and.reduce(list(masks.values())))
    if not len(columns):
        raise ValueError("No complete histories satisfy the physical selection")
    late = times >= times[0] + args.late_fraction * (times[-1] - times[0])
    if np.count_nonzero(late) < 2:
        raise ValueError("The late-time window needs at least two samples")

    metrics, log_density, log_temperature = activity_metrics(
        times, values[:, columns], late
    )
    order = np.lexsort((ids[columns], -metrics["score"]))
    shortlist = order[: max(100, args.keep)]
    chosen, scales = diverse_choice(
        metrics["score"], log_density, log_temperature, shortlist, args.keep
    )
    rows = [
        {
            "id": int(ids[columns[i]]),
            **{key: float(value[i]) for key, value in metrics.items()},
        }
        for i in order
    ]
    rows_by_id = {row["id"]: row for row in rows}
    selected_ids = [int(ids[columns[i]]) for i in chosen]
    selected_rows = [rows_by_id[value] for value in selected_ids]
    summary = selection_summary(masks, len(columns), shortlist, scales, selected_rows)
    write_selection(args, sources, snapshots, times, late, selected_ids, rows, summary)
    print(
        f"Selected {len(selected_ids)} of {len(columns)} eligible histories: {selected_ids}"
    )


def nozzle_row(archive, interval, late_catalogue, times_tfb, r_p, args):
    """Catalogue row of one census interval, or None if too few samples are near the nozzle."""
    row_start, row_stop = map(int, archive["offsets"][interval : interval + 2])
    indices = archive["snapshot_index"][row_start:row_stop]
    late = late_catalogue[indices]
    if np.count_nonzero(late) < 2:
        return None, 0
    values = archive["values"][row_start:row_stop]
    radius = np.linalg.norm(values[:, [X, Y, Z]], axis=1)
    near = late & (radius < args.radius_rp * r_p) & (values[:, X] > 0)
    near_points = int(np.count_nonzero(near))
    if near_points < args.min_points:
        return None, near_points

    sample_times = times_tfb[indices]
    near_times = sample_times[near]
    near_indices = np.flatnonzero(near)
    near_runs = np.split(near_indices, np.flatnonzero(np.diff(near_indices) > 1) + 1)
    snapshots = archive["snapshots"]
    row = {
        "id": int(archive["ids"][interval]),
        "interval_index": int(interval),
        "row_start": row_start,
        "row_stop": row_stop,
        "first_snapshot": int(snapshots[indices[0]]),
        "last_snapshot": int(snapshots[indices[-1]]),
        "n_samples": len(indices),
        "duration_tfb": float(sample_times[-1] - sample_times[0]),
        "near_points": near_points,
        "max_consecutive_near_points": max(map(len, near_runs)),
        "first_near_time_tfb": float(near_times[0]),
        "last_near_time_tfb": float(near_times[-1]),
        "time_before_first_near_tfb": float(near_times[0] - sample_times[0]),
        "time_after_last_near_tfb": float(sample_times[-1] - near_times[-1]),
        "min_late_radius_rp": float(np.min(radius[late & np.isfinite(radius)]) / r_p),
        "max_near_temperature_K": float(np.max(values[near, TEMPERATURE])),
        "max_positive_near_dissipation_code": float(
            max(0, np.max(values[near, DISSIPATION]))
        ),
    }
    return row, near_points


def write_nozzle_catalogue(output_dir, rows, metadata):
    os.makedirs(output_dir, exist_ok=True)
    rows.sort(key=lambda row: (-row["duration_tfb"], row["id"], row["interval_index"]))
    unique_ids = np.unique(np.asarray([row["id"] for row in rows], dtype=np.uint64))
    np.savetxt(
        os.path.join(output_dir, "ids.txt"),
        unique_ids,
        fmt="%d",
        header="Unique nozzle-region candidate IDs; see catalogue.tsv and metadata.json",
    )
    with open(os.path.join(output_dir, "catalogue.tsv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=NOZZLE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metadata["counts"]["selected_unique_ids"] = len(unique_ids)
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    print(
        f"Wrote {len(rows):,} intervals ({len(unique_ids):,} unique IDs) to {output_dir}",
        flush=True,
    )


def nozzle_candidates(archive, args, tfb, r_p):
    """Catalogue rows of long census intervals with enough late samples near the nozzle."""
    from dev.datapaths import DATAPATHS, snapshot_number
    from dev.tde import needs_frame_switch

    star_frame = {
        snapshot_number(path)
        for path in DATAPATHS(args.run)
        if needs_frame_switch(args.run, path)
    }
    times = archive["time_code"][:]
    late_catalogue = times > args.after_tfb * tfb
    if star_frame & set(archive["snapshots"][late_catalogue].tolist()):
        raise ValueError(
            "The time cut includes star-frame snapshots; increase --after-tfb"
        )
    starts, ends = archive["start_index"][:], archive["end_index"][:]
    candidates = np.flatnonzero(
        times[ends] - times[starts] >= args.min_duration_tfb * tfb
    )
    print(
        f"Lifetime cut: {len(candidates):,}/{len(starts):,} intervals >= {args.min_duration_tfb:g} tfb"
    )

    names = ("ids", "offsets", "snapshot_index", "snapshots", "values")
    datasets = {name: archive[name] for name in names}
    rows = []
    counts = {
        "source_intervals": len(starts),
        "lifetime_eligible_intervals": len(candidates),
        "near_points_ge_2": 0,
        "near_points_ge_3": 0,
    }
    for number, interval in enumerate(candidates, 1):
        row, near_points = nozzle_row(
            datasets, interval, late_catalogue, times / tfb, r_p, args
        )
        counts["near_points_ge_2"] += int(near_points >= 2)
        counts["near_points_ge_3"] += int(near_points >= 3)
        if row is not None:
            rows.append(row)
        if number % 10000 == 0:
            print(
                f"Inspected {number:,}/{len(candidates):,}; selected {len(rows):,}",
                flush=True,
            )
    counts["selected_intervals"] = len(rows)
    return rows, counts


def nozzle(args):
    """Index census intervals with late samples near the nozzle; copies no histories."""
    from dev.tde import tde_basics

    import richio

    outputs = ("ids.txt", "catalogue.tsv", "metadata.json")
    if args.output_dir and not args.overwrite:
        if any(os.path.exists(os.path.join(args.output_dir, name)) for name in outputs):
            raise FileExistsError(
                f"Existing output requires --overwrite: {args.output_dir}"
            )
    tde = tde_basics(args.run)
    tfb, r_p = (
        tde.t_fb.to_value(richio.units.tscale),
        tde.r_p.to_value(richio.units.lscale),
    )
    started = time.monotonic()
    with h5py.File(args.input_file, "r", rdcc_nbytes=32 * 1024**2) as archive:
        rows, counts = nozzle_candidates(archive, args, tfb, r_p)
    print(f"Late near-nozzle interval counts: {counts}", flush=True)
    if args.count_only:
        return
    if len(rows) > MAX_NOZZLE_ROWS:
        raise ValueError(
            f"{len(rows):,} intervals exceed {MAX_NOZZLE_ROWS:,}; raise --min-duration-tfb"
        )
    criteria = ("min_duration_tfb", "after_tfb", "radius_rp", "min_points")
    metadata = {
        "schema_version": 1,
        "command": "nozzle",
        "status": "complete",
        "run": args.run,
        "source": os.path.realpath(args.input_file),
        "criteria": {name: getattr(args, name) for name in criteria},
        "rp_code_length": r_p,
        "fallback_time_code": tfb,
        "counts": counts,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_nozzle_catalogue(args.output_dir, rows, metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    snapshots = argparse.ArgumentParser(add_help=False)
    snapshots.add_argument("--run", choices=("1e4", "1e5", "1e6"), default="1e4")
    snapshots.add_argument("--start-snapshot", type=int, default=0)
    snapshots.add_argument("--stop-snapshot", type=int, default=151)
    snapshots.add_argument("--snapshot-step", type=int, default=1)
    snapshots.add_argument("--workers", type=int, default=1)
    snapshots.add_argument("--output-dir", required=True)
    snapshots.add_argument("--overwrite", action="store_true")
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--min-star", type=float, default=0.99)
    filters.add_argument("--max-was-removed", type=float, default=1e-3)
    filters.add_argument(
        "--max-temperature", type=float, default=1e9, help="Strict upper limit in K"
    )

    census_parser = commands.add_parser("census", parents=[snapshots, filters])
    census_parser.add_argument("--min-duration-tfb", type=float, default=1.0)
    extract_parser = commands.add_parser("extract", parents=[snapshots])
    extract_parser.add_argument("--ids-file", required=True)
    select_parser = commands.add_parser("select", parents=[filters])
    select_parser.add_argument("--input-dir", required=True)
    select_parser.add_argument("--output-dir", required=True)
    select_parser.add_argument("--keep", type=int, default=6)
    select_parser.add_argument("--late-fraction", type=float, default=0.1)
    select_parser.add_argument("--overwrite", action="store_true")
    nozzle_parser = commands.add_parser("nozzle")
    nozzle_parser.add_argument("--input-file", required=True)
    nozzle_parser.add_argument("--output-dir")
    nozzle_parser.add_argument("--run", choices=("1e4", "1e5", "1e6"), default="1e4")
    nozzle_parser.add_argument("--min-duration-tfb", type=float, default=2.0)
    nozzle_parser.add_argument("--after-tfb", type=float, default=0.5)
    nozzle_parser.add_argument("--radius-rp", type=float, default=2.0)
    nozzle_parser.add_argument("--min-points", type=int, choices=(2, 3), default=3)
    nozzle_parser.add_argument("--count-only", action="store_true")
    nozzle_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "nozzle" and not (args.count_only or args.output_dir):
        parser.error("--output-dir is required unless --count-only is used")
    return args


def main():
    args = parse_args()
    if args.command in ("select", "nozzle"):
        {"select": select, "nozzle": nozzle}[args.command](args)
        return
    command = {"census": census, "extract": extract}[args.command]
    if args.workers == 1:
        command(args, None)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        command(args, executor)


if __name__ == "__main__":
    main()
