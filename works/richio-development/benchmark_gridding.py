r"""Measure streamed projection runtime and peak memory for a RICH snapshot.

Each grid-resolution/worker-count pair runs in a fresh subprocess. The field
is sampled on a cubic grid in stored X/Y/Z, integrated along z and converted
to cgs. Stage timings and Linux peak resident memory quantify implementation
cost. OS file-cache state affects timings; this is not a physical convergence
test and is not part of the unit-test suite.

Input files
-----------
``snapshot`` (positional argument)
    RICH HDF5 file or extracted NPY directory readable by ``richio.load``.
    Requires X/Y/Z, the snapshot box and ``--field`` (default density).
``--res``, ``--workers``, ``--z-spacing``, ``--sinh-scale``
    ``--res`` may be repeated (default 256). Worker counts are a space-separated
    list, default ``1 2 4 8 16 -1``; -1 uses all available query threads.
    Line-of-sight spacing is linear or sinh; sinh requires a central scale in
    RICH code lengths (solar-radius scale).

Output files
------------
The script writes no files directly. Stdout contains one JSON object per
completed pair; redirect it to a ``.jsonl`` file to retain the results. Each
object has the following keys:

``resolution``, ``workers``, ``cells``
    Integers: grid samples per axis, requested query threads (-1 means all),
    and cells used to build the tree.
``z_spacing``, ``sinh_scale``
    String ``"linear"`` or ``"sinh"``; numeric central scale in code lengths,
    or JSON ``null`` when omitted.
``field_loading_s``, ``grid_preparation_s``, ``tree_build_s``
    Floating-point wall seconds for reading cell fields, preparing grid
    coordinates and building the nearest-neighbour tree.
``query_integration_s``, ``total_s``
    Floating-point wall seconds for nearest-neighbour queries plus z
    integration, and total worker time from snapshot loading through result
    conversion (excluding parent/process-start overhead).
``peak_rss_mib``
    Floating-point peak resident memory of the worker in MiB (2**20 bytes).
``checksum``, ``unit``
    Floating-point sum of map values and their cgs unit string (density:
    ``g/cm**2``). The internal map is ``float64 (resolution-1, resolution-1)``,
    axes x then y; it is not saved. The checksum omits pixel area and is not
    an area-integrated physical total. Values are linear, not logarithms.

The JSONL has one record per completed case, not one rectangular NumPy array;
load each line with ``json.loads``. Snapshot path and field name are not stored
in the records, so retain the invocation with the file.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    mkdir -p data/processed/RichioDevelopment
    python works/richio-development/benchmark_gridding.py /path/to/snap_21.h5 \
        --res 64 --workers 1 4 > data/processed/RichioDevelopment/gridding.jsonl
    python works/richio-development/benchmark_gridding.py /path/to/snap_21.h5 \
        --res 128 --workers 4 --z-spacing sinh --sinh-scale 0.1

Every rerun repeats every pair. Shell ``>`` replaces the specified JSONL file;
there is no resume or append mode in the script. Importing does not benchmark.

Loading examples
----------------
Read retained records and compare timing/memory across cases::

    import json
    import numpy as np

    with open("data/processed/RichioDevelopment/gridding.jsonl") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    measurements = np.array([
        [row["total_s"], row["peak_rss_mib"]] for row in records
    ])  # float64 (N, 2): column 0=seconds, 1=MiB; N=completed pairs
    print(measurements)
    print([(row["resolution"], row["workers"]) for row in records])
"""

import argparse
import json
import multiprocessing as mp
import os
import resource
from pathlib import Path
from time import perf_counter

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import unyt as u
from richio.data import _iter_3d_nearest_slabs
from scipy.spatial import KDTree

import richio


def _run_one(
    snapshot_path,
    field,
    resolution,
    workers,
    z_spacing,
    sinh_scale,
    result_queue,
):
    started = perf_counter()
    snapshot = richio.load(str(snapshot_path))

    stage_started = perf_counter()
    x, y, z = snapshot.X, snapshot.Y, snapshot.Z
    field_values = snapshot._get_data(field)
    field_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    coordinates, source_indices, xspace, yspace, zspace = snapshot._prepare_3d_grid(
        res=resolution,
        X=x,
        Y=y,
        Z=z,
        spacing=("linear", "linear", z_spacing),
        sinh_scale=sinh_scale,
    )
    preparation_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    tree = KDTree(coordinates)
    tree_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    dz = np.asarray(zspace[1:] - zspace[:-1])
    values = np.asarray(field_values)
    projected = np.empty((resolution - 1, resolution - 1), dtype="float64")
    for slab_start, local_indices in _iter_3d_nearest_slabs(
        tree, xspace[:-1], yspace[:-1], zspace[:-1], workers=workers
    ):
        indices = (
            source_indices[local_indices]
            if source_indices is not None
            else local_indices
        )
        projected[slab_start : slab_start + len(indices)] = np.sum(
            values[indices] * dz, axis=-1
        )
    query_integration_seconds = perf_counter() - stage_started

    result = u.unyt_array(projected, field_values.units * zspace.units).in_base("cgs")
    result_queue.put(
        {
            "resolution": resolution,
            "workers": workers,
            "z_spacing": z_spacing,
            "sinh_scale": sinh_scale,
            "cells": len(coordinates),
            "field_loading_s": field_seconds,
            "grid_preparation_s": preparation_seconds,
            "tree_build_s": tree_seconds,
            "query_integration_s": query_integration_seconds,
            "total_s": perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "checksum": float(np.sum(np.asarray(result))),
            "unit": str(result.units),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="HDF5 snapshot file or NPY snapshot directory")
    parser.add_argument("--field", default="density", help="field to project")
    parser.add_argument(
        "--res",
        type=int,
        action="append",
        dest="resolutions",
        help="cubic grid resolution; repeat to benchmark several (default: 256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, -1],
        help="worker counts to compare (default: 1 2 4 8 16 -1)",
    )
    parser.add_argument(
        "--z-spacing",
        choices=("linear", "sinh"),
        default="linear",
        help="line-of-sight grid spacing (default: linear)",
    )
    parser.add_argument(
        "--sinh-scale",
        type=float,
        help="central scale in RICH code lengths; required with --z-spacing=sinh",
    )
    arguments = parser.parse_args()
    if arguments.z_spacing == "sinh" and arguments.sinh_scale is None:
        parser.error("--sinh-scale is required with --z-spacing=sinh")

    context = mp.get_context("spawn")
    snapshot_path = Path(arguments.snapshot)
    for resolution in arguments.resolutions or [256]:
        for workers in arguments.workers:
            result_queue = context.Queue()
            process = context.Process(
                target=_run_one,
                args=(
                    snapshot_path,
                    arguments.field,
                    resolution,
                    workers,
                    arguments.z_spacing,
                    arguments.sinh_scale,
                    result_queue,
                ),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise SystemExit(
                    f"benchmark failed for res={resolution}, workers={workers} "
                    f"with exit code {process.exitcode}"
                )
            print(json.dumps(result_queue.get(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
