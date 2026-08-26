"""Benchmark streamed projection timing and peak memory on a RICH snapshot.

This is intentionally not part of the test suite. Example::

    python works/richio-development/benchmark_gridding.py snap_0042.h5 \
        --res 256 --res 512
    python works/richio-development/benchmark_gridding.py snap_0042.h5 --res 1024 \
        --workers 16 --z-spacing sinh --sinh-scale 0.1
"""

import argparse
import json
import multiprocessing as mp
import resource
from pathlib import Path
from time import perf_counter

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
