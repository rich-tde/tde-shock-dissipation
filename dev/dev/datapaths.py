"""Paths to the TDE snapshots."""

import re
import warnings
from functools import cache
from math import pi, sqrt
from pathlib import Path

import h5py
import numpy as np


DATADIRS = {
    "1e4": (
        Path(
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/"
            "R0.47M0.5BH10000beta1S60ComptonHiRes"
        ),
    ),
    "1e5": (
        Path(
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/"
            "R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR"
        ),
    ),
    "1e6": (
        Path("/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE"),
        Path("/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4"),
        Path("/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new"),
    ),
}

# Mbh, Mstar, and Rstar in code units.  RICH uses G = 1.
TDE_PARAMETERS = {
    "1e4": (1e4, 0.5, 0.47),
    "1e5": (1e5, 0.5, 0.47),
    "1e6": (1e6, 1.0, 1.0),
}


def DATAPATHS(run):
    """Return sorted ``(snapnums, paths)`` for ``1e4``, ``1e5``, or ``1e6``."""

    snapshots = {}
    for datadir in DATADIRS[run]:
        for path in datadir.glob("snap_*.h5"):
            match = re.fullmatch(r"snap_(?:full_)?(\d+)\.h5", path.name)
            if match is None:
                continue

            snapnum = int(match.group(1))
            if datadir.name == "TEMPTDE4" and snapnum >= 820:
                continue

            # Prefer snap_full when both versions exist.
            old_path = snapshots.get(snapnum)
            if old_path is None or "snap_full_" in path.name:
                snapshots[snapnum] = path

    snapnums = sorted(snapshots)
    paths = [snapshots[snapnum] for snapnum in snapnums]
    return snapnums, paths


@cache
def _snapshot_times(run):
    """Read snapshot times once and keep them for later lookups."""

    snapnums, paths = DATAPATHS(run)
    times = []
    for path in paths:
        with h5py.File(path) as f:
            times.append(float(np.asarray(f["Time"]).squeeze()))
    return snapnums, paths, np.asarray(times)


def SNAPSHOT_TFB(run, tfb, warn_if=0.05):
    """Return ``(snapnum, path)`` closest to the requested ``t / t_fb``.

    A warning is raised when the closest snapshot is farther away than
    ``warn_if`` fallback times.
    """

    Mbh, Mstar, Rstar = TDE_PARAMETERS[run]
    fallback_time = pi / sqrt(2) * sqrt(Rstar**3 / Mstar) * sqrt(Mbh / Mstar)

    snapnums, paths, times = _snapshot_times(run)
    snapshot_tfbs = times / fallback_time
    index = int(np.argmin(abs(snapshot_tfbs - tfb)))
    difference = abs(snapshot_tfbs[index] - tfb)

    if difference > warn_if:
        warnings.warn(
            f"Closest {run} snapshot is at {snapshot_tfbs[index]:.3f} t_fb, "
            f"which is {difference:.3f} t_fb from the requested {tfb:.3f} t_fb",
            stacklevel=2,
        )

    return snapnums[index], paths[index]
