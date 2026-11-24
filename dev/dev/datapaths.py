"""Paths to the TDE snapshots."""

from functools import cache
import glob
from math import pi, sqrt
import os
import re
import warnings

import h5py
import numpy as np

TDE_DATA = "/data1/projects/pi-rossiem/TDE_data"
DATADIRS = {
    "1e4": (os.path.join(TDE_DATA, "NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes"),),
    "1e4midres": (os.path.join(TDE_DATA, "NewSnellius/R0.47M0.5BH10000beta1S60Compton"),),
    "1e4lowres": (os.path.join(TDE_DATA, "R0.47M0.5BH10000beta1S60n1.5ComptonLowResNewAMR"),),
    "1e5": (
        os.path.join(TDE_DATA, "YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR"),
    ),
    "1e6": (os.path.join(TDE_DATA, "R0.47M0.5BH1e+06beta1S50n1.5ComptonMG"),),
    "ss24": (
        os.path.join(TDE_DATA, "SS24_diag/TEMPTDE"),
        os.path.join(TDE_DATA, "SS24_diag/TEMPTDE4"),
        os.path.join(TDE_DATA, "SS24_diag/TEMPTDE4_new"),
    ),
}

# Mbh, Mstar, and Rstar in code units.  RICH uses G = 1.
TDE_PARAMETERS = {
    "1e4": (1e4, 0.5, 0.47),
    "1e4midres": (1e4, 0.5, 0.47),
    "1e4lowres": (1e4, 0.5, 0.47),
    "1e5": (1e5, 0.5, 0.47),
    "1e6": (1e6, 0.5, 0.47),
    "ss24": (1e6, 1.0, 1.0),
}


def snapshot_number(path):
    """The number ``N`` of ``snap_N.h5``, ``snap_full_N.h5`` or an NPY folder ``snap_N``."""
    return int(re.search(r"(\d+)(?:\.h5)?$", os.path.basename(os.path.normpath(path))).group(1))


def DATAPATHS(run):
    """Return snapshot paths sorted by number for a run in ``DATADIRS``.

    ``1e4lowres`` returns NumPy folders; the other runs return HDF5 files.
    ``snap_full_N.h5`` is preferred when both versions of a snapshot exist.
    """
    snapshots = {}
    numpy_folders = run == "1e4lowres"
    for datadir in DATADIRS[run]:
        for path in glob.glob(os.path.join(datadir, "snap_*" if numpy_folders else "snap_*.h5")):
            name = os.path.basename(path)
            match = re.fullmatch(r"snap_(?:full_)?(\d+)(?:\.h5)?", name)
            if match is None or (numpy_folders and not os.path.isdir(path)):
                continue
            snapnum = int(match.group(1))
            if os.path.basename(datadir) == "TEMPTDE4" and snapnum >= 820:
                continue
            if snapnum not in snapshots or name.startswith("snap_full_"):
                snapshots[snapnum] = path
    return [snapshots[snapnum] for snapnum in sorted(snapshots)]


@cache
def _snapshot_times(run):
    """Read snapshot times once and keep them for later lookups."""
    import richio

    paths = DATAPATHS(run)
    snapnums = [snapshot_number(path) for path in paths]
    times = []
    for path in paths:
        if os.path.isdir(path):
            # NPY snapshots store elapsed time in their calibrated fallback unit.
            time = richio.load(path).tfb.to_value(richio.units.tscale)
            times.append(float(np.asarray(time).squeeze()))
        else:
            with h5py.File(path) as f:
                times.append(float(np.asarray(f["Time"]).squeeze()))
    return snapnums, paths, np.asarray(times)


def SNAPSHOT_TIMES(run):
    """Return snapshot times as quantities in richio's code-time registry."""
    from unyt import unyt_array

    import richio

    return unyt_array(_snapshot_times(run)[2].copy(), richio.units.tscale)


def SNAPSHOT_TFB(run, tfb, warn_if=0.05):
    """Return the snapshot path closest to the requested ``t / t_fb``.

    A list of times returns a list of paths in the requested order. A warning
    is raised for each match farther away than ``warn_if`` fallback times.
    """

    requested_tfbs = np.atleast_1d(tfb)
    if len(requested_tfbs) == 0:
        return []

    Mbh, Mstar, Rstar = TDE_PARAMETERS[run]
    fallback_time = pi / sqrt(2) * sqrt(Rstar**3 / Mstar) * sqrt(Mbh / Mstar)

    _, paths, times = _snapshot_times(run)
    snapshot_tfbs = times / fallback_time
    selected = []
    for requested_tfb in requested_tfbs:
        index = int(np.argmin(abs(snapshot_tfbs - requested_tfb)))
        difference = abs(snapshot_tfbs[index] - requested_tfb)

        if difference > warn_if:
            warnings.warn(
                f"Closest {run} snapshot is at {snapshot_tfbs[index]:.3f} t_fb, "
                f"which is {difference:.3f} t_fb from the requested "
                f"{requested_tfb:.3f} t_fb",
                stacklevel=2,
            )

        selected.append(paths[index])

    return selected[0] if np.ndim(tfb) == 0 else selected
