"""Paths to the TDE snapshots."""

import re
from pathlib import Path


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
