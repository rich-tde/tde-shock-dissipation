import glob
import os

import numpy as np
import unyt as u

import richio


OUTPUT_DIR = "/home/hey4/rich_tde/data/processed/SS24-circularization-t"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "SS24-circularization-t-1e6.txt")
HEADER = "SNAPNUM\tTIME\tTIME_DAYS\tTFALLBACK\tEORB_BOUND\tMBOUND\tEDISS_TOTAL"
FOOTER = (
    "Merged from the serial 809--922 checkpoint and non-overlapping Slurm shards\n"
    "EORB_BOUND uses orbital-specific-energy < 0; EDISS_TOTAL = sum(dissipation * volume)"
)


def main():
    paths = [OUTPUT_FILE] + sorted(glob.glob(os.path.join(OUTPUT_DIR, "shard-*.txt")))
    if len(paths) != 5:
        raise ValueError(
            f"Expected the main checkpoint plus four shards; found {paths}"
        )
    raw = np.vstack([np.atleast_2d(np.loadtxt(path)) for path in paths])
    if raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError("A shard has the wrong schema or contains NaN/infinity")
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    snapnums = raw[:, 0].astype(int)
    if not np.array_equal(raw[:, 0], snapnums):
        raise ValueError("A shard contains a non-integer snapshot number")
    expected = np.arange(809, 1007)
    if not np.array_equal(snapnums, expected):
        raise ValueError(
            f"Merged snapshots are not exactly 809--1006: got {snapnums[0]}--{snapnums[-1]} "
            f"with {len(snapnums)} rows"
        )
    if np.any(np.diff(raw[:, 2]) <= 0):
        raise ValueError("Merged times are not strictly increasing")

    registry = richio.units.registry
    arrays = [
        u.unyt_array(snapnums),
        u.unyt_array(raw[:, 1], "code_time", registry=registry),
        u.unyt_array(raw[:, 2], "day"),
        u.unyt_array(raw[:, 3]),
        u.unyt_array(
            raw[:, 4],
            "code_length**2*code_mass/code_time**2",
            registry=registry,
        ),
        u.unyt_array(raw[:, 5], "code_mass", registry=registry),
        u.unyt_array(
            raw[:, 6],
            "code_length**2*code_mass/code_time**3",
            registry=registry,
        ),
    ]
    temporary = f"{OUTPUT_FILE}.tmp"
    u.savetxt(temporary, arrays=arrays, header=HEADER, footer=FOOTER)
    os.replace(temporary, OUTPUT_FILE)
    print(f"Merged {len(raw)} rows into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
