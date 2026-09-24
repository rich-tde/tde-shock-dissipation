"""Measure the smallest native-cell size in the compressed nozzle midplane.

Read raw RICH HDF5 arrays rank by rank, correct moving-frame positions to the
BH frame, and find the minimum-volume cell in ``0.6 < R/r_p < 1.75`` with
``|z| <= d``, where ``d = 2*(3*Volume/(4*pi))**(1/3)`` is each cell's equivalent
sphere diameter. Report that cell's size, density, mass and position. This is
a minimum native-cell-size diagnostic, not a convergence test or a measurement
of resolved vertical scale height. R denotes cylindrical radius.

Input files
-----------
The positional run is ``1e4``, ``1e5`` or ``1e6``.
``dev.datapaths.DATAPATHS`` resolves ``snap_full_<n>.h5`` or ``snap_<n>.h5`` with
restart boundaries. The files must contain ``Time`` (scalar or singleton) and
1D ``X``, ``Y``, ``Z``, ``Volume``, ``Density`` datasets either at the root or
inside ``rank<n>`` groups; each rank's arrays have equal native-cell counts.
Raw values use RICH code units. ``--input-file`` supplies an alternate snapshot
using the chosen run's stellar/BH parameters; its name must end in ``<n>.h5``.

Default nearest epochs in ``t/t_fb`` are 0.5, 1, 1.5, 2 for ``1e4``; 0.3, 0.5
for ``1e5``; and 1, 1.2, 1.4, 1.5 for ``1e6``, plus the final snapshot. Repeat
``--time-tfb`` or ``--snapshot-number`` to select other epochs. Explicit numbers
avoid scanning every Time dataset. ``--no-include-last`` omits the final epoch
when selecting by time.

Output files
------------
Default output is one JSON object per snapshot on stdout; ``--output PATH``
writes the same records as UTF-8 JSON Lines, not a single JSON array. Parse each
line with ``json.loads``. There is one record per unique selected snapshot and
no positional column schema. All numbers are JSON scalars, not NumPy arrays.
The top-level keys are:

``run``, ``path`` : str
    Mass label and source HDF5 path.
``snapnum``, ``cell_count`` : int
    Snapshot number and total native cells across all ranks (not selected count).
``time_code``, ``time_tfb`` : float
    Snapshot time in code time units and divided by this run's fallback time.
    One code time unit is ``richio.units.tscale``.
``rp_rsun`` : float
    Pericentre radius in solar radii under the study's code-unit convention.
``compressed_midplane`` : dict
    Minimum-volume selected cell, with the following complete nested schema.

``compressed_midplane.selection`` : str
    ``cylindrical_annulus_and_one_cell_diameter_midplane`` for the default
    height cut, otherwise ``cylindrical_annulus_and_cell_diameter_midplane``.
``compressed_midplane.radius_min_rp``, ``compressed_midplane.radius_max_rp`` : float
    Strict inner/outer cylindrical selection radii divided by r_p.
``compressed_midplane.midplane_diameters`` : float
    Maximum allowed ``|z|/d``; the height cut is inclusive.
``compressed_midplane.volume_rsun3``, ``compressed_midplane.diameter_rsun`` : float
    Cell volume in R_sun**3 and equivalent-sphere diameter in R_sun.
``compressed_midplane.x_rp``, ``compressed_midplane.y_rp``, ``compressed_midplane.z_rp`` : float
    BH-frame Cartesian coordinates divided by pericentre radius.
``compressed_midplane.z_rsun`` : float
    Vertical coordinate in R_sun.
``compressed_midplane.r_sph_rp``, ``compressed_midplane.r_cyl_rp`` : float
    Spherical and cylindrical radii divided by r_p.
``compressed_midplane.rank`` : str
    HDF5 group name such as ``rank12``; ``root`` for unpartitioned files.
``compressed_midplane.rank_index`` : int
    Zero-based index in that group's datasets, not a global cell index.
``compressed_midplane.density_code`` : float
    Raw code density, using M_sun/R_sun**3 under this study's convention.
``compressed_midplane.mass_msun`` : float
    Raw Density times Volume, reported as M_sun.

Lengths/masses above preserve the original requested solar-unit interpretation
of raw code values; no independent unit conversion is applied by this scanner.
All saved values are linear. An empty selection raises an error instead of
writing a null/missing result.

Usage
-----
Run from ``/home/hey4/rich_tde`` in the richanalysis environment::

    python works/nozzle-resolution/check_nozzle_resolution.py 1e4
    python works/nozzle-resolution/check_nozzle_resolution.py 1e4 --snapshot-number 108 --output data/processed/NozzleResolution/1e4/snap-0108.jsonl
    python works/nozzle-resolution/check_nozzle_resolution.py 1e4 --input-file data/external/my-run/snap_full_108.h5

Existing ``--output`` files are skipped unless ``--overwrite``; stdout always
recalculates. File output is staged alongside its destination and replaced after
success. Change ``--radius-min-rp``, ``--radius-max-rp`` or
``--midplane-diameters`` for different cuts; keep altered selections in separate
files or explicitly overwrite. Native arrays avoid a 3D grid but evolved
snapshots can still require substantial I/O and memory.

Loading examples
----------------
Load JSON Lines, print the minimum-cell diameter and construct coordinates::

    import json
    from pathlib import Path
    import numpy as np

    path = Path("data/processed/NozzleResolution/1e4/snap-0108.jsonl")
    with path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    for record in records:
        cell = record["compressed_midplane"]
        print(record["snapnum"], cell["diameter_rsun"], "R_sun")
    xyz = np.array([[row["compressed_midplane"][f"{axis}_rp"] for axis in "xyz"]
                    for row in records])
    # xyz.shape == (number_of_records, 3); columns 0=x, 1=y, 2=z, all in r_p units.

To retrieve the original chosen cell from the HDF5 file::

    import h5py
    record = records[0]
    cell = record["compressed_midplane"]
    with h5py.File(record["path"], "r") as handle:
        group = handle if cell["rank"] == "root" else handle[cell["rank"]]
        raw_volume = float(group["Volume"][cell["rank_index"]])
    print(raw_volume, cell["volume_rsun3"])
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

import h5py
import numpy as np
from dev.datapaths import DATAPATHS, TDE_PARAMETERS, _snapshot_times

import dev
import richio

REQUESTED_TFBS = {
    "1e4": (0.5, 1.0, 1.5, 2.0),
    "1e5": (0.3, 0.5),
    "1e6": (1.0, 1.2, 1.4, 1.5),
}


def scalar(dataset: h5py.Dataset) -> float:
    return float(np.asarray(dataset).squeeze())


def frame_offset(run: str, path: Path, time: float) -> tuple[float, float]:
    needs_offset = (
        path.parent.name == "TEMPTDE"
        if run == "1e6"
        else re.fullmatch(r"snap_\d+\.h5", path.name) is not None
    )
    if not needs_offset:
        return 0.0, 0.0
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    offset = dev.reference_frame_offset(
        t=time * richio.units.tscale,
        Mbh=mbh * richio.units.mscale,
        Mstar=mstar * richio.units.mscale,
        Rstar=rstar * richio.units.lscale,
        beta=1,
    )
    code_length_cm = float((1.0 * richio.units.lscale).to_value("cm"))
    return (
        float(offset[0].to_value("cm")) / code_length_cm,
        float(offset[1].to_value("cm")) / code_length_cm,
    )


def cell_count(handle: h5py.File, groups: list[str]) -> int:
    if groups:
        return sum(handle[f"{group}/X"].shape[0] for group in groups)
    return handle["X"].shape[0]


def scan_snapshot(
    run: str,
    snapnum: int,
    path: Path,
    tfb: float,
    radius_min_rp: float = 0.6,
    radius_max_rp: float = 1.75,
    midplane_diameters: float = 1.0,
) -> dict:
    """Return the minimum-volume cell in the requested cylindrical/midplane cut."""
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    rp = rstar * (mbh / mstar) ** (1.0 / 3.0)
    best: dict[str, float | int | str] | None = None

    with h5py.File(path, "r") as handle:
        time = scalar(handle["Time"])
        x_offset, y_offset = frame_offset(run, path, time)
        groups = sorted(
            (key for key in handle if key.startswith("rank")),
            key=lambda key: int(key[4:]),
        )
        prefixes = [f"{group}/" for group in groups] if groups else [""]
        count = cell_count(handle, groups)

        for prefix in prefixes:
            x = np.asarray(handle[f"{prefix}X"]) + x_offset
            y = np.asarray(handle[f"{prefix}Y"]) + y_offset
            z = np.asarray(handle[f"{prefix}Z"])
            volume = np.asarray(handle[f"{prefix}Volume"])

            cylindrical_r2 = x * x + y * y
            diameter = 2.0 * (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
            selection = (
                (cylindrical_r2 > (radius_min_rp * rp) ** 2)
                & (cylindrical_r2 < (radius_max_rp * rp) ** 2)
                & (np.abs(z) <= midplane_diameters * diameter)
            )
            if not np.any(selection):
                continue
            selected_indices = np.flatnonzero(selection)
            local_index = int(selected_indices[np.argmin(volume[selected_indices])])
            candidate = {
                "selection": (
                    "cylindrical_annulus_and_one_cell_diameter_midplane"
                    if midplane_diameters == 1.0
                    else "cylindrical_annulus_and_cell_diameter_midplane"
                ),
                "radius_min_rp": radius_min_rp,
                "radius_max_rp": radius_max_rp,
                "midplane_diameters": midplane_diameters,
                "volume_rsun3": float(volume[local_index]),
                "diameter_rsun": float(diameter[local_index]),
                "x_rp": float(x[local_index] / rp),
                "y_rp": float(y[local_index] / rp),
                "z_rp": float(z[local_index] / rp),
                "z_rsun": float(z[local_index]),
                "r_sph_rp": float(
                    math.sqrt(cylindrical_r2[local_index] + z[local_index] ** 2) / rp
                ),
                "r_cyl_rp": float(math.sqrt(cylindrical_r2[local_index]) / rp),
                "rank": prefix[:-1] if prefix else "root",
                "rank_index": local_index,
            }
            if best is None or candidate["volume_rsun3"] < best["volume_rsun3"]:
                best = candidate

        if best is None:
            raise RuntimeError(f"No cells in annulus for {path}")

        # Raw code mass and length units are reported as solar units by request.
        prefix = f"{best['rank']}/" if best["rank"] != "root" else ""
        density = float(handle[f"{prefix}Density"][int(best["rank_index"])])
        best["density_code"] = density
        best["mass_msun"] = density * float(best["volume_rsun3"])

    return {
        "run": run,
        "snapnum": snapnum,
        "path": str(path),
        "time_code": time,
        "time_tfb": time / tfb,
        "cell_count": count,
        "rp_rsun": rp,
        "compressed_midplane": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", choices=REQUESTED_TFBS, help="BH mass label")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--snapshot-number",
        type=int,
        action="append",
        help="Exact snapshot; repeat to select several",
    )
    selection.add_argument(
        "--time-tfb",
        type=float,
        action="append",
        help="Nearest fallback-time epoch; repeat to select several",
    )
    selection.add_argument(
        "--input-file", type=Path, help="One explicit RICH HDF5 snapshot"
    )
    parser.add_argument(
        "--include-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final snapshot when selecting by time",
    )
    parser.add_argument(
        "--radius-min-rp",
        type=float,
        default=0.6,
        help="Inner cylindrical radius / r_p",
    )
    parser.add_argument(
        "--radius-max-rp",
        type=float,
        default=1.75,
        help="Outer cylindrical radius / r_p",
    )
    parser.add_argument(
        "--midplane-diameters",
        type=float,
        default=1.0,
        help="Maximum |z| in local equivalent-sphere diameters",
    )
    parser.add_argument(
        "--output", type=Path, help="Persistent JSON Lines output; default stdout"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing --output file"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"Reusing {args.output}; pass --overwrite to recalculate", file=sys.stderr
        )
        return

    run = args.run
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    fallback_time = (
        math.pi / math.sqrt(2.0) * math.sqrt(rstar**3 / mstar) * math.sqrt(mbh / mstar)
    )
    if args.input_file is not None:
        snapnum = int(re.search(r"(\d+)\.h5$", args.input_file.name).group(1))
        selected = [(snapnum, args.input_file)]
    elif args.snapshot_number:
        snapnums, paths = DATAPATHS(run)
        selected = [
            (n, Path(paths[snapnums.index(n)]))
            for n in dict.fromkeys(args.snapshot_number)
        ]
    else:
        snapnums, paths, times = _snapshot_times(run)
        times_in_fallback_units = times / fallback_time
        indices = {
            int(np.argmin(np.abs(times_in_fallback_units - target)))
            for target in (args.time_tfb or REQUESTED_TFBS[run])
        }
        if args.include_last:
            indices.add(len(snapnums) - 1)
        selected = [(snapnums[i], Path(paths[i])) for i in sorted(indices)]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(args.output.suffix + ".part")
        destination = staging.open("w", encoding="utf-8")
    else:
        destination = nullcontext(sys.stdout)
    with destination as stream:
        for snapnum, path in selected:
            result = scan_snapshot(
                run,
                snapnum,
                path,
                fallback_time,
                args.radius_min_rp,
                args.radius_max_rp,
                args.midplane_diameters,
            )
            print(json.dumps(result), file=stream, flush=True)
    if args.output is not None:
        staging.replace(args.output)


if __name__ == "__main__":
    main()
