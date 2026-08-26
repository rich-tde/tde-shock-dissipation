#!/usr/bin/env python3
"""Measure native-cell resolution in the compressed nozzle midplane."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np

import dev
import richio
from dev.datapaths import TDE_PARAMETERS, _snapshot_times


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


def scan_snapshot(run: str, snapnum: int, path: Path, tfb: float) -> dict:
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    rp = rstar * (mbh / mstar) ** (1.0 / 3.0)
    best: dict[str, float | int | str] | None = None

    with h5py.File(path, "r") as handle:
        time = scalar(handle["Time"])
        dx, dy = frame_offset(run, path, time)
        groups = sorted(
            (key for key in handle if key.startswith("rank")),
            key=lambda key: int(key[4:]),
        )
        prefixes = [f"{group}/" for group in groups] if groups else [""]
        count = cell_count(handle, groups)

        for prefix in prefixes:
            x = np.asarray(handle[f"{prefix}X"]) + dx
            y = np.asarray(handle[f"{prefix}Y"]) + dy
            z = np.asarray(handle[f"{prefix}Z"])
            volume = np.asarray(handle[f"{prefix}Volume"])

            cylindrical_r2 = x * x + y * y
            diameter = 2.0 * (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
            select = (
                (cylindrical_r2 > (0.6 * rp) ** 2)
                & (cylindrical_r2 < (1.75 * rp) ** 2)
                & (np.abs(z) <= diameter)
            )
            if not np.any(select):
                continue
            selected_indices = np.flatnonzero(select)
            local = int(selected_indices[np.argmin(volume[selected_indices])])
            candidate = {
                "selection": "cylindrical_annulus_and_one_cell_diameter_midplane",
                "volume_rsun3": float(volume[local]),
                "diameter_rsun": float(diameter[local]),
                "x_rp": float(x[local] / rp),
                "y_rp": float(y[local] / rp),
                "z_rp": float(z[local] / rp),
                "z_rsun": float(z[local]),
                "r_sph_rp": float(math.sqrt(cylindrical_r2[local] + z[local] ** 2) / rp),
                "r_cyl_rp": float(math.sqrt(cylindrical_r2[local]) / rp),
                "rank": prefix[:-1] if prefix else "root",
                "rank_index": local,
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


def main(run: str) -> None:
    snapnums, paths, times = _snapshot_times(run)
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    tfb = math.pi / math.sqrt(2.0) * math.sqrt(rstar**3 / mstar) * math.sqrt(mbh / mstar)
    time_tfbs = times / tfb
    indices = sorted(
        {int(np.argmin(np.abs(time_tfbs - target))) for target in REQUESTED_TFBS[run]}
        | {len(snapnums) - 1}
    )
    for index in indices:
        result = scan_snapshot(run, snapnums[index], Path(paths[index]), tfb)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
