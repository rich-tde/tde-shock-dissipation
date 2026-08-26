#!/usr/bin/env python3
"""Dissipation time series with the near-pericenter region split in two.

This is intentionally separate from ``Ediss-t.py`` and writes to a dedicated
processed-data directory.  The four reported regions exclude ``x < -r_a``.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import typer
import unyt as u
from loguru import logger

import dev
import richio


app = typer.Typer()
OUTPUT_DIR = Path("/home/hey4/rich_tde/data/processed/EdissFourRegions")


@dataclass(frozen=True)
class ModeSettings:
    label: str
    datadirs: tuple[str, ...]
    rstar: u.unyt_quantity
    mstar: u.unyt_quantity
    mbh: u.unyt_quantity
    minimum_tfb: float
    cadence: int = 1


def mode_settings(mode: int) -> ModeSettings:
    if mode == 1:
        return ModeSettings(
            label="1e4",
            datadirs=(
                "/data1/projects/pi-rossiem/TDE_data/NewSnellius/"
                "R0.47M0.5BH10000beta1S60ComptonHiRes",
            ),
            rstar=0.47 * richio.units.lscale,
            mstar=0.5 * richio.units.mscale,
            mbh=1e4 * richio.units.mscale,
            minimum_tfb=0.1,
        )
    if mode == 2:
        return ModeSettings(
            label="1e5",
            datadirs=(
                "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/"
                "R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
            ),
            rstar=0.47 * richio.units.lscale,
            mstar=0.5 * richio.units.mscale,
            mbh=1e5 * richio.units.mscale,
            minimum_tfb=0.1,
        )
    if mode == 3:
        return ModeSettings(
            label="1e6",
            datadirs=(
                "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
                "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
                "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
            ),
            rstar=1.0 * richio.units.lscale,
            mstar=1.0 * richio.units.mscale,
            mbh=1e6 * richio.units.mscale,
            minimum_tfb=0.7,
        )
    raise ValueError("Invalid mode. Please choose 1, 2, or 3.")


def snapshot_files(datadir: str) -> list[str]:
    full = sorted(
        glob.glob(os.path.join(datadir, "snap_full_*.h5")),
        key=lambda path: int(re.search(r"snap_full_(\d+)\.h5", path).group(1)),
    )
    plain = [
        path
        for path in glob.glob(os.path.join(datadir, "snap_*.h5"))
        if re.fullmatch(r"snap_\d+\.h5", os.path.basename(path))
    ]
    plain.sort(key=lambda path: int(re.search(r"snap_(\d+)\.h5", path).group(1)))
    return full + plain


def snapshot_number(path: str) -> int:
    match = re.search(r"snap_(?:full_)?(\d+)\.h5", path)
    if match is None:
        raise ValueError(f"Cannot parse snapshot number from {path}")
    return int(match.group(1))


def selected_files(
    cfg: ModeSettings,
    tmin: u.unyt_quantity,
    npoints: int | None,
) -> list[tuple[str, str]]:
    candidates = []
    for datadir in cfg.datadirs:
        for snap_file in snapshot_files(datadir)[:: cfg.cadence]:
            snapnum = snapshot_number(snap_file)
            if os.path.basename(datadir) == "TEMPTDE4" and snapnum >= 826:
                continue
            candidates.append((datadir, snap_file))

    if npoints is None:
        return candidates
    if npoints < 2:
        raise ValueError("npoints must be at least 2")

    eligible = []
    tmin_code = float(tmin.to_value("code_time"))
    for datadir, snap_file in candidates:
        with h5py.File(snap_file) as handle:
            time_code = float(np.asarray(handle["Time"]).squeeze())
        if time_code / tmin_code >= cfg.minimum_tfb:
            eligible.append((time_code, datadir, snap_file))
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < npoints:
        raise ValueError(
            f"Only {len(eligible)} snapshots satisfy t/t_fb >= {cfg.minimum_tfb}"
        )

    indices = np.rint(np.linspace(0, len(eligible) - 1, npoints)).astype(int)
    return [(eligible[index][1], eligible[index][2]) for index in indices]


def save_timeseries(path: Path, columns: list[list], complete: bool) -> None:
    suffix = ".tmp" if complete else ".partial.tmp"
    destination = path if complete else path.with_suffix(path.suffix + ".partial")
    temporary = path.with_suffix(path.suffix + suffix)
    u.savetxt(
        temporary,
        arrays=[u.unyt_array(column) for column in columns],
        header=(
            "SNAPNUM\tTIME\tTFALLBACK\tEDISS_NOZZLE\tEDISS_STREAM_DISK\t"
            "EDISS_OUTGOING\tEDISS_INCOMING"
        ),
        footer=(
            "radius = sqrt(X**2 + Y**2 + Z**2)\n"
            "nozzle = (X > 0) & (radius < 3*r_p)\n"
            "stream_disk = (X > 0) & (radius >= 3*r_p)\n"
            "outgoing = (X > -r_a) & (X < 0) & (Y < 0)\n"
            "incoming = (X > -r_a) & (X < 0) & (Y > 0)\n"
            "X < -r_a is excluded"
        ),
    )
    os.replace(temporary, destination)


@app.command()
def main(
    mode: int = typer.Option(..., help="Run 1e4, 1e5, or 1e6"),
    npoints: int | None = typer.Option(
        None, help="Evenly sample this many snapshots across the plotted time window"
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing completed result"),
):
    cfg = mode_settings(mode)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_suffix = "" if npoints is None else f"-n{npoints}"
    output_file = OUTPUT_DIR / f"Ediss-t-four-regions-{cfg.label}{sample_suffix}.txt"
    if output_file.exists() and not overwrite:
        logger.info(f"Completed output exists; skipping: {output_file}")
        return

    r_amin = cfg.rstar * (cfg.mbh / cfg.mstar) ** (2 / 3)
    r_p = cfg.rstar * (cfg.mbh / cfg.mstar) ** (1 / 3)
    tmin = (
        np.pi
        / np.sqrt(2)
        * (cfg.rstar**3 / u.G / cfg.mstar) ** (1 / 2)
        * (cfg.mbh / cfg.mstar) ** (1 / 2)
    )

    files = selected_files(cfg, tmin, npoints)
    logger.info(
        "Selected {} snapshots spanning the requested plotted window", len(files)
    )

    columns: list[list] = [[] for _ in range(7)]
    previous_datadir = None
    for datadir, snap_file in files:
        if datadir != previous_datadir:
            logger.info(f"Processing directory: {datadir}")
            previous_datadir = datadir
        snapnum = snapshot_number(snap_file)
        snap = richio.load(snap_file)
        try:
            time = snap.t[0]
        except IndexError:
            time = snap.t
        tfb = time / tmin
        r_a = r_p if time < 0 else r_amin * tfb ** (2 / 3)

        if mode == 3:
            needs_switch = os.path.basename(datadir) == "TEMPTDE"
        else:
            needs_switch = bool(
                re.fullmatch(r"snap_\d+\.h5", os.path.basename(snap_file))
            )
        if needs_switch:
            offset = dev.reference_frame_offset(
                t=time,
                Mbh=cfg.mbh,
                Mstar=cfg.mstar,
                Rstar=cfg.rstar,
                beta=1,
            )
            x, y = snap.X + offset[0], snap.Y + offset[1]
        else:
            x, y = snap.X, snap.Y

        radius = np.sqrt(x**2 + y**2 + snap.Z**2)
        masks = (
            (x > 0) & (radius < 3 * r_p),
            (x > 0) & (radius >= 3 * r_p),
            (x > -r_a) & (x < 0) & (y < 0),
            (x > -r_a) & (x < 0) & (y > 0),
        )
        overlaps = any(
            np.any(left & right)
            for i, left in enumerate(masks)
            for right in masks[i + 1 :]
        )
        if overlaps:
            raise RuntimeError(f"Region masks overlap in snapshot {snap_file}")

        dissipation_power = snap.dissipation * snap.volume
        region_power = [np.sum(dissipation_power[mask]) for mask in masks]
        values = (snapnum, time, tfb, *region_power)
        for column, value in zip(columns, values):
            column.append(value)
        save_timeseries(output_file, columns, complete=False)
        logger.info(
            "{} {} {} nozzle={} stream_disk={} outgoing={} incoming={}",
            snapnum,
            time,
            tfb,
            *region_power,
        )

    save_timeseries(output_file, columns, complete=True)
    partial = output_file.with_suffix(output_file.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    logger.info(f"Saved {len(columns[0])} snapshots to {output_file}")


if __name__ == "__main__":
    app()
