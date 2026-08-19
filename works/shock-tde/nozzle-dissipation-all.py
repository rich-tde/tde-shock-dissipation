"""Measure nozzle-dissipation proxies in every snapshot of the three TDE runs.

This is the batch version of ``works/shocktubes/0.1-nozzle-dissipation-check.ipynb``.
For every snapshot it applies the notebook's current default selection in the
BH-centred frame: find the maximum-dissipation-density cell, keep cells within
+/-4.5 degrees of its longitude, and require r < 1100 Schwarzschild radii.  No
stellar-debris mask is applied.

Two dimensionless diagnostics are evaluated on that one shared selection:

* total internal energy / total kinetic energy;
* (total dissipation power / total kinetic energy) * (r_p / v_esc,p);
* the notebook's local-velocity proxy for the same rate-based quantity.

One compressed result is written per snapshot so the study can be restarted.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import typer
import unyt as u
from loguru import logger

import dev
import richio
from dev.datapaths import DATAPATHS, TDE_PARAMETERS


app = typer.Typer(add_completion=False)
OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/NozzleDissipationComparison/"
    "max-dissipation-nozzle"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
ANGULAR_HALF_WIDTH_DEG = 4.5
RADIAL_LIMIT_RS = 1100.0
STAR_ONLY = False


def scalar_time(snap):
    """Return snapshot time as a scalar code-time quantity."""

    return u.unyt_quantity(
        float(np.asarray(snap.time.to_value("code_time")).squeeze()),
        "code_time",
        registry=snap.time.units.registry,
    )


def run_scales(run: str):
    """Return TDE scales in the RICH code-unit registry."""

    mbh_value, mstar_value, rstar_value = TDE_PARAMETERS[run]
    mbh = mbh_value * richio.units.mscale
    mstar = mstar_value * richio.units.mscale
    rstar = rstar_value * richio.units.lscale
    r_p = rstar * (mbh / mstar) ** (1 / 3)
    v_esc = np.sqrt(2 * u.G * mbh / r_p)
    crossing_time = r_p / v_esc
    fallback_time = (
        np.pi
        / np.sqrt(2)
        * np.sqrt(rstar**3 / (u.G * mstar))
        * np.sqrt(mbh / mstar)
    )
    return mbh, mstar, rstar, r_p, v_esc, crossing_time, fallback_time


def needs_frame_switch(run: str, snapshot_path: Path) -> bool:
    """Match the established comoving-to-BH-frame convention in ``E-t.py``."""

    if run == "1e6":
        return snapshot_path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", snapshot_path.name) is not None


def bh_frame_coordinates_and_velocity(snap, run: str, snapshot_path: Path, scales):
    """Return position and velocity components in the BH frame."""

    x, y, z = snap.X, snap.Y, snap.Z
    vx, vy, vz = snap.vx, snap.vy, snap.vz
    switched = needs_frame_switch(run, snapshot_path)
    if switched:
        mbh, mstar, rstar = scales[:3]
        offset = dev.reference_frame_offset(
            t=scalar_time(snap),
            Mbh=mbh,
            Mstar=mstar,
            Rstar=rstar,
            beta=1,
        )
        x, y = x + offset[0], y + offset[1]
        vx, vy = vx + offset[2], vy + offset[3]
    return x, y, z, vx, vy, vz, switched


def wrapped_angular_distance(angle, center):
    """Signed angular distance in [-pi, pi), including across the branch cut."""

    return (angle - center + np.pi) % (2 * np.pi) - np.pi


def max_dissipation_nozzle_mask(
    radius,
    azimuth,
    dissipation_density,
    schwarzschild_radius,
    maximum_radius_rs=RADIAL_LIMIT_RS,
    angular_half_width_deg=ANGULAR_HALF_WIDTH_DEG,
):
    """Select the radial nozzle wedge centred on maximum dissipation density."""

    maximum_index = int(np.argmax(dissipation_density))
    center = float(azimuth[maximum_index])
    angular_cut = np.abs(wrapped_angular_distance(azimuth, center)) < np.deg2rad(
        angular_half_width_deg
    )
    radial_cut = radius < maximum_radius_rs * schwarzschild_radius
    return radial_cut & angular_cut, maximum_index, center


def sums_and_ratios(
    cell_mass,
    speed_squared,
    kinetic_energy,
    internal_energy,
    dissipation_power,
    selection,
    maximum_dissipation_index,
    crossing_time,
):
    """Calculate the notebook's sums and three dimensionless diagnostics."""

    mass = np.sum(cell_mass[selection]).to("g")
    kinetic = np.sum(kinetic_energy[selection]).to("erg")
    internal = np.sum(internal_energy[selection]).to("erg")
    power = np.sum(dissipation_power[selection]).to("erg/s")
    internal_over_kinetic = (internal / kinetic).to_value("dimensionless")
    rate_summed_kinetic = ((power / kinetic) * crossing_time).to_value("dimensionless")
    local_specific_kinetic = 0.5 * speed_squared[maximum_dissipation_index]
    deposited_specific_energy = power * crossing_time / mass
    rate_local_velocity = (
        deposited_specific_energy / local_specific_kinetic
    ).to_value("dimensionless")
    return {
        "mass_g": float(mass),
        "kinetic_erg": float(kinetic),
        "internal_erg": float(internal),
        "dissipation_power_erg_s": float(power),
        "internal_over_kinetic": float(internal_over_kinetic),
        "rate_fraction_summed_kinetic": float(rate_summed_kinetic),
        "rate_fraction_local_velocity": float(rate_local_velocity),
    }


def analyse_snapshot(run: str, snapnum: int, snapshot_path: Path):
    """Load and analyse one snapshot, returning scalar output columns."""

    scales = run_scales(run)
    mbh, _, _, r_p, v_esc, crossing_time, fallback_time = scales
    schwarzschild_radius = 2 * u.G * mbh / u.c**2
    snap = richio.load(snapshot_path)
    time = scalar_time(snap)
    x, y, z, vx, vy, vz, frame_switched = bh_frame_coordinates_and_velocity(
        snap, run, snapshot_path, scales
    )

    keep = (
        np.asarray(snap.mask_star_ratio(), dtype=bool)
        if STAR_ONLY
        else np.ones(len(snap), dtype=bool)
    )
    cell_mass = (snap.density * snap.volume)[keep]
    speed_squared = vx[keep] ** 2 + vy[keep] ** 2 + vz[keep] ** 2
    kinetic_energy = 0.5 * cell_mass * speed_squared
    internal_energy = snap.sie[keep] * cell_mass
    dissipation_density = snap.dissipation[keep]
    dissipation_power = dissipation_density * snap.volume[keep]
    radius = np.sqrt(x[keep] ** 2 + y[keep] ** 2 + z[keep] ** 2)
    azimuth = np.arctan2(np.asarray(y[keep]), np.asarray(x[keep]))

    nozzle_selection, maximum_index, selection_center = max_dissipation_nozzle_mask(
        radius,
        azimuth,
        dissipation_density,
        schwarzschild_radius,
    )
    selection_valid = bool(np.any(nozzle_selection))
    if selection_valid:
        nozzle = sums_and_ratios(
            cell_mass,
            speed_squared,
            kinetic_energy,
            internal_energy,
            dissipation_power,
            nozzle_selection,
            maximum_index,
            crossing_time,
        )
    else:
        logger.warning(
            "Empty nozzle selection for {}: maximum dissipation is at "
            "r={:.1f} R_s; recording NaN diagnostics",
            snapshot_path,
            float(
                (radius[maximum_index] / schwarzschild_radius).to_value(
                    "dimensionless"
                )
            ),
        )
        nozzle = {
            "mass_g": 0.0,
            "kinetic_erg": 0.0,
            "internal_erg": 0.0,
            "dissipation_power_erg_s": 0.0,
            "internal_over_kinetic": np.nan,
            "rate_fraction_summed_kinetic": np.nan,
            "rate_fraction_local_velocity": np.nan,
        }

    output = {
        "run": run,
        "snapnum": snapnum,
        "snap_path": str(snapshot_path),
        "time_code": float(time.to_value("code_time")),
        "time_tfb": float((time / fallback_time).to_value("dimensionless")),
        "frame_switched": frame_switched,
        "n_cells": len(snap),
        "star_only": STAR_ONLY,
        "n_kept": int(keep.sum()),
        "n_nozzle": int(nozzle_selection.sum()),
        "selection_valid": selection_valid,
        "selection_center_rad": selection_center,
        "selection_center_deg": float(np.rad2deg(selection_center)),
        "maximum_dissipation_radius_rs": float(
            (radius[maximum_index] / schwarzschild_radius).to_value("dimensionless")
        ),
        "angular_half_width_deg": ANGULAR_HALF_WIDTH_DEG,
        "radial_limit_rs": RADIAL_LIMIT_RS,
        "schwarzschild_radius_cm": float(schwarzschild_radius.to_value("cm")),
        "r_p_cm": float(r_p.to_value("cm")),
        "v_esc_cm_s": float(v_esc.to_value("cm/s")),
        "crossing_time_s": float(crossing_time.to_value("s")),
    }
    output.update({f"nozzle_{key}": value for key, value in nozzle.items()})
    if selection_valid:
        output["nozzle_mass_fraction"] = float(
            np.sum(cell_mass[nozzle_selection]) / np.sum(cell_mass)
        )
        output["nozzle_kinetic_fraction"] = float(
            np.sum(kinetic_energy[nozzle_selection]) / np.sum(kinetic_energy)
        )
    else:
        output["nozzle_mass_fraction"] = 0.0
        output["nozzle_kinetic_fraction"] = 0.0
    return output


def save_result(output: dict, output_path: Path):
    """Atomically write one scalar-only compressed result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **output)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


@app.command()
def main(
    mode: int = typer.Option(..., min=1, max=3, help="Run 1e4, 1e5, or 1e6"),
    snapshot_index: int | None = typer.Option(
        None,
        min=0,
        help="Process only this zero-based snapshot position (default: all in run).",
    ),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Root directory for per-run result files."
    ),
    overwrite: bool = typer.Option(False, help="Replace existing per-snapshot results."),
):
    """Process all snapshots in one run, skipping completed outputs by default."""

    run = RUN_BY_MODE[mode]
    snapnums, paths = DATAPATHS(run)
    items = list(zip(snapnums, paths))
    if snapshot_index is not None:
        if snapshot_index >= len(items):
            raise typer.BadParameter(
                f"--snapshot-index must be between 0 and {len(items) - 1} for {run}"
            )
        items = [items[snapshot_index]]

    output_dir = output_root / run
    logger.info(f"Processing {len(items)} of {len(paths)} snapshots for {run}")
    for position, (snapnum, snapshot_path) in enumerate(items, start=1):
        output_path = output_dir / f"nozzle_dissipation_snap_{snapnum:04d}.npz"
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping existing {output_path}")
            continue
        logger.info(
            f"[{position}/{len(items)}] {run} snapshot {snapnum}: {snapshot_path}"
        )
        output = analyse_snapshot(run, snapnum, snapshot_path)
        save_result(output, output_path)
        logger.info(
            "Saved {}: Eint/Ekin={:.4e}, rate-summed-KE={:.4e}, "
            "rate-local-velocity={:.4e}",
            output_path,
            output["nozzle_internal_over_kinetic"],
            output["nozzle_rate_fraction_summed_kinetic"],
            output["nozzle_rate_fraction_local_velocity"],
        )


if __name__ == "__main__":
    app()
