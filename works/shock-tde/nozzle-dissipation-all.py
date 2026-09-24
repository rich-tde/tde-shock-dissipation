"""Compare nozzle internal energy and dissipation-based kinetic-energy fractions.

Transform positions and velocities to the BH frame; retain ``star>0.99``
by default; find the maximum volumetric-dissipation cell in that material;
select a longitude wedge within +/-4.5 degrees of it and ``r<1100 R_s``.
On that shared selection calculate ``Eint/Ekin``, ``P_diss*t_cross/Ekin``,
and a proxy replacing the selected mass-mean kinetic energy with that at
the maximum-dissipation cell, where ``t_cross=r_p/v_esc,p``. Thresholds are
CLI options. This wedge need not isolate a nozzle shock at every epoch.
Accumulated internal energy differs from instantaneous heating; none of
these ratios alone measures radiative energy loss.

Input files
-----------
Raw snapshot HDF5 files selected by ``dev.datapaths.DATAPATHS`` for mode
``1=1e4``, ``2=1e5``, ``3=1e6`` solar-mass BH runs. Exact directories and
restart exclusions are in ``dev/dev/datapaths.py``. ``--snapshot-index`` is
a zero-based position in the catalogue, not a snapshot number. Alternatively,
``--snapshot-file`` selects one explicit ``snap_N.h5`` or ``snap_full_N.h5``
from the chosen run; filename/parent directory determine frame correction.
Required fields read through ``richio.load``: positions, velocities, density,
volume, specific internal energy, volumetric dissipation, stellar tracer and
time. If no cells exceed ``--star-min``, the calculation raises an error.

Output files
------------
``RUN/nozzle_dissipation_snap_NNNN.npz`` under
``/home/hey4/rich_tde/data/processed/NozzleDissipationComparison/``
``max-dissipation-nozzle-star-0.99/`` (override with ``--output-root``).
``NNNN`` is the zero-padded snapshot number. Each compressed NumPy archive
contains scalar arrays only: every key has shape ``()``, not ``(1,)`` or a
row in a rectangular table. Load with ``np.load`` and extract with ``.item()``.
All numbers are linear, not logarithmic. Every saved key is listed below.

``run``, ``snap_path`` : Unicode
    Run label and original HDF5 snapshot path.
``snapnum``, ``n_cells``, ``n_kept``, ``n_nozzle`` : int64
    Snapshot number; original cell count; cells passing the stellar tracer
    cut; cells in the final nozzle wedge, respectively.
``frame_switched``, ``selection_valid`` : bool
    Whether the BH-frame correction was applied; whether the wedge contains
    at least one cell. Validity records nonemptiness, not physical proof of
    nozzle-shock identification.
``time_code``, ``time_tfb`` : float64
    Snapshot time in code units (1603 s per unit), and time/fallback time.
``star_min`` : float64
    Dimensionless strict stellar-tracer threshold (default 0.99).
``selection_center_rad``, ``selection_center_deg`` : float64
    Azimuth of the maximum volumetric-dissipation cell in radians/degrees.
``angular_half_width_deg`` : float64
    Longitude-wedge half-width in degrees (strict angular cut).
``maximum_dissipation_radius_rs``, ``radial_limit_rs`` : float64
    Maximum-dissipation cell radius and wedge outer cutoff, divided by
    Schwarzschild radius ``R_s``. The radial cut is strict.
``schwarzschild_radius_cm``, ``r_p_cm`` : float64
    Schwarzschild radius and nominal pericentre radius in cm.
``v_esc_cm_s``, ``crossing_time_s`` : float64
    Escape speed at pericentre [cm/s] and ``r_p/v_esc,p`` [s].
``nozzle_mass_g`` : float64
    Total selected wedge mass [g].
``nozzle_kinetic_erg``, ``nozzle_internal_erg`` : float64
    Selected BH-frame kinetic energy and internal energy [erg].
``nozzle_dissipation_power_erg_s`` : float64
    Sum of volumetric dissipation times cell volume in the wedge [erg/s].
``nozzle_internal_over_kinetic`` : float64
    Dimensionless ``Eint/Ekin`` for the selected wedge.
``nozzle_rate_fraction_summed_kinetic`` : float64
    Dimensionless ``P_diss*t_cross/Ekin`` for the selected wedge.
``nozzle_rate_fraction_local_velocity`` : float64
    Dimensionless ``(P_diss*t_cross/M_wedge)/(0.5*v_peak**2)``. The speed
    is at the maximum-dissipation cell, which need not lie inside the wedge
    if that peak fails the radial cut.
``nozzle_mass_fraction``, ``nozzle_kinetic_fraction`` : float64
    Wedge mass/kinetic energy divided by that of all cells passing the
    stellar cut, respectively. Both are dimensionless.

For an empty wedge, ``selection_valid=False``; mass, energies, power and
fractions are zero, while the three energy-ratio diagnostics are NaN.
Nonempty selections have no further finite-value mask; a zero denominator
can also produce nonfinite ratios. No per-cell masks/arrays are saved.
Older ``NozzleDissipationComparison/RUN/`` outputs use a different ``beam_*``
schema and should not be confused with this default subdirectory.

Usage
-----
Run from ``/home/hey4/rich_tde`` with the richanalysis Python environment::

    python works/shock-tde/nozzle-dissipation-all.py --mode 1
    python works/shock-tde/nozzle-dissipation-all.py --mode 1 --snapshot-index 0
    python works/shock-tde/nozzle-dissipation-all.py --mode 1 --angular-half-width-deg 6 --radial-limit-rs 900 --output-root data/processed/NozzleDissipationComparison/wide-wedge

Matching existing source/selection outputs are skipped. Incompatible files
require ``--overwrite`` or another ``--output-root``. Each snapshot is saved
atomically beside its destination, so interrupted runs resume independently.
The directory name of a custom root does not select parameters; set the
corresponding CLI options explicitly.

Loading examples
----------------
Read the first result and recover the dissipation fraction from saved sums::

    from pathlib import Path
    import numpy as np

    root = Path("data/processed/NozzleDissipationComparison")
    root = root / "max-dissipation-nozzle-star-0.99/1e4"
    path = sorted(root.glob("nozzle_dissipation_snap_*.npz"))[0]
    with np.load(path) as data:
        valid = data["selection_valid"].item()
        kinetic = data["nozzle_kinetic_erg"].item()
        power = data["nozzle_dissipation_power_erg_s"].item()
        crossing_time = data["crossing_time_s"].item()
        saved_fraction = data["nozzle_rate_fraction_summed_kinetic"].item()
        time_tfb = data["time_tfb"].item()
    if valid and kinetic > 0:
        fraction = power * crossing_time / kinetic
        print(time_tfb, fraction, saved_fraction)  # Dimensionless values.
    else:
        print("Empty wedge or zero kinetic energy; inspect another snapshot.")
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

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

from dev.datapaths import DATAPATHS, TDE_PARAMETERS

import dev
import richio

app = typer.Typer(add_completion=False)
OUTPUT_ROOT = Path(
    "/home/hey4/rich_tde/data/processed/NozzleDissipationComparison/"
    "max-dissipation-nozzle-star-0.99"
)
RUN_BY_MODE = {1: "1e4", 2: "1e5", 3: "1e6"}
ANGULAR_HALF_WIDTH_DEG = 4.5
RADIAL_LIMIT_RS = 1100.0
STAR_MIN = 0.99


def scalar_time(snap):
    """Return snapshot time as a scalar code-time quantity."""

    return u.unyt_quantity(
        float(np.asarray(snap.time.to_value("code_time")).squeeze()),
        "code_time",
        registry=snap.time.units.registry,
    )


def run_scales(run: str):
    """Return TDE scales in the RICH code-unit registry."""

    black_hole_mass_value, stellar_mass_value, stellar_radius_value = TDE_PARAMETERS[
        run
    ]
    black_hole_mass = black_hole_mass_value * richio.units.mscale
    stellar_mass = stellar_mass_value * richio.units.mscale
    stellar_radius = stellar_radius_value * richio.units.lscale
    pericenter_radius = stellar_radius * (black_hole_mass / stellar_mass) ** (1 / 3)
    escape_speed = np.sqrt(2 * u.G * black_hole_mass / pericenter_radius)
    crossing_time = pericenter_radius / escape_speed
    fallback_time = (
        np.pi
        / np.sqrt(2)
        * np.sqrt(stellar_radius**3 / (u.G * stellar_mass))
        * np.sqrt(black_hole_mass / stellar_mass)
    )
    return (
        black_hole_mass,
        stellar_mass,
        stellar_radius,
        pericenter_radius,
        escape_speed,
        crossing_time,
        fallback_time,
    )


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
        black_hole_mass, stellar_mass, stellar_radius = scales[:3]
        offset = dev.reference_frame_offset(
            t=scalar_time(snap),
            Mbh=black_hole_mass,
            Mstar=stellar_mass,
            Rstar=stellar_radius,
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
    rate_local_velocity = (deposited_specific_energy / local_specific_kinetic).to_value(
        "dimensionless"
    )
    return {
        "mass_g": float(mass),
        "kinetic_erg": float(kinetic),
        "internal_erg": float(internal),
        "dissipation_power_erg_s": float(power),
        "internal_over_kinetic": float(internal_over_kinetic),
        "rate_fraction_summed_kinetic": float(rate_summed_kinetic),
        "rate_fraction_local_velocity": float(rate_local_velocity),
    }


def analyse_snapshot(
    run: str,
    snapnum: int,
    snapshot_path: Path,
    star_min=STAR_MIN,
    angular_half_width_deg=ANGULAR_HALF_WIDTH_DEG,
    radial_limit_rs=RADIAL_LIMIT_RS,
):
    """Load and analyse one snapshot, returning scalar output columns."""

    scales = run_scales(run)
    (
        black_hole_mass,
        _,
        _,
        pericenter_radius,
        escape_speed,
        crossing_time,
        fallback_time,
    ) = scales
    schwarzschild_radius = 2 * u.G * black_hole_mass / u.c**2
    snap = richio.load(snapshot_path)
    time = scalar_time(snap)
    x, y, z, vx, vy, vz, frame_switched = bh_frame_coordinates_and_velocity(
        snap, run, snapshot_path, scales
    )

    keep = np.asarray(snap.star > star_min, dtype=bool)
    if not np.any(keep):
        raise ValueError(f"No cells have star tracer > {star_min} in {snapshot_path}")
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
        maximum_radius_rs=radial_limit_rs,
        angular_half_width_deg=angular_half_width_deg,
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
                (radius[maximum_index] / schwarzschild_radius).to_value("dimensionless")
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
        "star_min": star_min,
        "n_kept": int(keep.sum()),
        "n_nozzle": int(nozzle_selection.sum()),
        "selection_valid": selection_valid,
        "selection_center_rad": selection_center,
        "selection_center_deg": float(np.rad2deg(selection_center)),
        "maximum_dissipation_radius_rs": float(
            (radius[maximum_index] / schwarzschild_radius).to_value("dimensionless")
        ),
        "angular_half_width_deg": angular_half_width_deg,
        "radial_limit_rs": radial_limit_rs,
        "schwarzschild_radius_cm": float(schwarzschild_radius.to_value("cm")),
        "r_p_cm": float(pericenter_radius.to_value("cm")),
        "v_esc_cm_s": float(escape_speed.to_value("cm/s")),
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
    snapshot_file: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="One explicit snapshot from this mode; cannot combine with --snapshot-index.",
    ),
    star_min: float = typer.Option(
        STAR_MIN,
        min=0,
        max=1,
        help="Require stellar tracer strictly above this threshold.",
    ),
    angular_half_width_deg: float = typer.Option(
        ANGULAR_HALF_WIDTH_DEG,
        min=0,
        max=180,
        help="Longitude wedge half-width in degrees.",
    ),
    radial_limit_rs: float = typer.Option(
        RADIAL_LIMIT_RS, min=0, help="Outer wedge radius in Schwarzschild radii."
    ),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Root directory for per-run result files."
    ),
    overwrite: bool = typer.Option(
        False, help="Replace existing per-snapshot results."
    ),
):
    """Process all snapshots in one run, skipping completed outputs by default."""

    run = RUN_BY_MODE[mode]
    if snapshot_file is not None:
        if snapshot_index is not None:
            raise typer.BadParameter(
                "Use --snapshot-file or --snapshot-index, not both"
            )
        match = re.fullmatch(r"snap_(?:full_)?(\d+)\.h5", snapshot_file.name)
        if match is None:
            raise typer.BadParameter(
                "--snapshot-file must be snap_<n>.h5 or snap_full_<n>.h5"
            )
        snapnums, paths = [int(match.group(1))], [snapshot_file.resolve()]
    else:
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
            with np.load(output_path) as cached:
                compatible = (
                    cached["run"].item() == run
                    and Path(cached["snap_path"].item()).resolve()
                    == Path(snapshot_path).resolve()
                    and float(cached["star_min"]) == star_min
                    and float(cached["angular_half_width_deg"])
                    == angular_half_width_deg
                    and float(cached["radial_limit_rs"]) == radial_limit_rs
                )
            if not compatible:
                raise ValueError(
                    f"{output_path} has different input/selection settings; use --overwrite or another --output-root"
                )
            logger.info(f"Skipping existing {output_path}")
            continue
        logger.info(
            f"[{position}/{len(items)}] {run} snapshot {snapnum}: {snapshot_path}"
        )
        output = analyse_snapshot(
            run,
            snapnum,
            snapshot_path,
            star_min,
            angular_half_width_deg,
            radial_limit_rs,
        )
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
