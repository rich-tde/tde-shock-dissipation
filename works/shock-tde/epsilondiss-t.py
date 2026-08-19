import glob
import os
import re
from collections import Counter

import numpy as np
import typer
import unyt as u
from loguru import logger

import dev
import richio

app = typer.Typer()

EPSILONDISS_UNIT = "code_length**2/code_time**3"
OUTPUT_HEADER = (
    "SNAPNUM\tTIME\tTFALLBACK\tEPSILONDISS1\tEPSILONDISS2\t"
    "EPSILONDISS3\tEPSILONDISS4"
)
OUTPUT_FOOTER = (
    "EPSILONDISS = sum(dissipation * volume) / "
    "sum(density * volume) in each region\n"
    "NaN means that the corresponding region contains no positive mass\n"
    "shock1_cut = X > 0\n"
    "shock2_cut = (X > -r_a) & (X < 0) & (Y < 0)\n"
    "shock3_cut = (X > -r_a) & (X < 0) & (Y > 0)\n"
    "shock4_cut = X < -r_a"
)


def load_existing_output(output_file):
    """Load and validate a partial output so the study can resume safely."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return ([], [], [], [], [], [], [])

    raw = np.atleast_2d(np.loadtxt(output_file, delimiter="\t"))
    if raw.shape[1] != 7:
        raise ValueError(
            f"{output_file} has {raw.shape[1]} columns; expected exactly 7"
        )
    if not np.isfinite(raw[:, :3]).all():
        raise ValueError(f"{output_file} has a non-finite snapshot, time, or t/t_fb")
    if np.isinf(raw[:, 3:]).any():
        raise ValueError(f"{output_file} has infinite specific-dissipation values")

    snapnums = raw[:, 0].astype(int).tolist()
    if not np.array_equal(raw[:, 0], snapnums):
        raise ValueError(f"{output_file} has a non-integer snapshot number")

    registry = richio.units.registry
    ts = list(u.unyt_array(raw[:, 1], "code_time", registry=registry))
    tfbs = list(u.unyt_array(raw[:, 2]))
    epsilondiss = u.unyt_array(raw[:, 3:7], EPSILONDISS_UNIT, registry=registry)
    return (
        snapnums,
        ts,
        tfbs,
        list(epsilondiss[:, 0]),
        list(epsilondiss[:, 1]),
        list(epsilondiss[:, 2]),
        list(epsilondiss[:, 3]),
    )


def save_output_atomic(output_file, arrays):
    """Write a complete checkpoint without exposing a partially written file."""
    temporary_file = f"{output_file}.tmp"
    u.savetxt(
        temporary_file,
        arrays=[u.unyt_array(array) for array in arrays],
        header=OUTPUT_HEADER,
        footer=OUTPUT_FOOTER,
    )
    os.replace(temporary_file, output_file)


@app.command()
def main(mode: int = typer.Option(..., help="Run 1e4, 1e5, or 1e6")):
    if mode == 1:
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes",
        )
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/epsilondiss-t/"
            "epsilondiss-t-1e4-final.txt"
        )
        NCADENCE = 1
        Rstar = 0.47 * richio.units.lscale
        Mstar = 0.5 * richio.units.mscale
        Mbh = 1e4 * richio.units.mscale
    elif mode == 2:
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/epsilondiss-t/"
            "epsilondiss-t-1e5-final.txt"
        )
        NCADENCE = 1
        Rstar = 0.47 * richio.units.lscale
        Mstar = 0.5 * richio.units.mscale
        Mbh = 1e5 * richio.units.mscale
    elif mode == 3:
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
        )
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/epsilondiss-t/"
            "epsilondiss-t-1e6-final.txt"
        )
        NCADENCE = 1
        Rstar = 1 * richio.units.lscale
        Mstar = 1 * richio.units.mscale
        Mbh = 1e6 * richio.units.mscale
    else:
        raise ValueError("Invalid mode. Please choose 1, 2, or 3.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    r_amin = Rstar * (Mbh / Mstar) ** (2 / 3)
    r_p = Rstar * (Mbh / Mstar) ** (1 / 3)
    tmin = (
        np.pi
        / np.sqrt(2)
        * (Rstar**3 / u.G / Mstar) ** (1 / 2)
        * (Mbh / Mstar) ** (1 / 2)
    )

    (
        snapnums,
        ts,
        tfbs,
        epsilondiss1s,
        epsilondiss2s,
        epsilondiss3s,
        epsilondiss4s,
    ) = load_existing_output(OUTPUT_FILE)
    remaining_completed_snapshots = Counter(snapnums)
    if snapnums:
        logger.info(f"Resuming {OUTPUT_FILE} with {len(snapnums)} completed rows")

    for dir in DATADIRS:
        logger.info(f"Processing directory: {dir}")
        snap_files = sorted(
            glob.glob(os.path.join(dir, "snap_full_*.h5")),
            key=lambda f: int(re.search(r"snap_full_(\d+)\.h5", f).group(1)),
        )
        plain_snap_files = [
            f
            for f in glob.glob(os.path.join(dir, "snap_*.h5"))
            if re.fullmatch(r"snap_\d+\.h5", os.path.basename(f))
        ]
        snap_files += sorted(
            plain_snap_files,
            key=lambda f: int(re.search(r"snap_(\d+)\.h5", f).group(1)),
        )

        for snap_file in snap_files[::NCADENCE]:
            try:
                snapnum = int(re.search(r"snap_full_(\d+)\.h5", snap_file).group(1))
            except AttributeError:
                snapnum = int(re.search(r"snap_(\d+)\.h5", snap_file).group(1))

            if os.path.basename(dir) == "TEMPTDE4" and snapnum >= 826:
                continue

            # Count occurrences rather than using a set because snapshots 820--825
            # occur in both TEMPTDE4 and its high-resolution restart.
            if remaining_completed_snapshots[snapnum] > 0:
                remaining_completed_snapshots[snapnum] -= 1
                logger.info(f"Skipping completed snapshot {snapnum}: {snap_file}")
                continue

            snap = richio.load(snap_file)
            try:
                t = snap.t[0]
            except IndexError:
                t = snap.t
            tfb = t / tmin
            if t < 0:
                r_a = r_p
            else:
                r_a = r_amin * tfb ** (2 / 3)

            if mode == 3:
                needs_switch = os.path.basename(dir) == "TEMPTDE"
            else:
                needs_switch = bool(
                    re.fullmatch(r"snap_\d+\.h5", os.path.basename(snap_file))
                )

            if needs_switch:
                x0 = dev.reference_frame_offset(
                    t=t, Mbh=Mbh, Mstar=Mstar, Rstar=Rstar, beta=1
                )
                X = snap.X + x0[0]
                Y = snap.Y + x0[1]
            else:
                X, Y = snap.X, snap.Y

            shock_cuts = (
                X > 0,
                (X > -r_a) & (X < 0) & (Y < 0),
                (X > -r_a) & (X < 0) & (Y > 0),
                X < -r_a,
            )

            dissipation_power = snap.dissipation * snap.volume
            cell_mass = snap.density * snap.volume
            specific_dissipation_unit = (
                dissipation_power.units / cell_mass.units
            )
            epsilondiss = []
            for region, shock_cut in enumerate(shock_cuts, start=1):
                region_mass = np.sum(cell_mass[shock_cut])
                if region_mass <= 0:
                    logger.warning(
                        f"Region {region} has non-positive mass in snapshot "
                        f"{snapnum}; saving NaN"
                    )
                    value = u.unyt_quantity(np.nan, specific_dissipation_unit)
                else:
                    value = np.sum(dissipation_power[shock_cut]) / region_mass
                epsilondiss.append(value)

            snapnums.append(snapnum)
            ts.append(t)
            tfbs.append(tfb)
            epsilondiss1s.append(epsilondiss[0])
            epsilondiss2s.append(epsilondiss[1])
            epsilondiss3s.append(epsilondiss[2])
            epsilondiss4s.append(epsilondiss[3])

            logger.info(
                f"{snapnum} {t} {tfb} "
                + " ".join(str(value) for value in epsilondiss)
            )

            save_output_atomic(
                OUTPUT_FILE,
                [
                    snapnums,
                    ts,
                    tfbs,
                    epsilondiss1s,
                    epsilondiss2s,
                    epsilondiss3s,
                    epsilondiss4s,
                ],
            )


if __name__ == "__main__":
    app()
