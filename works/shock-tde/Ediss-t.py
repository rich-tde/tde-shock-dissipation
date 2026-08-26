import glob
import os
import re

import numpy as np
import typer
import unyt as u
from loguru import logger

import dev
import richio

app = typer.Typer()


@app.command()
def main(mode: int = typer.Option(..., help="Run 1e4, 1e5, or 1e6")):
    if mode == 1:
        # 1e4
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes",
        )
        output_file = (
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Ediss-t-1e4-final.txt"
        )
        cadence = 1
        stellar_radius = 0.47 * richio.units.lscale
        stellar_mass = 0.5 * richio.units.mscale
        black_hole_mass = 1e4 * richio.units.mscale
    elif mode == 2:
        # 1e5
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        output_file = (
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Ediss-t-1e5-final.txt"
        )
        cadence = 1
        stellar_radius = 0.47 * richio.units.lscale
        stellar_mass = 0.5 * richio.units.mscale
        black_hole_mass = 1e5 * richio.units.mscale
    elif mode == 3:
        # 1e6
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
        )
        output_file = (
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Ediss-t-1e6-final.txt"
        )
        cadence = 1
        stellar_radius = 1 * richio.units.lscale
        stellar_mass = 1 * richio.units.mscale
        black_hole_mass = 1e6 * richio.units.mscale
    else:
        raise ValueError("Invalid mode. Please choose 1, 2, or 3.")

    most_bound_apocenter = stellar_radius * (black_hole_mass / stellar_mass) ** (2 / 3)
    pericenter_radius = stellar_radius * (black_hole_mass / stellar_mass) ** (1 / 3)
    fallback_time = (
        np.pi
        / np.sqrt(2)
        * (stellar_radius**3 / u.G / stellar_mass) ** (1 / 2)
        * (black_hole_mass / stellar_mass) ** (1 / 2)
    )
    snapnums = []
    times = []
    fallback_times = []
    pericenter_powers = []
    outgoing_powers = []
    incoming_powers = []
    outer_powers = []

    for data_directory in data_directories:
        logger.info(f"Processing directory: {data_directory}")
        snap_files = sorted(
            glob.glob(os.path.join(data_directory, "snap_full_*.h5")),
            key=lambda f: int(re.search(r"snap_full_(\d+)\.h5", f).group(1)),
        )
        plain_snap_files = [
            f
            for f in glob.glob(os.path.join(data_directory, "snap_*.h5"))
            if re.fullmatch(r"snap_\d+\.h5", os.path.basename(f))
        ]
        snap_files += sorted(
            plain_snap_files,
            key=lambda f: int(re.search(r"snap_(\d+)\.h5", f).group(1)),
        )

        for snap_file in snap_files[::cadence]:
            try:
                snapnum = int(re.search(r"snap_full_(\d+)\.h5", snap_file).group(1))
            except AttributeError:
                snapnum = int(re.search(r"snap_(\d+)\.h5", snap_file).group(1))

            if (
                os.path.basename(data_directory) == "TEMPTDE4" and snapnum >= 826
            ):  # use hi-res restart of TEMPTDE4_new
                continue

            snap = richio.load(snap_file)
            try:
                t = snap.t[0]
            except IndexError:
                t = snap.t
            tfb = t / fallback_time
            if t < 0:
                apocenter_radius = pericenter_radius
            else:
                apocenter_radius = most_bound_apocenter * tfb ** (2 / 3)

            if mode == 3:
                needs_switch = os.path.basename(data_directory) == "TEMPTDE"
            else:
                needs_switch = bool(
                    re.fullmatch(r"snap_\d+\.h5", os.path.basename(snap_file))
                )

            if needs_switch:
                frame_offset = dev.reference_frame_offset(
                    t=t,
                    Mbh=black_hole_mass,
                    Mstar=stellar_mass,
                    Rstar=stellar_radius,
                    beta=1,
                )
                x = snap.X + frame_offset[0]
                y = snap.Y + frame_offset[1]
            else:
                x, y = snap.X, snap.Y

            pericenter_region = x > 0
            outgoing_region = (x > -apocenter_radius) & (x < 0) & (y < 0)
            incoming_region = (x > -apocenter_radius) & (x < 0) & (y > 0)
            outer_region = x < -apocenter_radius

            cell_dissipation_power = snap.dissipation * snap.volume
            pericenter_power = np.sum(cell_dissipation_power[pericenter_region])
            outgoing_power = np.sum(cell_dissipation_power[outgoing_region])
            incoming_power = np.sum(cell_dissipation_power[incoming_region])
            outer_power = np.sum(cell_dissipation_power[outer_region])

            snapnums.append(snapnum)
            times.append(t)
            fallback_times.append(tfb)
            pericenter_powers.append(pericenter_power)
            outgoing_powers.append(outgoing_power)
            incoming_powers.append(incoming_power)
            outer_powers.append(outer_power)

            logger.info(
                "{} {} {} {} {} {} {}",
                snapnum,
                t,
                tfb,
                pericenter_power,
                outgoing_power,
                incoming_power,
                outer_power,
            )

            u.savetxt(
                output_file,
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(times),
                    u.unyt_array(fallback_times),
                    u.unyt_array(pericenter_powers),
                    u.unyt_array(outgoing_powers),
                    u.unyt_array(incoming_powers),
                    u.unyt_array(outer_powers),
                ],
                header="SNAPNUM\tTIME\tTFALLBACK\tEDISS1\tEDISS2\tEDISS3\tEDISS4",
                footer="shock1_cut = X > 0\nshock2_cut = (X > -r_a) & (X < 0) & (Y < 0)\nshock3_cut = (X > -r_a) & (X < 0) & (Y > 0)\nshock4_cut = X < -r_a",
            )


if __name__ == "__main__":
    app()
