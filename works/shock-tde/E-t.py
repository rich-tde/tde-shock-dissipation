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
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e4.txt"
        cadence = 1
        stellar_radius = 0.47 * richio.units.lscale
        stellar_mass = 0.5 * richio.units.mscale
        black_hole_mass = 1e4 * richio.units.mscale
    elif mode == 2:
        # 1e5
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e5.txt"
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
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e6.txt"
        cadence = 1
        stellar_radius = 1 * richio.units.lscale
        stellar_mass = 1 * richio.units.mscale
        black_hole_mass = 1e6 * richio.units.mscale
    else:
        raise ValueError("Invalid mode. Please choose 1, 2, or 3.")

    tmin = (
        np.pi
        / np.sqrt(2)
        * (stellar_radius**3 / u.G / stellar_mass) ** (1 / 2)
        * (black_hole_mass / stellar_mass) ** (1 / 2)
    )
    pericenter_radius = stellar_radius * (black_hole_mass / stellar_mass) ** (1 / 3)

    snapnums = []
    times = []
    fallback_times = []
    orbital_energies = []
    radiation_energies = []
    internal_energies = []
    gravitational_energies = []
    kinetic_energies = []
    warned_zero_erad = False

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
                os.path.basename(data_directory) == "TEMPTDE4" and snapnum >= 820
            ):  # use hi-res restart of TEMPTDE4_new
                continue

            snap = richio.load(snap_file)
            try:
                t = snap.t[0]
            except IndexError:
                t = snap.t
            tfb = t / tmin

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
                velocity_x = snap.vx + frame_offset[2]
                velocity_y = snap.vy + frame_offset[3]
            else:
                x = snap.X
                y = snap.Y
                velocity_x = snap.vx
                velocity_y = snap.vy
            z = snap.Z
            velocity_z = snap.vz

            radius = np.sqrt(x**2 + y**2 + z**2)

            gravitational_radius = u.G * black_hole_mass / u.c**2

            speed_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
            density = snap.density
            volume = snap.volume

            softening_radius = 0.6 * pericenter_radius
            # smoothed PW
            cell_gravitational_energy = np.where(
                radius > softening_radius,
                -u.G
                * black_hole_mass
                * density
                * volume
                / (radius - 2 * gravitational_radius),
                -u.G
                * black_hole_mass
                * density
                * volume
                * radius**2
                / (
                    2
                    * softening_radius
                    * (softening_radius - 2 * gravitational_radius) ** 2
                ),
            )
            cell_kinetic_energy = 1 / 2 * speed_squared * density * volume
            cell_orbital_energy = cell_kinetic_energy + cell_gravitational_energy
            specific_radiation_energy = snap.Erad
            cell_radiation_energy = specific_radiation_energy * volume * density
            cell_internal_energy = snap.sie * volume * density

            kinetic_energy = np.sum(cell_kinetic_energy)
            orbital_energy = np.sum(cell_orbital_energy)
            # np.where may preserve Egrav_i in cgs time units.  Deriving the
            # total from Eorb = Egrav + Ekin keeps all output energies in the
            # same code-energy unit and guarantees exact budget closure.
            gravitational_energy = orbital_energy - kinetic_energy
            radiation_energy = np.sum(cell_radiation_energy)
            internal_energy = np.sum(cell_internal_energy)

            if not warned_zero_erad and np.all(specific_radiation_energy == 0):
                logger.warning(
                    "Erad is present but identically zero in {}; no stored "
                    "radiation energy is available for this snapshot",
                    snap_file,
                )
                warned_zero_erad = True

            snapnums.append(snapnum)
            times.append(t)
            fallback_times.append(tfb)
            orbital_energies.append(orbital_energy)
            radiation_energies.append(radiation_energy)
            internal_energies.append(internal_energy)
            gravitational_energies.append(gravitational_energy)
            kinetic_energies.append(kinetic_energy)

            logger.info(
                "snapnum={} t={} tfb={} Eorb={} Erad={} Eint={} Egrav={} Ekin={}",
                snapnum,
                t,
                tfb,
                orbital_energy,
                radiation_energy,
                internal_energy,
                gravitational_energy,
                kinetic_energy,
            )

            u.savetxt(
                output_file,
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(times),
                    u.unyt_array(fallback_times),
                    u.unyt_array(orbital_energies),
                    u.unyt_array(radiation_energies),
                    u.unyt_array(internal_energies),
                    u.unyt_array(gravitational_energies),
                    u.unyt_array(kinetic_energies),
                ],
                header=("SNAPNUM\tTIME\tTFALLBACK\tEORB\tERAD\tEINT\tEGRAV\tEKIN"),
            )


if __name__ == "__main__":
    app()
