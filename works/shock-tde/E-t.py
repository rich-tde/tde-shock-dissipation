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
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes",
        )
        OUTPUT_FILE = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e4.txt"
        NCADENCE = 1
        Rstar = 0.47 * richio.units.lscale
        Mstar = 0.5 * richio.units.mscale
        Mbh = 1e4 * richio.units.mscale
    elif mode == 2:
        # 1e5
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        OUTPUT_FILE = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e5.txt"
        NCADENCE = 1
        Rstar = 0.47 * richio.units.lscale
        Mstar = 0.5 * richio.units.mscale
        Mbh = 1e5 * richio.units.mscale
    elif mode == 3:
        # 1e6
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
        )
        OUTPUT_FILE = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e6.txt"
        NCADENCE = 1
        Rstar = 1 * richio.units.lscale
        Mstar = 1 * richio.units.mscale
        Mbh = 1e6 * richio.units.mscale
    else:
        raise ValueError("Invalid mode. Please choose 1, 2, or 3.")

    r_amin = Rstar * (Mbh / Mstar) ** (2 / 3)
    tmin = (
        np.pi
        / np.sqrt(2)
        * (Rstar**3 / u.G / Mstar) ** (1 / 2)
        * (Mbh / Mstar) ** (1 / 2)
    )
    r_p = Rstar * (Mbh / Mstar) ** (1 / 3)

    snapnums = []
    ts = []
    tfbs = []
    Eorbs = []
    Erads = []
    Eints = []
    Egravs = []
    Ekins = []
    warned_zero_erad = False

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

            if (
                os.path.basename(dir) == "TEMPTDE4" and snapnum >= 820
            ):  # use hi-res restart of TEMPTDE4_new
                continue

            snap = richio.load(snap_file)
            try:
                t = snap.t[0]
            except IndexError:
                t = snap.t
            tfb = t / tmin

            # if t < 0:
            #     r_a = r_p
            # else:
            #     r_a = r_amin * (tfb) ** (2 / 3)

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
                vx = snap.vx + x0[2]
                vy = snap.vy + x0[3]
            else:
                X = snap.X
                Y = snap.Y
                vx = snap.vx
                vy = snap.vy
            Z = snap.Z
            vz = snap.vz

            r = np.sqrt(X**2 + Y**2 + Z**2)

            r_g = u.G * Mbh / u.c**2

            v2 = vx**2 + vy**2 + vz**2
            rho = snap.density
            V = snap.volume

            r_0 = 0.6 * r_p
            # smoothed PW
            Egrav_i = np.where(
                r > 0.6 * r_p,
                -u.G * Mbh * rho * V / (r - 2 * r_g),
                -u.G * Mbh * rho * V * r**2 / (2 * r_0 * (r_0 - 2 * r_g) ** 2),
            )
            Ekin_i = 1 / 2 * v2 * rho * V
            Eorb_i = Ekin_i + Egrav_i
            specific_Erad = snap.Erad
            Erad_i = specific_Erad * V * rho
            Eint_i = snap.sie * V * rho

            Ekin = np.sum(Ekin_i)
            Eorb = np.sum(Eorb_i)
            # np.where may preserve Egrav_i in cgs time units.  Deriving the
            # total from Eorb = Egrav + Ekin keeps all output energies in the
            # same code-energy unit and guarantees exact budget closure.
            Egrav = Eorb - Ekin
            Erad = np.sum(Erad_i)
            Eint = np.sum(Eint_i)

            if not warned_zero_erad and np.all(specific_Erad == 0):
                logger.warning(
                    "Erad is present but identically zero in {}; no stored "
                    "radiation energy is available for this snapshot",
                    snap_file,
                )
                warned_zero_erad = True

            snapnums.append(snapnum)
            ts.append(t)
            tfbs.append(tfb)
            Eorbs.append(Eorb)
            Erads.append(Erad)
            Eints.append(Eint)
            Egravs.append(Egrav)
            Ekins.append(Ekin)

            logger.info(
                "snapnum={} t={} tfb={} Eorb={} Erad={} Eint={} Egrav={} Ekin={}",
                snapnum,
                t,
                tfb,
                Eorb,
                Erad,
                Eint,
                Egrav,
                Ekin,
            )

            u.savetxt(
                OUTPUT_FILE,
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(ts),
                    u.unyt_array(tfbs),
                    u.unyt_array(Eorbs),
                    u.unyt_array(Erads),
                    u.unyt_array(Eints),
                    u.unyt_array(Egravs),
                    u.unyt_array(Ekins),
                ],
                header=(
                    "SNAPNUM\tTIME\tTFALLBACK\tEORB\tERAD\tEINT\tEGRAV\tEKIN"
                ),
            )


if __name__ == "__main__":
    app()
