import numpy as np
import unyt as u
import os
import glob
import re
import typer
from loguru import logger

import richio

app = typer.Typer()


@app.command()
def main(mode: int = typer.Option(..., help="Run 1e4, 1e5, or 1e6")):
    if mode == 1:
        # 1e4
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes",
        )
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/tde_Ediss_t_per_region/Ediss-t-1e4.txt"
        )
        NCADENCE = 1
        Rstar = 0.47 * richio.units.lscale
        Mstar = 0.5 * richio.units.mscale
        Mbh = 1e4 * richio.units.mscale
    elif mode == 2:
        # 1e5
        DATADIRS = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/tde_Ediss_t_per_region/Ediss-t-1e5.txt"
        )
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
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/tde_Ediss_t_per_region/Ediss-t-1e6.txt"
        )
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
    # r_p = Rstar * (Mbh / Mstar) ** (1 / 3) * 1
    # Delta = u.G * Mbh / (4 * r_p) * Mstar / 2

    snapnums = []
    ts = []
    tfbs = []
    Ediss1s = []
    Ediss2s = []
    Ediss3s = []
    Ediss4s = []

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

            snap = richio.load(snap_file)
            try:
                t = snap.t[0]
            except IndexError:
                t = snap.t
            tfb = t / tmin
            r_a = r_amin * (tfb) ** (2 / 3)

            X, Y = snap.X, snap.Y
            shock1_cut = X > 0
            shock2_cut = (X > -r_a) & (X < 0) & (Y < 0)
            shock3_cut = (X > -r_a) & (X < 0) & (Y > 0)
            shock4_cut = X < -r_a

            Ediss = snap.dissipation * snap.volume
            Ediss1 = np.sum(Ediss[shock1_cut])
            Ediss2 = np.sum(Ediss[shock2_cut])
            Ediss3 = np.sum(Ediss[shock3_cut])
            Ediss4 = np.sum(Ediss[shock4_cut])

            snapnums.append(snapnum)
            ts.append(t)
            tfbs.append(tfb)
            Ediss1s.append(Ediss1)
            Ediss2s.append(Ediss2)
            Ediss3s.append(Ediss3)
            Ediss4s.append(Ediss4)

            logger.info(f"{snapnum} {t} {tfb} {Ediss1} {Ediss2} {Ediss3} {Ediss4}")

            u.savetxt(
                OUTPUT_FILE,
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(ts),
                    u.unyt_array(tfbs),
                    u.unyt_array(Ediss1s),
                    u.unyt_array(Ediss2s),
                    u.unyt_array(Ediss3s),
                    u.unyt_array(Ediss4s),
                ],
            )


if __name__ == "__main__":
    app()
