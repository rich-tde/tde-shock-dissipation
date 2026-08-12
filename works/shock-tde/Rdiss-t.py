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
        OUTPUT_FILE = (
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Rdiss-t-1e4-final.txt"
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
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Rdiss-t-1e5-final.txt"
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
            "/home/hey4/rich_tde/data/processed/SimpleTimeseries/Rdiss-t-1e6-final.txt"
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
    r_p = Rstar * (Mbh / Mstar) ** (1 / 3)

    snapnums = []
    ts = []
    tfbs = []
    Rdiss_vec_xs = []
    Rdiss_vec_ys = []
    Rdiss_vec_zs = []
    Rdisss = []

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
            if t < 0:
                r_a = r_p
            else:
                r_a = r_amin * (tfb) ** (2 / 3)

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
                X = snap.X
                Y = snap.Y

            selection = X > -r_a

            r_vec = u.unyt_array([X, Y, snap.Z])[:, selection]
            r = np.sum(r_vec**2, axis=0) ** 0.5
            r_unit_vec = r_vec / r

            diss = snap.dissipation[selection]
            V = snap.volume[selection]

            Rdiss_vec = [
                np.sum(diss * V * r_unit_vec[i, :]) / np.sum(diss * V) for i in range(3)
            ]

            Rdiss = np.sum(diss * V * r) / np.sum(diss * V)

            snapnums.append(snapnum)
            ts.append(t)
            tfbs.append(tfb)
            Rdiss_vec_xs.append(Rdiss_vec[0])
            Rdiss_vec_ys.append(Rdiss_vec[1])
            Rdiss_vec_zs.append(Rdiss_vec[2])
            Rdisss.append(Rdiss)

            logger.info(f"{snapnum} {t} {tfb} {Rdiss_vec} {Rdiss}")

            u.savetxt(
                OUTPUT_FILE,
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(ts),
                    u.unyt_array(tfbs),
                    u.unyt_array(Rdiss_vec_xs),
                    u.unyt_array(Rdiss_vec_ys),
                    u.unyt_array(Rdiss_vec_zs),
                    u.unyt_array(Rdisss),
                ],
                header="SNAPNUM\tTIME\tTFALLBACK\tRDISSVECX\tRDISSVECY\tRDISSVECZ\tRDISS",
            )


if __name__ == "__main__":
    app()
