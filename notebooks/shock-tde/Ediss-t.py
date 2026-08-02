import numpy as np
import time
import unyt as u
import os
import glob
import re

import richio
import dev


DATADIR = "/disks/emrdata/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes"
OUTPUT_FILE = "Ediss-t-1e4-Ra-select.txt"
NCADENCE = 1

Rstar = 0.47 * richio.units.lscale
Mstar = 0.5 * richio.units.mscale
Mbh = 1e4 * richio.units.mscale
r_p = Rstar * (Mbh / Mstar) ** (1 / 3) * 1
r_amin = Rstar * (Mbh / Mstar) ** (2 / 3)
tmin = (
    np.pi / np.sqrt(2) * (Rstar**3 / u.G / Mstar) ** (1 / 2) * (Mbh / Mstar) ** (1 / 2)
)
Delta = u.G * Mbh / (4 * r_p) * Mstar

snapnums = []
ts = []
t_in_tmins = []
Ediss_shock1s = []
Ediss_shock2s = []
Ediss_shock3s = []
Ediss_shock4s = []

snap_files = sorted(
    glob.glob(os.path.join(DATADIR, "snap_full_*.h5")),
    key=lambda f: int(re.search(r"snap_full_(\d+)\.h5", f).group(1)),
)

for snap_file in snap_files[::NCADENCE]:
    snapnum = int(re.search(r"snap_full_(\d+)\.h5", snap_file).group(1))
    snap = richio.load(snap_file)

    t = snap.t[0]
    t_in_tmin = t / tmin

    r_a = r_amin * (t_in_tmin) ** (2 / 3)

    X, Y = snap.X, snap.Y
    shock1_cut = X > 0
    shock2_cut = (X > -r_a) & (X < 0) & (Y < 0)
    shock3_cut = (X > -r_a) & (X < 0) & (Y > 0)
    shock4_cut = X < -r_a

    Ediss = snap.dissipation * snap.volume
    Ediss_shock1 = np.sum(Ediss[shock1_cut])
    Ediss_shock2 = np.sum(Ediss[shock2_cut])
    Ediss_shock3 = np.sum(Ediss[shock3_cut])
    Ediss_shock4 = np.sum(Ediss[shock4_cut])

    snapnums.append(snapnum)
    ts.append(t)
    t_in_tmins.append(t_in_tmin)
    Ediss_shock1s.append(Ediss_shock1)
    Ediss_shock2s.append(Ediss_shock2)
    Ediss_shock3s.append(Ediss_shock3)
    Ediss_shock4s.append(Ediss_shock4)

    print(
        snapnum,
        t,
        t_in_tmin,
        Ediss_shock1 / Delta * tmin,
        Ediss_shock2 / Delta * tmin,
        Ediss_shock3 / Delta * tmin,
        Ediss_shock4 / Delta * tmin,
    )

    u.savetxt(
        OUTPUT_FILE,
        arrays=[
            u.unyt_array(snapnums),
            u.unyt_array(ts),
            u.unyt_array(t_in_tmins),
            u.unyt_array(Ediss_shock1s) / Delta,
            u.unyt_array(Ediss_shock2s) / Delta,
            u.unyt_array(Ediss_shock3s) / Delta,
            u.unyt_array(Ediss_shock4s) / Delta,
        ],
    )
