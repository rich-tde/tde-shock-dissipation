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
Delta = u.G * Mbh / (4 * r_p) * Mstar / 2

snapnums = []
ts = []
t_in_tmins = []
Ediss1s = []
Ediss2s = []
Ediss3s = []
Ediss4s = []

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
    Ediss1 = np.sum(Ediss[shock1_cut])
    Ediss2 = np.sum(Ediss[shock2_cut])
    Ediss3 = np.sum(Ediss[shock3_cut])
    Ediss4 = np.sum(Ediss[shock4_cut])

    snapnums.append(snapnum)
    ts.append(t)
    t_in_tmins.append(t_in_tmin)
    Ediss1s.append(Ediss1)
    Ediss2s.append(Ediss2)
    Ediss3s.append(Ediss3)
    Ediss4s.append(Ediss4)

    print(
        snapnum,
        t,
        t_in_tmin,
        Ediss1,
        Ediss2,
        Ediss3,
        Ediss4,
    )

    u.savetxt(
        OUTPUT_FILE,
        arrays=[
            u.unyt_array(snapnums),
            u.unyt_array(ts),
            u.unyt_array(t_in_tmins),
            u.unyt_array(Ediss1s),
            u.unyt_array(Ediss2s),
            u.unyt_array(Ediss3s),
            u.unyt_array(Ediss4s),
        ],
    )
