r"""Measure dissipation-weighted radius and mean radial direction versus time.

Select BH-frame ``x > -r_a(t)`` and weight cells by ``dissipation*volume``.
Calculate the mean radius and mean radial unit vector. ``r_a`` grows as
``(t/t_fb)**(2/3)`` and equals ``r_p`` for negative time. The three direction
components are dimensionless means, not a Cartesian position or a unit vector
of guaranteed length one. This is a broad dissipation locator, not a detected
shock surface.

Input files
-----------
``snap_full_<n>.h5`` and ``snap_<n>.h5`` in the selected run
    Read with ``richio.load``; required fields are ``X/Y/Z``, ``Volume``,
    ``Dissipation`` and ``Time``. Mode 1 uses the ``1e4`` NewSnellius run,
    mode 2 the ``1e5`` YujieSnellius run, mode 3 the ``1e6`` run under
    ``/data1/projects/pi-rossiem/TDE_data/SS24_diag/`` (``TEMPTDE``,
    ``TEMPTDE4``, ``TEMPTDE4_new``). Exact paths are the mode defaults below
    and also in ``dev.datapaths.DATADIRS``.

Repeat ``--data-dir`` for relocated copies with the same directory and
snapshot naming, which determine BH-frame corrections. ``TEMPTDE4`` >=820
is replaced by its high-resolution restart.

Output files
------------
``data/processed/SimpleTimeseries/Rdiss-t-<run>-final.txt``
    Tab-separated text with ``#`` names/unit comments; override ``--output``.
    ``np.loadtxt(path, ndmin=2)`` returns ``float64``, shape ``(N, 7)``.
    N counts completed snapshot occurrences; rows follow processing order.

    Column 0, ``SNAPNUM``
        Integer-valued snapshot ID; cast to integer after loading.
    Column 1, ``TIME``
        Time in ``code_time``.
    Column 2, ``TFALLBACK``
        Dimensionless ``t/t_fb``.
    Columns 3, 4, 5: ``RDISSVECX``, ``RDISSVECY``, ``RDISSVECZ``
        Respectively x, y, z components of the power-weighted mean radial
        unit vector. ``table[:, 3:6]`` has shape ``(N, 3)``.
    Column 6, ``RDISS``
        Power-weighted mean radius in ``code_length``.

    Zero selected power gives undefined ratios (NaN); the code does not
    replace these with zeros. ``np.loadtxt`` does not attach physical units.
    Use the ``richio`` unit registry to convert the radius or time.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/Rdiss-t.py --mode 2
    python works/shock-tde/Rdiss-t.py --mode 1 --stride 10 \
        --output data/processed/RadiusCheck/1e4.txt

``--start-snapshot`` / ``--end-snapshot`` are inclusive. Reruns resume by
snapshot occurrence; ``--overwrite`` recomputes the selected series. Each
snapshot atomically replaces the checkpoint. Use a different output for a
changed data source or selection: resume assumes the same configuration.

Loading examples
----------------
Load the time, direction vectors, and unitful radius::

    import numpy as np
    import unyt as u
    import richio

    path = "data/processed/SimpleTimeseries/Rdiss-t-1e4-final.txt"
    table = np.loadtxt(path, ndmin=2)  # (N, 7)
    time_tfb = table[:, 2]
    direction_xyz = table[:, 3:6]     # (N, 3): x, y, z; dimensionless
    radius = u.unyt_array(table[:, 6], richio.units.lscale).to("cm")
    valid = np.isfinite(radius) & np.isfinite(direction_xyz).all(axis=1)
    order = np.argsort(time_tfb[valid])
    time_sorted = time_tfb[valid][order]
    radius_cm = radius[valid][order].to_value("cm")
"""

import glob
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import typer
import unyt as u
from loguru import logger

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import dev
import richio

app = typer.Typer(add_completion=False)


@app.command()
def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6 Msun."),
    data_dir: list[Path] | None = typer.Option(
        None, help="Repeat for relocated copies of this mode's run/restart directories."
    ),
    output: Path | None = typer.Option(
        None,
        help="Override the output table; use a fresh path for a changed selection.",
    ),
    start_snapshot: int = typer.Option(
        0, min=0, help="First snapshot number, inclusive."
    ),
    end_snapshot: int = typer.Option(
        10000, min=0, help="Last snapshot number, inclusive."
    ),
    stride: int = typer.Option(
        1, min=1, help="Take every Nth snapshot within each directory."
    ),
    overwrite: bool = typer.Option(
        False, help="Recompute instead of resuming rows from the existing output."
    ),
):
    """Process the selected run and checkpoint its time series after every snapshot."""
    if end_snapshot < start_snapshot:
        raise typer.BadParameter("--end-snapshot must be at least --start-snapshot")
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

    DATADIRS = tuple(str(p) for p in data_dir) if data_dir else DATADIRS
    OUTPUT_FILE = str(output) if output is not None else OUTPUT_FILE
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    NCADENCE = stride

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

    columns = [snapnums, ts, tfbs, Rdiss_vec_xs, Rdiss_vec_ys, Rdiss_vec_zs, Rdisss]
    if (
        not overwrite
        and Path(OUTPUT_FILE).exists()
        and Path(OUTPUT_FILE).stat().st_size
    ):
        with open(OUTPUT_FILE) as handle:
            header = handle.readline().lstrip("# ").split()
        if (
            header
            != "SNAPNUM TIME TFALLBACK RDISSVECX RDISSVECY RDISSVECZ RDISS".split()
        ):
            raise ValueError("Existing output has a different time-series schema")
        raw = np.loadtxt(OUTPUT_FILE, ndmin=2)
        if raw.shape[1] != 7:
            raise ValueError("Existing output has the wrong number of columns")
        units = [
            "dimensionless",
            "code_time",
            "dimensionless",
            "dimensionless",
            "dimensionless",
            "dimensionless",
            "code_length",
        ]
        for column, values, unit in zip(columns, raw.T, units):
            column.extend(u.unyt_array(values, unit, registry=richio.units.registry))
    remaining_completed_snapshots = Counter(int(n) for n in snapnums)

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

            if not start_snapshot <= snapnum <= end_snapshot:
                continue

            if (
                os.path.basename(dir) == "TEMPTDE4" and snapnum >= 820
            ):  # use hi-res restart of TEMPTDE4_new
                continue

            if remaining_completed_snapshots[snapnum] > 0:
                remaining_completed_snapshots[snapnum] -= 1
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
                f"{OUTPUT_FILE}.tmp",
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

            os.replace(f"{OUTPUT_FILE}.tmp", OUTPUT_FILE)


if __name__ == "__main__":
    app()
