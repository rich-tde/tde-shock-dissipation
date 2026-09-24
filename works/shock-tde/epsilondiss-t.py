r"""Compute mass-specific dissipation rates in four BH-frame spatial regions.

Evaluate ``sum(dissipation*volume)/sum(density*volume)`` separately in the
pericenter, outgoing, incoming and outer regions defined below. No stellar
tracer cut is applied. ``r_a`` grows as ``(t/t_fb)**(2/3)`` and equals ``r_p``
for negative time. These are instantaneous specific heating rates, not
specific energies or a radiative cooling measurement.

Input files
-----------
``snap_full_<n>.h5`` and ``snap_<n>.h5`` in the selected mode's run
    Read with ``richio.load``; needs ``X/Y/Z``, ``Density``, ``Volume``,
    ``Dissipation`` and ``Time``. Mode 1 is the ``1e4`` NewSnellius run,
    mode 2 the ``1e5`` YujieSnellius run, mode 3 the ``1e6`` SS24 run.
    Directory defaults are given below and in ``dev.datapaths.DATADIRS``,
    under ``/data1/projects/pi-rossiem/TDE_data``.

Repeat ``--data-dir`` to use relocated copies of the same run. Retain
plain/full snapshot names and restart-directory names for frame corrections.
``TEMPTDE4`` snapshots >=826 are excluded; the historical 820--825 overlap
with ``TEMPTDE4_new`` is retained in this diagnostic.

Output files
------------
``data/processed/epsilondiss-t/epsilondiss-t-<run>-final.txt``
    Default repository-relative path; override with ``--output``. A
    tab-separated unyt text table with ``#`` names/unit/selection comments.
    ``np.loadtxt(path, ndmin=2)`` returns ``float64``, shape ``(N, 7)``.
    N counts snapshot occurrences; duplicate IDs can describe restart overlap.

    Column 0, ``SNAPNUM``
        Integer-valued snapshot number.
    Column 1, ``TIME``
        Time in ``code_time``.
    Column 2, ``TFALLBACK``
        Dimensionless ``t/t_fb``.
    Column 3, ``EPSILONDISS1``
        Pericenter region: ``X > 0``.
    Column 4, ``EPSILONDISS2``
        Outgoing region: ``-r_a < X < 0`` and ``Y < 0``.
    Column 5, ``EPSILONDISS3``
        Incoming region: ``-r_a < X < 0`` and ``Y > 0``.
    Column 6, ``EPSILONDISS4``
        Outer region: ``X < -r_a``.

    Columns 3--6 have units ``code_length**2/code_time**3``. Empty or
    non-positive-mass regions give NaN. Rows follow processing order; sort
    before time integration and decide how to treat restart duplicates.
    Plain NumPy arrays do not retain units; use ``richio.units.registry``.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/epsilondiss-t.py --mode 1
    python works/shock-tde/epsilondiss-t.py --mode 2 \
        --start-snapshot 100 --end-snapshot 110 \
        --output data/processed/SpecificDissipationCheck/1e5.txt

``--stride`` selects every Nth file within each directory. Each completed
snapshot atomically replaces the checkpoint. Existing rows resume by counting
snapshot occurrences; ``--overwrite`` recomputes. Use fresh output paths
when changing inputs or selections.

Loading examples
----------------
Load the four region columns and convert them to specific power::

    import numpy as np
    import unyt as u
    import richio

    path = "data/processed/epsilondiss-t/epsilondiss-t-1e4-final.txt"
    table = np.loadtxt(path, ndmin=2)  # (N, 7), float64
    snapshot = table[:, 0].astype(int)
    time_tfb = table[:, 2]
    rate = u.unyt_array(
        table[:, 3:7], "code_length**2/code_time**3",
        registry=richio.units.registry,
    ).to("erg/(g*s)")
    # rate is (N, 4): pericenter, outgoing, incoming, outer.
    pericenter_rate = rate[:, 0]
    valid = np.isfinite(pericenter_rate)
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

EPSILONDISS_UNIT = "code_length**2/code_time**3"
OUTPUT_HEADER = (
    "SNAPNUM\tTIME\tTFALLBACK\tEPSILONDISS1\tEPSILONDISS2\tEPSILONDISS3\tEPSILONDISS4"
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

    with open(output_file) as handle:
        header = handle.readline().lstrip("# ").split()
    if header != OUTPUT_HEADER.split():
        raise ValueError(f"{output_file} has a different time-series schema")
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

    DATADIRS = tuple(str(p) for p in data_dir) if data_dir else DATADIRS
    OUTPUT_FILE = str(output) if output is not None else OUTPUT_FILE
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    NCADENCE = stride

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
    ) = ([], [], [], [], [], [], []) if overwrite else load_existing_output(OUTPUT_FILE)
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

            if not start_snapshot <= snapnum <= end_snapshot:
                continue

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
            specific_dissipation_unit = dissipation_power.units / cell_mass.units
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
                f"{snapnum} {t} {tfb} " + " ".join(str(value) for value in epsilondiss)
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
