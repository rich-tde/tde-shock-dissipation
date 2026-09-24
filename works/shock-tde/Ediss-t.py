r"""Measure dissipation power versus time in four spatial regions of a TDE.

Correct positions to the BH frame and sum ``dissipation*volume`` in the
chosen regions. ``--regions standard`` keeps the pericenter, outgoing,
incoming and outer components. ``--regions nozzle-split`` separates the
pericenter region at spherical radius ``3*r_p`` and excludes the outer region.
``r_a`` grows as ``(t/t_fb)**(2/3)`` and equals ``r_p`` for negative time.
These are instantaneous powers; a spatial region need not contain only one
physical shock.

Input files
-----------
``snap_full_<n>.h5`` and ``snap_<n>.h5``
    Read with ``richio.load`` from ``dev.datapaths.DATADIRS``. Modes 1/2/3
    select the ``1e4`` NewSnellius, ``1e5`` YujieSnellius and ``1e6`` SS24
    runs, respectively, under ``/data1/projects/pi-rossiem/TDE_data``.
    Required fields: ``X/Y/Z``, ``Time``, ``Dissipation`` and ``Volume``.
    ``--npoints`` also reads the HDF5 ``Time`` dataset to select samples.

Repeat ``--data-dir`` for relocated copies with the same stellar/BH parameters.
Directory names and plain/full snapshot naming control the frame correction.
The historical ``TEMPTDE4`` cutoff is >=826: snapshots 820--825 from both
restart directories remain in this diagnostic.

Output files
------------
``data/processed/SimpleTimeseries/Ediss-t-<run>[-n<N>]-final.txt``
    Default for ``--regions standard``. The optional ``-n<N>`` suffix
    appears only when ``--npoints N`` is supplied.
``data/processed/EdissFourRegions/Ediss-t-four-regions-<run>[-n<N>].txt``
    Default for ``--regions nozzle-split``; preserves the former standalone
    four-region tool's filenames. ``--output`` overrides either destination.

Both are tab-separated text with ``#`` names, units and selection comments.
``np.loadtxt(path, ndmin=2)`` returns ``float64``, shape ``(Nsnap, 7)``;
Nsnap is the number of completed snapshot occurrences, not necessarily unique
snapshot IDs. Rows follow selected-file order. Column indices are zero-based:

Columns 0, 1, 2: ``SNAPNUM``, ``TIME``, ``TFALLBACK``
    Integer-valued snapshot number, time in ``code_time``, and dimensionless
    ``t/t_fb``, respectively. Cast column 0 to integer after loading.
Columns 3--6 for ``--regions standard``
    3: ``EDISS1``, pericenter ``X > 0``.
    4: ``EDISS2``, outgoing ``-r_a < X < 0`` and ``Y < 0``.
    5: ``EDISS3``, incoming ``-r_a < X < 0`` and ``Y > 0``.
    6: ``EDISS4``, outer ``X < -r_a``.
Columns 3--6 for ``--regions nozzle-split``
    3: ``EDISS_NOZZLE``, ``X > 0`` and spherical ``r < 3*r_p``.
    4: ``EDISS_STREAM_DISK``, ``X > 0`` and ``r >= 3*r_p``.
    5: ``EDISS_OUTGOING``, ``-r_a < X < 0`` and ``Y < 0``.
    6: ``EDISS_INCOMING``, ``-r_a < X < 0`` and ``Y > 0``.

All four powers use ``code_mass*code_length**2/code_time**3``; empty selections
sum to zero. The two seven-column schemas have different meanings: check the
header before combining them. Strict region boundaries omit cells exactly on
their boundaries. NumPy loading discards units; reattach the ``richio`` code
units explicitly. A legacy nozzle-split ``.txt.partial`` checkpoint may be read
if no final table exists; new checkpoints are saved to the final text path.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/Ediss-t.py --mode 1
    python works/shock-tde/Ediss-t.py --mode 3 \
        --regions nozzle-split --npoints 10
    python works/shock-tde/Ediss-t.py --mode 2 \
        --start-snapshot 100 --end-snapshot 110 \
        --output data/processed/EdissCheck/1e5.txt

``--stride`` subsamples within each directory. ``--npoints`` samples evenly
in time-sorted snapshot index after ``t/t_fb >= 0.1`` (1e4/1e5) or ``>=0.7``
(1e6). Without it, earlier snapshots remain eligible. Each completed snapshot
is saved atomically. Resume counts snapshot occurrences, including overlaps;
``--overwrite`` recomputes. Use separate outputs after changing sources,
selection, cadence or region scheme. Same-schema changed settings are not
content-checked against an existing checkpoint.

Loading examples
----------------
Load a standard-region table and convert the four powers to CGS::

    from pathlib import Path
    import numpy as np
    import unyt as u
    import richio

    path = Path("data/processed/SimpleTimeseries/Ediss-t-1e4-final.txt")
    table = np.loadtxt(path, ndmin=2)  # (Nsnap, 7)
    snapshot = table[:, 0].astype(int)
    time_tfb = table[:, 2]
    power = u.unyt_array(
        table[:, 3:7], "code_mass*code_length**2/code_time**3",
        registry=richio.units.registry,
    ).to("erg/s")
    # power is (Nsnap, 4): pericenter, outgoing, incoming, outer.
    total_selected_power = power.sum(axis=1)  # (Nsnap,), erg/s

For the second command above, load the nozzle component explicitly::

    root = Path("data/processed/EdissFourRegions")
    table = np.loadtxt(root / "Ediss-t-four-regions-1e6-n10.txt", ndmin=2)
    time_tfb = table[:, 2]
    nozzle_power = u.unyt_array(
        table[:, 3], "code_mass*code_length**2/code_time**3",
        registry=richio.units.registry,
    ).to("erg/s")
    # Column 6 is INCOMING here, not the standard table's OUTER component.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import typer
import unyt as u
from loguru import logger

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

from dev.datapaths import DATADIRS, TDE_PARAMETERS

import dev
import richio

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[2]
POWER_UNIT = "code_mass*code_length**2/code_time**3"


class Regions(str, Enum):
    standard = "standard"
    nozzle_split = "nozzle-split"


def snapshot_number(path: Path) -> int:
    return int(re.fullmatch(r"snap_(?:full_)?(\d+)\.h5", path.name).group(1))


def selected_files(datadirs, stride, start, end, npoints, fallback_time, minimum_tfb):
    paths = []
    for directory in datadirs:
        full = sorted(directory.glob("snap_full_*.h5"), key=snapshot_number)
        plain = sorted(
            (
                p
                for p in directory.glob("snap_*.h5")
                if re.fullmatch(r"snap_\d+\.h5", p.name)
            ),
            key=snapshot_number,
        )
        for path in (full + plain)[::stride]:
            number = snapshot_number(path)
            if start <= number <= end and not (
                directory.name == "TEMPTDE4" and number >= 826
            ):
                paths.append(path)
    if npoints is None:
        return paths
    eligible = []
    for path in paths:
        with h5py.File(path) as handle:
            time_code = float(np.asarray(handle["Time"]).squeeze())
        if time_code / fallback_time.to_value("code_time") >= minimum_tfb:
            eligible.append((time_code, path))
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < npoints:
        raise ValueError(
            f"Only {len(eligible)} snapshots meet the sampling window; requested {npoints}"
        )
    indices = np.rint(np.linspace(0, len(eligible) - 1, npoints)).astype(int)
    return [eligible[i][1] for i in indices]


def region_masks(x, y, z, r_p, r_a, regions):
    outgoing = (x > -r_a) & (x < 0) & (y < 0)
    incoming = (x > -r_a) & (x < 0) & (y > 0)
    if regions == Regions.standard:
        return x > 0, outgoing, incoming, x < -r_a
    radius = np.sqrt(x**2 + y**2 + z**2)
    return (
        (x > 0) & (radius < 3 * r_p),
        (x > 0) & (radius >= 3 * r_p),
        outgoing,
        incoming,
    )


def load_checkpoint(path, regions):
    if not path.exists() or path.stat().st_size == 0:
        return [[] for _ in range(7)]
    with path.open() as handle:
        header = handle.readline()
    expected = (
        "SNAPNUM TIME TFALLBACK EDISS1 EDISS2 EDISS3 EDISS4"
        if regions == Regions.standard
        else "SNAPNUM TIME TFALLBACK EDISS_NOZZLE EDISS_STREAM_DISK EDISS_OUTGOING EDISS_INCOMING"
    )
    if header.lstrip("# ").split() != expected.split():
        raise ValueError(
            f"{path} does not use the requested {regions.value} region scheme"
        )
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError(f"{path} must have seven finite columns")
    if not np.array_equal(raw[:, 0], raw[:, 0].astype(int)):
        raise ValueError(f"{path} has non-integer snapshot numbers")
    units = ["dimensionless", "code_time", "dimensionless"] + [POWER_UNIT] * 4
    return [
        list(u.unyt_array(column, unit, registry=richio.units.registry))
        for column, unit in zip(raw.T, units)
    ]


def save_timeseries(path, columns, regions, sources):
    if regions == Regions.standard:
        names = "EDISS1\tEDISS2\tEDISS3\tEDISS4"
        definitions = (
            "pericenter X>0; outgoing -r_a<X<0,Y<0; incoming -r_a<X<0,Y>0; outer X<-r_a"
        )
    else:
        names = "EDISS_NOZZLE\tEDISS_STREAM_DISK\tEDISS_OUTGOING\tEDISS_INCOMING"
        definitions = "nozzle X>0,r<3*r_p; stream_disk X>0,r>=3*r_p; outgoing -r_a<X<0,Y<0; incoming -r_a<X<0,Y>0; X<-r_a excluded"
    temporary = path.with_suffix(path.suffix + ".tmp")
    u.savetxt(
        temporary,
        arrays=[u.unyt_array(c) for c in columns],
        header="SNAPNUM\tTIME\tTFALLBACK\t" + names,
        footer=definitions + "\nSource directories: " + ", ".join(map(str, sources)),
    )
    os.replace(temporary, path)


@app.command()
def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6 Msun."),
    regions: Regions = typer.Option(
        Regions.standard, help="Spatial partition of dissipation power."
    ),
    data_dir: list[Path] | None = typer.Option(
        None,
        help="Repeat for relocated run/restart directories; same physics and naming as --mode.",
    ),
    output: Path | None = typer.Option(
        None, help="Override output table; use a distinct file for a changed selection."
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
    npoints: int | None = typer.Option(
        None,
        min=2,
        help="Evenly sample N snapshots in the established plotting window.",
    ),
    overwrite: bool = typer.Option(
        False, help="Recompute instead of resuming existing rows."
    ),
):
    """Sum regional dissipation powers; choose --regions nozzle-split for the nozzle comparison."""
    if end_snapshot < start_snapshot:
        raise typer.BadParameter("--end-snapshot must be at least --start-snapshot")
    run = {1: "1e4", 2: "1e5", 3: "1e6"}[mode]
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    mbh, mstar, rstar = (
        mbh * richio.units.mscale,
        mstar * richio.units.mscale,
        rstar * richio.units.lscale,
    )
    r_p = rstar * (mbh / mstar) ** (1 / 3)
    r_amin = rstar * (mbh / mstar) ** (2 / 3)
    fallback_time = (
        np.pi / np.sqrt(2) * np.sqrt(rstar**3 / u.G / mstar) * np.sqrt(mbh / mstar)
    )
    directories = data_dir or list(DATADIRS[run])
    if output is None:
        suffix = "" if npoints is None else f"-n{npoints}"
        if regions == Regions.standard:
            output = (
                REPO
                / f"data/processed/SimpleTimeseries/Ediss-t-{run}{suffix}-final.txt"
            )
        else:
            output = (
                REPO
                / f"data/processed/EdissFourRegions/Ediss-t-four-regions-{run}{suffix}.txt"
            )
    paths = selected_files(
        directories,
        stride,
        start_snapshot,
        end_snapshot,
        npoints,
        fallback_time,
        0.7 if mode == 3 else 0.1,
    )
    if not paths:
        raise ValueError("No snapshots match the selected input directories and range")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The former nozzle-split tool stored unfinished work in a .partial file.
    checkpoint = (
        output if output.exists() else output.with_suffix(output.suffix + ".partial")
    )
    columns = (
        [[] for _ in range(7)] if overwrite else load_checkpoint(checkpoint, regions)
    )
    remaining = Counter(int(n) for n in columns[0])
    for path in paths:
        number = snapshot_number(path)
        if remaining[number]:
            remaining[number] -= 1
            continue
        snap = richio.load(path)
        time = snap.t.reshape(-1)[0]
        tfb = time / fallback_time
        r_a = r_p if time < 0 else r_amin * tfb ** (2 / 3)
        needs_switch = (
            path.parent.name == "TEMPTDE"
            if mode == 3
            else bool(re.fullmatch(r"snap_\d+\.h5", path.name))
        )
        x, y = snap.X, snap.Y
        if needs_switch:
            offset = dev.reference_frame_offset(
                t=time, Mbh=mbh, Mstar=mstar, Rstar=rstar, beta=1
            )
            x, y = x + offset[0], y + offset[1]
        power = snap.dissipation * snap.volume
        values = (
            number,
            time,
            tfb,
            *(
                np.sum(power[mask])
                for mask in region_masks(x, y, snap.Z, r_p, r_a, regions)
            ),
        )
        for column, value in zip(columns, values):
            column.append(value)
        save_timeseries(output, columns, regions, directories)
        logger.info("{} {}: powers={}", run, number, values[3:])
    if columns[0]:
        save_timeseries(output, columns, regions, directories)
    logger.info("Saved {} rows to {}", len(columns[0]), output)


if __name__ == "__main__":
    app()
