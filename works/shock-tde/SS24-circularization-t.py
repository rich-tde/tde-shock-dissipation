r"""Prepare bound-orbital-energy and fallback tables for the SS24 diagnostic.

Compute BH-frame specific orbital energy with a Paczynski-Wiita potential
continuously matched to a harmonic core inside ``0.6*r_p``. Sum energy and
mass only where the specific orbital energy is negative; also sum all-cell
dissipation power. At snapshots 239 and 712, histogram bound ``dM/dE`` on
``[-2.5*Delta_epsilon_tidal, 0]`` with 2048 bins, smooth with a Gaussian of
sigma=3 bins, and convert energy to Keplerian return time and fallback rate.
These tables support a circularization analysis; this script does not compute
chi or equate shock power to loss of bound orbital energy.

Input files
-----------
``/data1/projects/pi-rossiem/TDE_data/SS24_diag/``
    Default input base, containing ``TEMPTDE``, ``TEMPTDE4`` and
    ``TEMPTDE4_new`` with ``snap_<n>.h5`` and ``snap_full_<n>.h5``.
    Uses the fixed ``1e6`` BH mass, ``1`` stellar mass, ``1`` stellar radius
    (in code units), beta=1 model. Required snapshot fields: positions,
    velocities, density, volume, dissipation and time. ``TEMPTDE`` is
    corrected to the BH frame; ``TEMPTDE4`` >=820 is excluded for its restart.
    Repeat ``--data-dir`` for relocated copies preserving these names.
``--merge-input PATH`` (repeatable)
    Optional existing time-series text files with the exact seven-column
    schema below. This mode reads no raw snapshots or fallback profiles.

Output files
------------
The default ``OUT`` is repository-relative
``data/processed/SS24-circularization-t`` (override ``--output-dir``).
Tables are tab-separated text written by unyt, with ``#`` name/unit/footer
comments. ``np.loadtxt(path, ndmin=2)`` returns a plain ``float64`` array,
without unit objects. All column indices below are zero-based.

``OUT/SS24-circularization-t-1e6.txt``
    Override with ``--timeseries-file``. Shape ``(N, 7)``: N completed
    snapshot occurrences, normally starting at snapshot 809. Worker rows
    follow file-processing order; merge mode sorts by snapshot number and
    requires strictly increasing times.

    Column 0, ``SNAPNUM``
        Integer-valued snapshot ID; cast to integer after loading.
    Column 1, ``TIME``
        Time in ``code_time``.
    Column 2, ``TIME_DAYS``
        The same time converted to days.
    Column 3, ``TFALLBACK``
        Dimensionless ``t/t_fb``.
    Column 4, ``EORB_BOUND``
        Total orbital energy of bound cells, in
        ``code_mass*code_length**2/code_time**2``.
    Column 5, ``MBOUND``
        Bound-cell mass in ``code_mass`` using the same energy mask.
    Column 6, ``EDISS_TOTAL``
        All-cell dissipation power in ``code_mass*code_length**2/code_time**3``.

``OUT/fallback-rate-snap239-0.40d.txt`` and
``OUT/fallback-rate-snap712-23.17d.txt``
    Shape ``(2048, 6)``; one row per energy bin, sorted in increasing
    Keplerian return time. The dates in the filenames label the source
    snapshots; column 0 is the debris return time, not that snapshot time.

    Column 0, ``RETURN_TIME``
        Keplerian return time in days, ``2*pi*G*Mbh/(2*abs(E))**1.5``.
    Column 1, ``SPECIFIC_ENERGY``
        Negative energy-bin centre in ``erg/g``.
    Column 2, ``DMDENERGY_RAW``
        Mass histogram divided by energy-bin width, in ``g**2/erg``.
    Column 3, ``DMDENERGY_SMOOTH``
        Gaussian-smoothed mass distribution, same units as column 2.
    Column 4, ``MDOT_RAW``
        Unsmoothed ``(dM/dE)*abs(dE/dt)`` in ``Msun/yr``.
    Column 5, ``MDOT_SMOOTH``
        Smoothed fallback rate in ``Msun/yr``.

    Zero histogram support can give zero fallback rate; no NaN mask is
    introduced here. Mass below ``-2.5*Delta_epsilon_tidal`` is excluded.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/SS24-circularization-t.py
    python works/shock-tde/SS24-circularization-t.py \
        --start-snapshot 923 --end-snapshot 950 --skip-fallback \
        --timeseries-file data/processed/SS24-circularization-t/shard-a.txt
    python works/shock-tde/SS24-circularization-t.py \
        --merge-input data/processed/SS24-circularization-t/shard-a.txt \
        --merge-input data/processed/SS24-circularization-t/shard-b.txt \
        --timeseries-file data/processed/SS24-circularization-t/merged.txt \
        --require-contiguous

The merge example assumes ``shard-b.txt`` has been generated for a subsequent
range. Repeat ``--merge-input`` for every wanted file, including an old main
checkpoint if it should be included. Identical duplicate rows are collapsed;
conflicting duplicates, wrong schemas and non-increasing times are rejected.
``--require-contiguous`` additionally rejects snapshot-number gaps. Replacing
an existing merge destination requires ``--overwrite``.

Workers resume existing rows by snapshot occurrence and reuse existing fallback
tables. ``--overwrite`` recomputes the selected products; ``--skip-fallback``
omits fallback work. Checkpoint writes are atomic. Use a new output when
changing sources/ranges; resuming assumes compatible previous settings.

Loading examples
----------------
Load bound energy and mass, and recover their physical units::

    from pathlib import Path
    import numpy as np
    import unyt as u
    import richio

    root = Path("data/processed/SS24-circularization-t")
    table = np.loadtxt(root / "SS24-circularization-t-1e6.txt", ndmin=2)
    # table is (N, 7); columns 1 and 2 contain the same time in different units.
    snapshot = table[:, 0].astype(int)
    time_days = table[:, 2]
    registry = richio.units.registry
    energy = u.unyt_array(
        table[:, 4], "code_mass*code_length**2/code_time**2",
        registry=registry,
    ).to("erg")
    mass = u.unyt_array(table[:, 5], richio.units.mscale).to("g")
    # Dividing these gives mean specific orbital energy of the bound gas.
    specific_energy = energy / mass  # (N,), erg/g

Load the late fallback profile for plotting or interpolation::

    fallback = np.loadtxt(root / "fallback-rate-snap712-23.17d.txt", ndmin=2)
    assert fallback.shape == (2048, 6)
    return_time_days = fallback[:, 0]
    mdot_msun_per_year = fallback[:, 5]  # Smoothed; raw is column 4.
    mdot_at_snapshot_times = np.interp(
        time_days, return_time_days, mdot_msun_per_year,
        left=np.nan, right=np.nan,
    )
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
from scipy.ndimage import gaussian_filter1d

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import dev
import richio

app = typer.Typer(add_completion=False)

DATADIRS = (
    "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
    "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
    "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
)
OUTPUT_DIR = "/home/hey4/rich_tde/data/processed/SS24-circularization-t"
TIMESERIES_FILE = os.path.join(OUTPUT_DIR, "SS24-circularization-t-1e6.txt")
FALLBACK_SNAPSHOTS = {
    239: os.path.join(OUTPUT_DIR, "fallback-rate-snap239-0.40d.txt"),
    712: os.path.join(OUTPUT_DIR, "fallback-rate-snap712-23.17d.txt"),
}

# Figure 2 begins at day 40.  Starting a few days earlier gives the numerical
# derivative enough padding while avoiding a second full pass over early data.
FIRST_TIMESERIES_SNAPSHOT = 809
FALLBACK_BINS = 2048
FALLBACK_SMOOTHING_BINS = 3.0

STELLAR_RADIUS = 1.0 * richio.units.lscale
STELLAR_MASS = 1.0 * richio.units.mscale
BLACK_HOLE_MASS = 1.0e6 * richio.units.mscale
PERICENTER_RADIUS = STELLAR_RADIUS * (BLACK_HOLE_MASS / STELLAR_MASS) ** (1 / 3)
TIDAL_ENERGY_SPREAD = u.G * BLACK_HOLE_MASS * STELLAR_RADIUS / PERICENTER_RADIUS**2
FALLBACK_TIME = (
    np.pi
    / np.sqrt(2)
    * (STELLAR_RADIUS**3 / u.G / STELLAR_MASS) ** (1 / 2)
    * (BLACK_HOLE_MASS / STELLAR_MASS) ** (1 / 2)
)

TIMESERIES_HEADER = (
    "SNAPNUM\tTIME\tTIME_DAYS\tTFALLBACK\tEORB_BOUND\tMBOUND\tEDISS_TOTAL"
)
TIMESERIES_FOOTER = (
    "EORB_BOUND = sum[(0.5*v_BH**2 + Phi_BH) * cell_mass] for orbital-energy-bound cells\n"
    "Phi_BH is the Paczynski-Wiita potential, continuously matched to the harmonic "
    "force used inside h = 0.6 r_p\n"
    "MBOUND uses the same specific-orbital-energy < 0 mask\n"
    "EDISS_TOTAL = sum(dissipation * volume); it is retained as a shock-power proxy\n"
    "TEMPTDE4 snapshots >= 820 are replaced by the TEMPTDE4_new high-resolution restart"
)


def snapshot_number(path):
    match = re.search(r"snap(?:_full)?_(\d+)\.h5", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot parse snapshot number from {path}")
    return int(match.group(1))


def snapshot_files(directory):
    full = sorted(
        glob.glob(os.path.join(directory, "snap_full_*.h5")),
        key=snapshot_number,
    )
    plain = sorted(
        (
            path
            for path in glob.glob(os.path.join(directory, "snap_*.h5"))
            if re.fullmatch(r"snap_\d+\.h5", os.path.basename(path))
        ),
        key=snapshot_number,
    )
    return full + plain


def load_existing_timeseries(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ([], [], [], [], [], [], [])

    with open(path) as handle:
        header = handle.readline().lstrip("# ").split()
    if header != TIMESERIES_HEADER.split():
        raise ValueError(f"{path} does not have the SS24 time-series column schema")
    raw = np.atleast_2d(np.loadtxt(path, delimiter="\t"))
    if raw.shape[1] != 7:
        raise ValueError(f"{path} has {raw.shape[1]} columns; expected exactly 7")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path} contains NaN or infinity")

    snapnums = raw[:, 0].astype(int).tolist()
    if not np.array_equal(raw[:, 0], snapnums):
        raise ValueError(f"{path} has a non-integer snapshot number")

    registry = richio.units.registry
    return (
        snapnums,
        list(u.unyt_array(raw[:, 1], "code_time", registry=registry)),
        list(u.unyt_array(raw[:, 2], "day")),
        list(u.unyt_array(raw[:, 3])),
        list(
            u.unyt_array(
                raw[:, 4],
                "code_length**2*code_mass/code_time**2",
                registry=registry,
            )
        ),
        list(u.unyt_array(raw[:, 5], "code_mass", registry=registry)),
        list(
            u.unyt_array(
                raw[:, 6],
                "code_length**2*code_mass/code_time**3",
                registry=registry,
            )
        ),
    )


def save_timeseries_atomic(path, arrays):
    temporary = f"{path}.tmp"
    u.savetxt(
        temporary,
        arrays=[u.unyt_array(array) for array in arrays],
        header=TIMESERIES_HEADER,
        footer=TIMESERIES_FOOTER,
    )
    os.replace(temporary, path)


def bh_frame(snapshot, snapshot_path, time):
    needs_switch = os.path.basename(os.path.dirname(snapshot_path)) == "TEMPTDE"
    if not needs_switch:
        return (
            snapshot.X,
            snapshot.Y,
            snapshot.Z,
            snapshot.vx,
            snapshot.vy,
            snapshot.vz,
        )

    offset = dev.reference_frame_offset(
        t=time,
        Mbh=BLACK_HOLE_MASS,
        Mstar=STELLAR_MASS,
        Rstar=STELLAR_RADIUS,
        beta=1,
    )
    return (
        snapshot.X + offset[0],
        snapshot.Y + offset[1],
        snapshot.Z,
        snapshot.vx + offset[2],
        snapshot.vy + offset[3],
        snapshot.vz,
    )


def orbital_specific_energy(snapshot, snapshot_path, time):
    x, y, z, vx, vy, vz = bh_frame(snapshot, snapshot_path, time)
    radius = np.sqrt(x**2 + y**2 + z**2)
    speed_squared = vx**2 + vy**2 + vz**2

    gravitational_radius = u.G * BLACK_HOLE_MASS / u.c**2
    softening_radius = 0.6 * PERICENTER_RADIUS
    potential = -u.G * BLACK_HOLE_MASS / (radius - 2 * gravitational_radius)
    inner = radius < softening_radius
    if np.any(inner):
        boundary_potential = (
            -u.G * BLACK_HOLE_MASS / (softening_radius - 2 * gravitational_radius)
        )
        omega_squared = (
            u.G
            * BLACK_HOLE_MASS
            / (softening_radius * (softening_radius - 2 * gravitational_radius) ** 2)
        )
        inner_potential = boundary_potential + 0.5 * omega_squared * (
            radius[inner] ** 2 - softening_radius**2
        )
        potential[inner] = inner_potential.to(potential.units)

    return (0.5 * speed_squared + potential).to("code_length**2/code_time**2")


def save_fallback_profile(path, snapnum, time, specific_energy, cell_mass):
    bound = (
        np.isfinite(specific_energy) & np.isfinite(cell_mass) & (specific_energy < 0)
    )
    energy = specific_energy[bound].to_value("erg/g")
    mass = cell_mass[bound].to_value("g")
    if energy.size == 0 or np.sum(mass) <= 0:
        raise ValueError(f"Snapshot {snapnum} has no finite bound mass")

    # Use one physical grid at both epochs.  Choosing each snapshot's most
    # negative cell as its lower edge is not robust: a negligible amount of
    # gas already near the softened central potential at 23.17 d otherwise
    # makes those bins tens of times wider than the 0.40 d bins.  The actual
    # first-return debris lies near -Delta_epsilon_tidal; -2.5 Delta safely
    # contains it at both epochs while excluding the irrelevant deep tail.
    lower_edge = (-2.5 * TIDAL_ENERGY_SPREAD).to_value("erg/g")
    excluded_mass = np.sum(mass[energy < lower_edge])
    if excluded_mass > 0:
        logger.info(
            "Snapshot {}: excluding {:.3e} Msun below -2.5 Delta_epsilon_tidal "
            "from the fallback histogram",
            snapnum,
            (excluded_mass * u.g).to_value("Msun"),
        )
    edges = np.linspace(lower_edge, 0.0, FALLBACK_BINS + 1)
    mass_per_bin, _ = np.histogram(energy, bins=edges, weights=mass)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    dmdenergy_raw = mass_per_bin / widths
    dmdenergy_smooth = gaussian_filter1d(
        dmdenergy_raw, FALLBACK_SMOOTHING_BINS, mode="nearest"
    )

    energy_quantity = u.unyt_array(centers, "erg/g")
    return_time = (
        2 * np.pi * u.G * BLACK_HOLE_MASS / (2 * np.abs(energy_quantity)) ** 1.5
    ).to("day")
    dedt = (
        (1 / 3)
        * (2 * np.pi * u.G * BLACK_HOLE_MASS) ** (2 / 3)
        * return_time.to("s") ** (-5 / 3)
    ).to("erg/g/s")
    dmdenergy_unit = "g/(erg/g)"
    mdot_raw = (u.unyt_array(dmdenergy_raw, dmdenergy_unit) * dedt).to("Msun/yr")
    mdot_smooth = (u.unyt_array(dmdenergy_smooth, dmdenergy_unit) * dedt).to("Msun/yr")

    order = np.argsort(return_time)
    temporary = f"{path}.tmp"
    u.savetxt(
        temporary,
        arrays=[
            return_time[order],
            energy_quantity[order],
            u.unyt_array(dmdenergy_raw, dmdenergy_unit)[order],
            u.unyt_array(dmdenergy_smooth, dmdenergy_unit)[order],
            mdot_raw[order],
            mdot_smooth[order],
        ],
        header=(
            "RETURN_TIME\tSPECIFIC_ENERGY\tDMDENERGY_RAW\tDMDENERGY_SMOOTH\t"
            "MDOT_RAW\tMDOT_SMOOTH"
        ),
        footer=(
            f"SS24 fallback estimate from snapshot {snapnum} at "
            f"{time.to_value('day'):.8f} day\n"
            "Mdot_fb = (dM/dE) * |dE/dt|, with Keplerian return time "
            "t(E) = 2*pi*G*Mbh/(2*|E|)^(3/2)\n"
            f"Linear energy histogram: {FALLBACK_BINS} bins; Gaussian smoothing "
            f"sigma = {FALLBACK_SMOOTHING_BINS} bins; shared range "
            "[-2.5 Delta_epsilon_tidal, 0]"
        ),
    )
    os.replace(temporary, path)
    logger.info(
        "Saved fallback profile from snapshot {} at {:.6f} day to {}",
        snapnum,
        time.to_value("day"),
        path,
    )


def merge_timeseries(paths, output, require_contiguous=False):
    """Merge matching-schema shards; reject conflicting rows and preserve units."""
    columns = [[] for _ in range(7)]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty merge input: {path}")
        for target, source in zip(columns, load_existing_timeseries(path)):
            target.extend(source)
    if not columns[0]:
        raise ValueError("No time-series rows to merge")
    raw = np.column_stack(columns)
    raw = raw[np.argsort(raw[:, 0], kind="stable")]
    kept = []
    for row in raw:
        if kept and row[0] == kept[-1][0]:
            if not np.array_equal(row, kept[-1]):
                raise ValueError(f"Conflicting rows for snapshot {int(row[0])}")
        else:
            kept.append(row)
    raw = np.asarray(kept)
    if np.any(np.diff(raw[:, 2]) <= 0):
        raise ValueError("Merged times are not strictly increasing")
    if require_contiguous and np.any(np.diff(raw[:, 0]) != 1):
        raise ValueError("Merged snapshot numbers contain gaps")
    units = [
        "dimensionless",
        "code_time",
        "day",
        "dimensionless",
        "code_length**2*code_mass/code_time**2",
        "code_mass",
        "code_length**2*code_mass/code_time**3",
    ]
    arrays = [
        u.unyt_array(values, unit, registry=richio.units.registry)
        for values, unit in zip(raw.T, units)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_timeseries_atomic(output, arrays)
    logger.info("Merged {} rows into {}", len(raw), output)


@app.command()
def main(
    start_snapshot: int = typer.Option(
        FIRST_TIMESERIES_SNAPSHOT, help="First snapshot included in the time series"
    ),
    end_snapshot: int = typer.Option(
        10_000, help="Last snapshot included in the time series"
    ),
    timeseries_file: Path | None = typer.Option(
        None,
        help="Checkpoint/output path; defaults to OUTPUT_DIR/SS24-circularization-t-1e6.txt.",
    ),
    output_dir: Path = typer.Option(
        Path(OUTPUT_DIR),
        help="Directory for fallback profiles and the default time series.",
    ),
    data_dir: list[Path] | None = typer.Option(
        None, help="Repeat for relocated copies of the 1e6 run/restart directories."
    ),
    merge_input: list[Path] | None = typer.Option(
        None, help="Merge only these tables; repeat per input. No snapshots loaded."
    ),
    require_contiguous: bool = typer.Option(
        False, help="Reject gaps in snapshot numbers when merging."
    ),
    overwrite: bool = typer.Option(
        False,
        help="Recompute selected products, or permit replacing a merge destination.",
    ),
    skip_fallback: bool = typer.Option(
        False, help="Do not build the two fallback profiles (for Slurm shards)"
    ),
):
    """Reproduce the SS24 fallback-normalized circularization diagnostic."""
    timeseries_file = timeseries_file or output_dir / "SS24-circularization-t-1e6.txt"
    if merge_input:
        if timeseries_file.exists() and not overwrite:
            raise typer.BadParameter(
                "Merge destination exists; use --overwrite or another --timeseries-file"
            )
        merge_timeseries(merge_input, timeseries_file, require_contiguous)
        return
    if end_snapshot < start_snapshot:
        raise typer.BadParameter("--end-snapshot must be at least --start-snapshot")
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_file.parent.mkdir(parents=True, exist_ok=True)
    fallback_paths = {
        number: output_dir / Path(path).name
        for number, path in FALLBACK_SNAPSHOTS.items()
    }
    arrays = (
        tuple([] for _ in range(7))
        if overwrite
        else load_existing_timeseries(timeseries_file)
    )
    (
        snapnums,
        times,
        times_days,
        tfallbacks,
        eorb_bounds,
        mbounds,
        ediss_totals,
    ) = arrays
    remaining_completed = Counter(snapnums)
    if snapnums:
        logger.info(
            "Resuming {} with {} completed rows", timeseries_file, len(snapnums)
        )

    for directory in data_dir or DATADIRS:
        logger.info("Scanning {}", directory)
        for snapshot_path in snapshot_files(directory):
            snapnum = snapshot_number(snapshot_path)
            if os.path.basename(directory) == "TEMPTDE4" and snapnum >= 820:
                continue

            fallback_path = fallback_paths.get(snapnum)
            needs_fallback = (
                not skip_fallback
                and fallback_path is not None
                and (overwrite or not os.path.exists(fallback_path))
            )
            needs_timeseries = start_snapshot <= snapnum <= end_snapshot
            if not needs_fallback and not needs_timeseries:
                continue
            if needs_timeseries and remaining_completed[snapnum] > 0:
                remaining_completed[snapnum] -= 1
                needs_timeseries = False
            if not needs_fallback and not needs_timeseries:
                continue

            snapshot = richio.load(snapshot_path)
            try:
                time = snapshot.t[0]
            except IndexError:
                time = snapshot.t
            time_day = time.to("day")
            specific_energy = orbital_specific_energy(snapshot, snapshot_path, time)
            cell_mass = snapshot.density * snapshot.volume

            if needs_fallback:
                save_fallback_profile(
                    fallback_path, snapnum, time_day, specific_energy, cell_mass
                )

            if needs_timeseries:
                bound = (
                    np.isfinite(specific_energy)
                    & np.isfinite(cell_mass)
                    & (specific_energy < 0)
                )
                bound_mass = np.sum(cell_mass[bound])
                bound_orbital_energy = np.sum(specific_energy[bound] * cell_mass[bound])
                dissipation_power = np.sum(snapshot.dissipation * snapshot.volume)

                snapnums.append(snapnum)
                times.append(time)
                times_days.append(time_day)
                tfallbacks.append(time / FALLBACK_TIME)
                eorb_bounds.append(bound_orbital_energy)
                mbounds.append(bound_mass)
                ediss_totals.append(dissipation_power)

                logger.info(
                    "snap={} day={:.6f} tfb={:.6f} Ebound={} Mbound={} Ediss={}",
                    snapnum,
                    time_day.to_value(),
                    (time / FALLBACK_TIME).to_value(),
                    bound_orbital_energy,
                    bound_mass,
                    dissipation_power,
                )
                save_timeseries_atomic(
                    timeseries_file,
                    [
                        snapnums,
                        times,
                        times_days,
                        tfallbacks,
                        eorb_bounds,
                        mbounds,
                        ediss_totals,
                    ],
                )


if __name__ == "__main__":
    app()
