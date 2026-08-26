import glob
import os
import re
from collections import Counter

import numpy as np
import typer
import unyt as u
from loguru import logger
from scipy.ndimage import gaussian_filter1d

import dev
import richio


app = typer.Typer()

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

RSTAR = 1.0 * richio.units.lscale
MSTAR = 1.0 * richio.units.mscale
MBH = 1.0e6 * richio.units.mscale
RP = RSTAR * (MBH / MSTAR) ** (1 / 3)
DELTA_EPSILON_TIDAL = u.G * MBH * RSTAR / RP**2
TMIN = (
    np.pi
    / np.sqrt(2)
    * (RSTAR**3 / u.G / MSTAR) ** (1 / 2)
    * (MBH / MSTAR) ** (1 / 2)
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
        t=time, Mbh=MBH, Mstar=MSTAR, Rstar=RSTAR, beta=1
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

    rg = u.G * MBH / u.c**2
    h = 0.6 * RP
    potential = -u.G * MBH / (radius - 2 * rg)
    inner = radius < h
    if np.any(inner):
        phi_h = -u.G * MBH / (h - 2 * rg)
        omega_squared = u.G * MBH / (h * (h - 2 * rg) ** 2)
        inner_potential = phi_h + 0.5 * omega_squared * (radius[inner] ** 2 - h**2)
        potential[inner] = inner_potential.to(potential.units)

    return (0.5 * speed_squared + potential).to(
        "code_length**2/code_time**2"
    )


def save_fallback_profile(path, snapnum, time, specific_energy, cell_mass):
    bound = np.isfinite(specific_energy) & np.isfinite(cell_mass) & (specific_energy < 0)
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
    lower_edge = (-2.5 * DELTA_EPSILON_TIDAL).to_value("erg/g")
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
        2 * np.pi * u.G * MBH / (2 * np.abs(energy_quantity)) ** 1.5
    ).to("day")
    dedt = (
        (1 / 3)
        * (2 * np.pi * u.G * MBH) ** (2 / 3)
        * return_time.to("s") ** (-5 / 3)
    ).to("erg/g/s")
    dmdenergy_unit = "g/(erg/g)"
    mdot_raw = (u.unyt_array(dmdenergy_raw, dmdenergy_unit) * dedt).to("Msun/yr")
    mdot_smooth = (
        u.unyt_array(dmdenergy_smooth, dmdenergy_unit) * dedt
    ).to("Msun/yr")

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


@app.command()
def main(
    start_snapshot: int = typer.Option(
        FIRST_TIMESERIES_SNAPSHOT, help="First snapshot included in the time series"
    ),
    end_snapshot: int = typer.Option(
        10_000, help="Last snapshot included in the time series"
    ),
    timeseries_file: str = typer.Option(
        TIMESERIES_FILE, help="Checkpoint/output path for this snapshot range"
    ),
    skip_fallback: bool = typer.Option(
        False, help="Do not build the two fallback profiles (for Slurm shards)"
    ),
):
    """Reproduce the SS24 fallback-normalized circularization diagnostic."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arrays = load_existing_timeseries(timeseries_file)
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
        logger.info("Resuming {} with {} completed rows", timeseries_file, len(snapnums))

    for directory in DATADIRS:
        logger.info("Scanning {}", directory)
        for snapshot_path in snapshot_files(directory):
            snapnum = snapshot_number(snapshot_path)
            if os.path.basename(directory) == "TEMPTDE4" and snapnum >= 820:
                continue

            fallback_path = FALLBACK_SNAPSHOTS.get(snapnum)
            needs_fallback = (
                not skip_fallback
                and fallback_path is not None
                and not os.path.exists(fallback_path)
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
                tfallbacks.append(time / TMIN)
                eorb_bounds.append(bound_orbital_energy)
                mbounds.append(bound_mass)
                ediss_totals.append(dissipation_power)

                logger.info(
                    "snap={} day={:.6f} tfb={:.6f} Ebound={} Mbound={} Ediss={}",
                    snapnum,
                    time_day.to_value(),
                    (time / TMIN).to_value(),
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
                    ]
                )


if __name__ == "__main__":
    app()
