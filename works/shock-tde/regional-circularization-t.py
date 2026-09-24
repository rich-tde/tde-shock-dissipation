r"""Normalize regional dissipation powers by fallback circularization power.

For each region evaluate ``chi=Ediss/(Mdot_late*G*Mbh/(4*r_p))``. Compare
the total with normalization by an earlier fallback estimate to expose epoch
sensitivity. For the 1e4/1e5 runs, compute bound ``dM/dE`` in a continuously
matched Paczynski-Wiita/harmonic-core potential and convert it to Keplerian
fallback using 2048 bins and sigma=3-bin Gaussian smoothing. For 1e6, reuse
SS24 fallback tables. This regional shock-power proxy differs from the SS24
derivative of total bound orbital energy.

Input files
-----------
``data/processed/SimpleTimeseries/Ediss-t-<run>-final.txt``
    Produced by ``Ediss-t.py --regions standard``; override ``--ediss-dir``.
    A ``(N, 7)`` text table: columns 0/1/2 are snapshot/time/time divided by
    fallback time; 3/4/5/6 are pericenter/outgoing/incoming/outer powers in
    code units. Nozzle-split tables have a different schema and are rejected.
``dev.datapaths.DATAPATHS(<run>)`` snapshot HDF5 files
    Modes 1 and 2 load early/late snapshots 11/56 (1e4) and 38/153 (1e5).
    Requires positions, velocities, density, volume and time for bound energy
    and mass. Run paths/parameters come from ``dev/datapaths.py`` in the
    installed ``dev`` package (source: ``dev/dev/datapaths.py``).
``data/processed/SS24-circularization-t/fallback-rate-snap239-0.40d.txt``
``data/processed/SS24-circularization-t/fallback-rate-snap712-23.17d.txt``
    Required only for mode 3 (1e6), produced by ``SS24-circularization-t.py``.
    Override their common directory with ``--ss24-fallback-dir``. Both must
    be finite ``(2048, 6)`` tables on the same energy grid, as specified below.

Output files
------------
``OUT`` defaults to ``data/processed/RegionalCircularization`` relative to the
repository; override ``--output-root``. ``RUN`` is ``1e4``, ``1e5`` or ``1e6``.
Text files have ``#`` comments and load with ``np.loadtxt(path, ndmin=2)``
as plain ``float64`` arrays. Column indices below are zero-based.

``OUT/RUN/regional-chi-RUN.txt``
    Shape ``(N, 16)``, one row per unique snapshot. The last occurrence of
    each restart-overlap ID is retained and rows are sorted by time.

    Column 0, ``SNAPNUM``
        Integer-valued snapshot ID; cast to integer after loading.
    Column 1, ``TIME``
        Time in ``code_time``.
    Column 2, ``TIME_DAYS``
        The same time in days.
    Column 3, ``TFALLBACK``
        Dimensionless ``t/t_fb``.
    Columns 4, 5: ``MDOT_EARLY``, ``MDOT_LATE``
        Early/late fallback profiles interpolated at each time, in ``Msun/yr``.
    Columns 6, 7, 8, 9
        ``EDISS_PERICENTER``, ``EDISS_OUTGOING``, ``EDISS_INCOMING``,
        ``EDISS_OUTER``, respectively, in ``erg/s``. Regions are ``X>0``;
        ``-r_a<X<0,Y<0``; ``-r_a<X<0,Y>0``; and ``X<-r_a``.
    Columns 10, 11, 12, 13
        ``CHI_PERICENTER``, ``CHI_OUTGOING``, ``CHI_INCOMING``, ``CHI_OUTER``:
        dimensionless regional powers divided by the late normalization.
    Column 14, ``CHI_TOTAL``
        Sum of the four late-normalized regional chi values.
    Column 15, ``CHI_TOTAL_EARLY``
        Total regional power divided by the early normalization.

    Unsupported, nonfinite or nonpositive fallback rates and their chi values
    are NaN, never infinity. NumPy supplies numbers without units. The powers
    here are CGS, unlike the code-power inputs; chi is already dimensionless.

``OUT/RUN/fallback-early-snap<NNN>.txt``
``OUT/RUN/fallback-late-snap<NNN>.txt``
    Generated only for modes 1/2. Snapshot numbers are padded to three digits.
    Shape ``(2048, 6)``, one row per energy bin sorted by return time. This is
    also the schema of mode 3's input fallback tables.

    Column 0, ``RETURN_TIME``
        Keplerian return time in days, not the source snapshot time.
    Column 1, ``SPECIFIC_ENERGY``
        Energy-bin centre in ``erg/g``.
    Columns 2, 3: ``DMDENERGY_RAW``, ``DMDENERGY_SMOOTH``
        Raw/smoothed mass per unit specific energy in ``g**2/erg``.
    Columns 4, 5: ``MDOT_RAW``, ``MDOT_SMOOTH``
        Raw/smoothed fallback rates in ``Msun/yr``.

    The histogram covers ``[-2.5*Delta_epsilon_tidal, 0]``. Deeper-bound mass
    is excluded. Empty bins may have zero rates; no NaN mask is applied here.

Usage
-----
From ``/home/hey4/rich_tde`` with the ``richanalysis`` environment::

    python works/shock-tde/regional-circularization-t.py --mode 1
    python works/shock-tde/regional-circularization-t.py --mode 3 \
        --ediss-dir data/processed/SimpleTimeseries \
        --ss24-fallback-dir data/processed/SS24-circularization-t \
        --output-root data/processed/RegionalCircularization

Existing final chi tables are skipped unless ``--overwrite`` is given.
``--overwrite-fallback`` rebuilds mode 1/2 fallback profiles and also permits
rebuilding chi; mode 3 always reads the SS24 profiles. Writes are atomic.
Use a new output root after changing source data; caches are not content-hashed.

Loading examples
----------------
Load regional curves and compare their sum with the saved total::

    from pathlib import Path
    import numpy as np

    root = Path("data/processed/RegionalCircularization/1e4")
    table = np.loadtxt(root / "regional-chi-1e4.txt", ndmin=2)
    assert table.shape[1] == 16
    time_tfb = table[:, 3]
    power_erg_s = table[:, 6:10]  # (N, 4): pericenter, outgoing, incoming, outer
    chi_regions = table[:, 10:14]  # Same region order; dimensionless.
    chi_total = table[:, 14]
    valid = np.isfinite(chi_total)
    np.testing.assert_allclose(chi_regions[valid].sum(axis=1), chi_total[valid])

Load the fallback rate used to construct those curves::

    fallback = np.loadtxt(root / "fallback-late-snap056.txt", ndmin=2)
    return_time_days = fallback[:, 0]  # (2048,)
    mdot_msun_per_year = fallback[:, 5]  # Smoothed rate, (2048,).

Make a compact plot of one regional result::

    import dev
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(time_tfb[valid], chi_regions[valid, 0])
    ax.set(xlabel="t / t_fb", ylabel="Pericenter chi")
    plt.show()
"""

import os
import re
from pathlib import Path

import numpy as np
import typer
import unyt as u
from loguru import logger
from scipy.ndimage import gaussian_filter1d

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

from dev.datapaths import DATAPATHS

import dev
import richio

app = typer.Typer(add_completion=False)

REPO = Path("/home/hey4/rich_tde")
EDISS_DIR = REPO / "data/processed/SimpleTimeseries"
OUTPUT_ROOT = REPO / "data/processed/RegionalCircularization"
SS24_FALLBACK_DIR = REPO / "data/processed/SS24-circularization-t"

FALLBACK_BINS = 2048
FALLBACK_SMOOTHING_BINS = 3.0

HEADER = (
    "SNAPNUM\tTIME\tTIME_DAYS\tTFALLBACK\tMDOT_EARLY\tMDOT_LATE\t"
    "EDISS_PERICENTER\tEDISS_OUTGOING\tEDISS_INCOMING\tEDISS_OUTER\t"
    "CHI_PERICENTER\tCHI_OUTGOING\tCHI_INCOMING\tCHI_OUTER\t"
    "CHI_TOTAL\tCHI_TOTAL_EARLY"
)


def mode_settings(mode):
    settings = {
        1: {
            "run": "1e4",
            "Mbh": 1.0e4 * richio.units.mscale,
            "Mstar": 0.5 * richio.units.mscale,
            "Rstar": 0.47 * richio.units.lscale,
            "fallback_snapshots": {"early": 11, "late": 56},
        },
        2: {
            "run": "1e5",
            "Mbh": 1.0e5 * richio.units.mscale,
            "Mstar": 0.5 * richio.units.mscale,
            "Rstar": 0.47 * richio.units.lscale,
            "fallback_snapshots": {"early": 38, "late": 153},
        },
        3: {
            "run": "1e6",
            "Mbh": 1.0e6 * richio.units.mscale,
            "Mstar": 1.0 * richio.units.mscale,
            "Rstar": 1.0 * richio.units.lscale,
            "fallback_snapshots": {"early": 239, "late": 712},
        },
    }
    if mode not in settings:
        raise ValueError("Mode must be 1, 2, or 3")
    cfg = settings[mode]
    cfg["rp"] = cfg["Rstar"] * (cfg["Mbh"] / cfg["Mstar"]) ** (1 / 3)
    cfg["tmin"] = (
        np.pi
        / np.sqrt(2)
        * (cfg["Rstar"] ** 3 / u.G / cfg["Mstar"]) ** 0.5
        * (cfg["Mbh"] / cfg["Mstar"]) ** 0.5
    )
    cfg["delta_tidal"] = u.G * cfg["Mbh"] * cfg["Rstar"] / cfg["rp"] ** 2
    cfg["delta_circ"] = u.G * cfg["Mbh"] / (4 * cfg["rp"])
    return cfg


def snapshot_path(run, snapnum):
    snapnums, paths = DATAPATHS(run)
    matches = [path for number, path in zip(snapnums, paths) if number == snapnum]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {run} path for snapshot {snapnum}; got {matches}"
        )
    return matches[0]


def snapshot_time(snapshot):
    try:
        return snapshot.t[0]
    except IndexError:
        return snapshot.t


def bh_frame(snapshot, path, time, cfg):
    plain_snapshot = re.fullmatch(r"snap_\d+\.h5", path.name) is not None
    needs_switch = path.parent.name == "TEMPTDE" or (
        cfg["run"] != "1e6" and plain_snapshot
    )
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
        Mbh=cfg["Mbh"],
        Mstar=cfg["Mstar"],
        Rstar=cfg["Rstar"],
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


def orbital_specific_energy(snapshot, path, time, cfg):
    x, y, z, vx, vy, vz = bh_frame(snapshot, path, time, cfg)
    radius = np.sqrt(x**2 + y**2 + z**2)
    speed_squared = vx**2 + vy**2 + vz**2

    rg = u.G * cfg["Mbh"] / u.c**2
    h = 0.6 * cfg["rp"]
    potential = -u.G * cfg["Mbh"] / (radius - 2 * rg)
    inner = radius < h
    if np.any(inner):
        phi_h = -u.G * cfg["Mbh"] / (h - 2 * rg)
        omega_squared = u.G * cfg["Mbh"] / (h * (h - 2 * rg) ** 2)
        potential[inner] = (
            phi_h + 0.5 * omega_squared * (radius[inner] ** 2 - h**2)
        ).to(potential.units)

    return (0.5 * speed_squared + potential).to("code_length**2/code_time**2")


def save_fallback_profile(path, snapnum, time, specific_energy, cell_mass, cfg):
    bound = (
        np.isfinite(specific_energy) & np.isfinite(cell_mass) & (specific_energy < 0)
    )
    energy = specific_energy[bound].to_value("erg/g")
    mass = cell_mass[bound].to_value("g")
    if energy.size == 0 or not np.isfinite(mass).all() or np.sum(mass) <= 0:
        raise ValueError(f"Snapshot {snapnum} has no finite bound mass")

    lower_edge = (-2.5 * cfg["delta_tidal"]).to_value("erg/g")
    excluded_mass = np.sum(mass[energy < lower_edge]) * u.g
    if excluded_mass > 0 * u.g:
        logger.info(
            "{} snapshot {}: excluding {:.3e} Msun below -2.5 Delta_epsilon_tidal",
            cfg["run"],
            snapnum,
            excluded_mass.to_value("Msun"),
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
        2 * np.pi * u.G * cfg["Mbh"] / (2 * np.abs(energy_quantity)) ** 1.5
    ).to("day")
    dedt = (
        (1 / 3)
        * (2 * np.pi * u.G * cfg["Mbh"]) ** (2 / 3)
        * return_time.to("s") ** (-5 / 3)
    ).to("erg/g/s")
    dmdenergy_unit = "g/(erg/g)"
    mdot_raw = (u.unyt_array(dmdenergy_raw, dmdenergy_unit) * dedt).to("Msun/yr")
    mdot_smooth = (u.unyt_array(dmdenergy_smooth, dmdenergy_unit) * dedt).to("Msun/yr")

    order = np.argsort(return_time)
    temporary = path.with_suffix(path.suffix + ".tmp")
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
            f"{cfg['run']} fallback estimate from snapshot {snapnum} at "
            f"{time.to_value('day'):.8f} day = "
            f"{(time / cfg['tmin']).to_value():.8f} t_fb\n"
            "Mdot_fb = (dM/dE) * |dE/dt| with Keplerian return time\n"
            f"Linear histogram: {FALLBACK_BINS} bins; Gaussian sigma = "
            f"{FALLBACK_SMOOTHING_BINS} bins; range [-2.5 Delta_epsilon_tidal, 0]"
        ),
    )
    os.replace(temporary, path)
    logger.info("Saved {}", path)


def validate_fallback(path):
    raw = np.atleast_2d(np.loadtxt(path))
    if raw.shape != (FALLBACK_BINS, 6):
        raise ValueError(f"{path} has shape {raw.shape}; expected ({FALLBACK_BINS}, 6)")
    if not np.isfinite(raw).all() or np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError(f"{path} is non-finite or not ordered in return time")
    return raw


def fallback_paths(cfg, output_root=OUTPUT_ROOT, ss24_fallback_dir=SS24_FALLBACK_DIR):
    if cfg["run"] == "1e6":
        return {
            "early": ss24_fallback_dir / "fallback-rate-snap239-0.40d.txt",
            "late": ss24_fallback_dir / "fallback-rate-snap712-23.17d.txt",
        }
    output_dir = output_root / cfg["run"]
    return {
        epoch: output_dir / f"fallback-{epoch}-snap{snapnum:03d}.txt"
        for epoch, snapnum in cfg["fallback_snapshots"].items()
    }


def build_fallback_profiles(
    cfg, overwrite, output_root=OUTPUT_ROOT, ss24_fallback_dir=SS24_FALLBACK_DIR
):
    paths = fallback_paths(cfg, output_root, ss24_fallback_dir)
    if cfg["run"] == "1e6":
        for path in paths.values():
            validate_fallback(path)
        logger.info("Reusing the validated SS24 fallback profiles for 1e6")
        return paths

    for epoch, snapnum in cfg["fallback_snapshots"].items():
        output = paths[epoch]
        if output.exists() and not overwrite:
            validate_fallback(output)
            logger.info("Reusing {}", output)
            continue

        path = snapshot_path(cfg["run"], snapnum)
        logger.info("Loading {} {} fallback snapshot {}", cfg["run"], epoch, path)
        snapshot = richio.load(path)
        time = snapshot_time(snapshot)
        specific_energy = orbital_specific_energy(snapshot, path, time, cfg)
        cell_mass = snapshot.density * snapshot.volume
        save_fallback_profile(output, snapnum, time, specific_energy, cell_mass, cfg)
    return paths


def load_ediss(cfg, ediss_dir=EDISS_DIR):
    path = ediss_dir / f"Ediss-t-{cfg['run']}-final.txt"
    with path.open() as handle:
        header = handle.readline().lstrip("# ").split()
    if header != "SNAPNUM TIME TFALLBACK EDISS1 EDISS2 EDISS3 EDISS4".split():
        raise ValueError(f"{path} is not a standard-region Ediss table")
    raw = np.atleast_2d(np.loadtxt(path))
    if raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError(f"{path} has the wrong schema or non-finite values")
    snapnums = raw[:, 0].astype(int)
    if not np.array_equal(raw[:, 0], snapnums):
        raise ValueError(f"{path} contains a non-integer snapshot number")

    # Ediss-t-1e6 contains six restart-overlap rows.  Its directory order puts
    # TEMPTDE4_new after TEMPTDE4, so retaining the last occurrence selects the
    # high-resolution restart consistently with dev.datapaths.DATAPATHS.
    last_index = {}
    for index, snapnum in enumerate(snapnums):
        last_index[snapnum] = index
    raw = raw[list(last_index.values())]
    raw = raw[np.argsort(raw[:, 1], kind="stable")]
    if np.any(np.diff(raw[:, 1]) <= 0):
        raise ValueError(
            f"{path} is not strictly increasing after restart de-duplication"
        )
    return path, raw


def interpolate_mdot(profile, time_days):
    mdot = np.interp(
        time_days,
        profile[:, 0],
        profile[:, 5],
        left=np.nan,
        right=np.nan,
    )
    mdot[~np.isfinite(mdot) | (mdot <= 0)] = np.nan
    return mdot


def save_regional_chi(cfg, paths, ediss_dir=EDISS_DIR, output_root=OUTPUT_ROOT):
    source, raw = load_ediss(cfg, ediss_dir)
    registry = richio.units.registry
    time = u.unyt_array(raw[:, 1], "code_time", registry=registry)
    time_days = time.to_value("day")
    powers = u.unyt_array(
        raw[:, 3:7],
        "code_length**2*code_mass/code_time**3",
        registry=registry,
    ).to_value("erg/s")

    early_profile = validate_fallback(paths["early"])
    late_profile = validate_fallback(paths["late"])
    if not np.array_equal(early_profile[:, 1], late_profile[:, 1]):
        raise ValueError(f"{cfg['run']} fallback profiles do not share an energy grid")

    mdot_early = interpolate_mdot(early_profile, time_days)
    mdot_late = interpolate_mdot(late_profile, time_days)
    mdot_early_g_s = u.unyt_array(mdot_early, "Msun/yr").to_value("g/s")
    mdot_late_g_s = u.unyt_array(mdot_late, "Msun/yr").to_value("g/s")
    delta_circ = cfg["delta_circ"].to_value("erg/g")
    normalization_late = mdot_late_g_s * delta_circ
    normalization_early = mdot_early_g_s * delta_circ

    chi_regions = np.full_like(powers, np.nan, dtype=float)
    good_late = np.isfinite(normalization_late) & (normalization_late > 0)
    chi_regions[good_late] = powers[good_late] / normalization_late[good_late, None]
    chi_total = np.sum(chi_regions, axis=1)
    chi_total[~good_late] = np.nan
    chi_total_early = np.full(len(raw), np.nan)
    good_early = np.isfinite(normalization_early) & (normalization_early > 0)
    chi_total_early[good_early] = (
        np.sum(powers[good_early], axis=1) / normalization_early[good_early]
    )

    output = output_root / cfg["run"] / f"regional-chi-{cfg['run']}.txt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    arrays = [
        raw[:, 0].astype(int),
        raw[:, 1],
        time_days,
        raw[:, 2],
        mdot_early,
        mdot_late,
        *powers.T,
        *chi_regions.T,
        chi_total,
        chi_total_early,
    ]
    values = np.column_stack(arrays)
    if np.isinf(values).any():
        raise ValueError("Refusing to save infinity in the regional chi table")
    if not np.isfinite(values[:, [0, 1, 2, 3, 6, 7, 8, 9]]).all():
        raise ValueError("A required time or power column is non-finite")

    bs21_nozzle_ratio = (cfg["Mstar"] / cfg["Mbh"]) ** (2 / 3)
    np.savetxt(
        temporary,
        values,
        delimiter="\t",
        header=HEADER,
        footer=(
            f"Regional powers reused from {source}\n"
            "Regions: pericenter X>0; outgoing -r_a<X<0,Y<0; incoming "
            "-r_a<X<0,Y>0; outer X<-r_a\n"
            "CHI_REGION = EDISS_REGION / (MDOT_LATE * G*Mbh/(4*rp)); "
            "the four regions are additive\n"
            "CHI_TOTAL_EARLY uses MDOT_EARLY to expose fallback-epoch sensitivity\n"
            "NaN means the fallback profile has no positive support at that time; "
            "infinity is never written\n"
            f"Bonnerot & Stone (2021) beta=1 nozzle estimate = "
            f"{bs21_nozzle_ratio.to_value():.12e}"
        ),
    )
    os.replace(temporary, output)
    logger.info(
        "Saved {} rows to {}; {} have a finite late-normalized chi",
        len(values),
        output,
        np.count_nonzero(good_late),
    )


@app.command()
def main(
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6"),
    ediss_dir: Path = typer.Option(
        EDISS_DIR, help="Directory with standard-region Ediss-t-<run>-final.txt tables."
    ),
    ss24_fallback_dir: Path = typer.Option(
        SS24_FALLBACK_DIR,
        help="Directory with precomputed mode-3 SS24 fallback profiles.",
    ),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Root for per-run fallback and regional-chi tables."
    ),
    overwrite: bool = typer.Option(False, help="Recompute an existing chi table."),
    overwrite_fallback: bool = typer.Option(
        False, help="Recompute an existing 1e4/1e5 fallback profile"
    ),
):
    """Normalize existing regional Ediss-t powers by a dynamical fallback rate."""
    cfg = mode_settings(mode)
    destination = output_root / cfg["run"] / f"regional-chi-{cfg['run']}.txt"
    if destination.exists() and not (overwrite or overwrite_fallback):
        logger.info(
            "Existing chi table retained: {}; use --overwrite to replace", destination
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = build_fallback_profiles(
        cfg, overwrite_fallback, output_root, ss24_fallback_dir
    )
    save_regional_chi(cfg, paths, ediss_dir, output_root)


if __name__ == "__main__":
    app()
