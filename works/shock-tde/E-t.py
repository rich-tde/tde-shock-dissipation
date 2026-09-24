r"""Compute the total orbital, kinetic, gravitational, gas and radiation energies.

Sum energies over all cells without a boundness or stellar-tracer cut. Positions
and velocities are corrected to the BH frame when needed. ``EORB=EKIN+EGRAV``;
``ERAD=sum(Erad*mass)`` and ``EINT=sum(sie*mass)``. The historical quadratic
potential inside ``0.6*r_p`` differs from the continuously matched potential
in ``SS24-circularization-t.py``. Their energy zero points and bound masks
must not be interchanged.

Input files
-----------
``snap_full_<n>.h5`` and ``snap_<n>.h5`` in the selected mode's directories
    Read with ``richio.load``. Required fields: coordinates, velocities,
    ``Density``, ``Volume``, ``Erad`` (specific radiation energy),
    ``InternalEnergy`` (specific gas energy) and ``Time``.

The defaults under ``/data1/projects/pi-rossiem/TDE_data`` are:

* ``--mode 1``: ``NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes``.
* ``--mode 2``: ``YujieSnellius/`` followed by
  ``R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR``.
* ``--mode 3``: ``SS24_diag/TEMPTDE``, ``TEMPTDE4`` and ``TEMPTDE4_new``.

These correspond to BH masses ``1e4``, ``1e5`` and ``1e6`` solar masses. Repeat
``--data-dir`` for relocated copies of the same run. Preserve directory and
plain/full snapshot naming: they determine frame corrections. ``TEMPTDE4``
snapshots >=820 are excluded in favour of the high-resolution restart.

Output files
------------
``data/processed/SimpleTimeseries/E-t-<run>.txt``
    Default repository-relative output; ``--output`` overrides the path.
    Tab-separated numerical text, written by ``unyt.savetxt`` with ``#``
    comment lines for names and units. Load using
    ``np.loadtxt(path, ndmin=2)``: a ``float64`` array of shape ``(N, 8)``.
    N is the number of completed snapshot occurrences, not necessarily the
    number of unique snapshot IDs. Rows follow directory/file processing
    order; sort by column 1 before a time derivative or integration.

    Column 0, ``SNAPNUM``
        Integer-valued snapshot number; cast to integer after loading.
    Column 1, ``TIME``
        Simulation time in ``code_time``.
    Column 2, ``TFALLBACK``
        Dimensionless ``t/t_fb``.
    Column 3, ``EORB``
        Total orbital energy, ``EKIN+EGRAV``.
    Column 4, ``ERAD``
        Total stored radiation energy. An identically zero stored field is
        reported; radiation energy is not reconstructed.
    Column 5, ``EINT``
        Total gas internal energy.
    Column 6, ``EGRAV``
        Total gravitational energy in the potential described above.
    Column 7, ``EKIN``
        Total BH-frame kinetic energy.

    Current energy columns 3--7 use ``code_mass*code_length**2/code_time**2``.
    The registry ``richio.units.registry`` defines code units. NumPy loading
    returns numbers without units; reattach them explicitly as shown below.
    Some historical files use ``code_length**2*code_mass/s**2`` for EGRAV:
    the file's units comment is authoritative for those files.

Usage
-----
From ``/home/hey4/rich_tde``, using the ``richanalysis`` environment::

    python works/shock-tde/E-t.py --mode 1
    python works/shock-tde/E-t.py --mode 3 \
        --start-snapshot 820 --end-snapshot 825 \
        --output data/processed/EnergyCheck/1e6.txt

``--stride`` takes every Nth snapshot within each directory. Checkpoints are
written atomically after each snapshot; reruns resume by snapshot occurrence.
``--overwrite`` recomputes the selected series. Resume assumes unchanged
inputs, selection and current column units: use a fresh output or overwrite
for a legacy table with different units.

Loading examples
----------------
Load all eight columns and recover physical energies from the units header::

    from pathlib import Path
    import numpy as np
    import unyt as u
    import richio

    path = Path("data/processed/SimpleTimeseries/E-t-1e4.txt")
    table = np.loadtxt(path, ndmin=2)  # (N, 8), numerical values only
    snapshot = table[:, 0].astype(int)
    time_tfb = table[:, 2]
    with path.open() as stream:
        next(stream)  # Column names.
        next(stream)  # '# Units'.
        units = next(stream).lstrip("#").split()
    registry = richio.units.registry
    eorb = u.unyt_array(table[:, 3], units[3], registry=registry).to("erg")
    egrav = u.unyt_array(table[:, 6], units[6], registry=registry).to("erg")
    ekin = u.unyt_array(table[:, 7], units[7], registry=registry).to("erg")
    closure_error = eorb - (egrav + ekin)  # Unitful (N,) energy array.
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
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/NewSnellius/R0.47M0.5BH10000beta1S60ComptonHiRes",
        )
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e4.txt"
        cadence = 1
        stellar_radius = 0.47 * richio.units.lscale
        stellar_mass = 0.5 * richio.units.mscale
        black_hole_mass = 1e4 * richio.units.mscale
    elif mode == 2:
        # 1e5
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR",
        )
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e5.txt"
        cadence = 1
        stellar_radius = 0.47 * richio.units.lscale
        stellar_mass = 0.5 * richio.units.mscale
        black_hole_mass = 1e5 * richio.units.mscale
    elif mode == 3:
        # 1e6
        data_directories = (
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4",
            "/data1/projects/pi-rossiem/TDE_data/SS24_diag/TEMPTDE4_new",
        )
        output_file = "/home/hey4/rich_tde/data/processed/SimpleTimeseries/E-t-1e6.txt"
        cadence = 1
        stellar_radius = 1 * richio.units.lscale
        stellar_mass = 1 * richio.units.mscale
        black_hole_mass = 1e6 * richio.units.mscale
    else:
        raise ValueError("Invalid mode. Please choose 1, 2, or 3.")

    data_directories = tuple(str(p) for p in data_dir) if data_dir else data_directories
    output_file = str(output) if output is not None else output_file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    cadence = stride

    tmin = (
        np.pi
        / np.sqrt(2)
        * (stellar_radius**3 / u.G / stellar_mass) ** (1 / 2)
        * (black_hole_mass / stellar_mass) ** (1 / 2)
    )
    pericenter_radius = stellar_radius * (black_hole_mass / stellar_mass) ** (1 / 3)

    snapnums = []
    times = []
    fallback_times = []
    orbital_energies = []
    radiation_energies = []
    internal_energies = []
    gravitational_energies = []
    kinetic_energies = []
    warned_zero_erad = False

    columns = [
        snapnums,
        times,
        fallback_times,
        orbital_energies,
        radiation_energies,
        internal_energies,
        gravitational_energies,
        kinetic_energies,
    ]
    if (
        not overwrite
        and Path(output_file).exists()
        and Path(output_file).stat().st_size
    ):
        with open(output_file) as handle:
            header = handle.readline().lstrip("# ").split()
        if header != "SNAPNUM TIME TFALLBACK EORB ERAD EINT EGRAV EKIN".split():
            raise ValueError("Existing output has a different time-series schema")
        raw = np.loadtxt(output_file, ndmin=2)
        if raw.shape[1] != 8:
            raise ValueError("Existing output has the wrong number of columns")
        units = [
            "dimensionless",
            "code_time",
            "dimensionless",
            "code_length**2*code_mass/code_time**2",
            "code_length**2*code_mass/code_time**2",
            "code_length**2*code_mass/code_time**2",
            "code_length**2*code_mass/code_time**2",
            "code_length**2*code_mass/code_time**2",
        ]
        for column, values, unit in zip(columns, raw.T, units):
            column.extend(u.unyt_array(values, unit, registry=richio.units.registry))
    remaining_completed_snapshots = Counter(int(n) for n in snapnums)

    for data_directory in data_directories:
        logger.info(f"Processing directory: {data_directory}")
        snap_files = sorted(
            glob.glob(os.path.join(data_directory, "snap_full_*.h5")),
            key=lambda f: int(re.search(r"snap_full_(\d+)\.h5", f).group(1)),
        )
        plain_snap_files = [
            f
            for f in glob.glob(os.path.join(data_directory, "snap_*.h5"))
            if re.fullmatch(r"snap_\d+\.h5", os.path.basename(f))
        ]
        snap_files += sorted(
            plain_snap_files,
            key=lambda f: int(re.search(r"snap_(\d+)\.h5", f).group(1)),
        )

        for snap_file in snap_files[::cadence]:
            try:
                snapnum = int(re.search(r"snap_full_(\d+)\.h5", snap_file).group(1))
            except AttributeError:
                snapnum = int(re.search(r"snap_(\d+)\.h5", snap_file).group(1))

            if not start_snapshot <= snapnum <= end_snapshot:
                continue

            if (
                os.path.basename(data_directory) == "TEMPTDE4" and snapnum >= 820
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

            if mode == 3:
                needs_switch = os.path.basename(data_directory) == "TEMPTDE"
            else:
                needs_switch = bool(
                    re.fullmatch(r"snap_\d+\.h5", os.path.basename(snap_file))
                )

            if needs_switch:
                frame_offset = dev.reference_frame_offset(
                    t=t,
                    Mbh=black_hole_mass,
                    Mstar=stellar_mass,
                    Rstar=stellar_radius,
                    beta=1,
                )
                x = snap.X + frame_offset[0]
                y = snap.Y + frame_offset[1]
                velocity_x = snap.vx + frame_offset[2]
                velocity_y = snap.vy + frame_offset[3]
            else:
                x = snap.X
                y = snap.Y
                velocity_x = snap.vx
                velocity_y = snap.vy
            z = snap.Z
            velocity_z = snap.vz

            radius = np.sqrt(x**2 + y**2 + z**2)

            gravitational_radius = u.G * black_hole_mass / u.c**2

            speed_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
            density = snap.density
            volume = snap.volume

            softening_radius = 0.6 * pericenter_radius
            # smoothed PW
            cell_gravitational_energy = np.where(
                radius > softening_radius,
                -u.G
                * black_hole_mass
                * density
                * volume
                / (radius - 2 * gravitational_radius),
                -u.G
                * black_hole_mass
                * density
                * volume
                * radius**2
                / (
                    2
                    * softening_radius
                    * (softening_radius - 2 * gravitational_radius) ** 2
                ),
            )
            cell_kinetic_energy = 1 / 2 * speed_squared * density * volume
            cell_orbital_energy = cell_kinetic_energy + cell_gravitational_energy
            specific_radiation_energy = snap.Erad
            cell_radiation_energy = specific_radiation_energy * volume * density
            cell_internal_energy = snap.sie * volume * density

            kinetic_energy = np.sum(cell_kinetic_energy)
            orbital_energy = np.sum(cell_orbital_energy)
            # np.where may preserve Egrav_i in cgs time units.  Deriving the
            # total from Eorb = Egrav + Ekin keeps all output energies in the
            # same code-energy unit and guarantees exact budget closure.
            gravitational_energy = orbital_energy - kinetic_energy
            radiation_energy = np.sum(cell_radiation_energy)
            internal_energy = np.sum(cell_internal_energy)

            if not warned_zero_erad and np.all(specific_radiation_energy == 0):
                logger.warning(
                    "Erad is present but identically zero in {}; no stored "
                    "radiation energy is available for this snapshot",
                    snap_file,
                )
                warned_zero_erad = True

            snapnums.append(snapnum)
            times.append(t)
            fallback_times.append(tfb)
            orbital_energies.append(orbital_energy)
            radiation_energies.append(radiation_energy)
            internal_energies.append(internal_energy)
            gravitational_energies.append(gravitational_energy)
            kinetic_energies.append(kinetic_energy)

            logger.info(
                "snapnum={} t={} tfb={} Eorb={} Erad={} Eint={} Egrav={} Ekin={}",
                snapnum,
                t,
                tfb,
                orbital_energy,
                radiation_energy,
                internal_energy,
                gravitational_energy,
                kinetic_energy,
            )

            u.savetxt(
                f"{output_file}.tmp",
                arrays=[
                    u.unyt_array(snapnums),
                    u.unyt_array(times),
                    u.unyt_array(fallback_times),
                    u.unyt_array(orbital_energies),
                    u.unyt_array(radiation_energies),
                    u.unyt_array(internal_energies),
                    u.unyt_array(gravitational_energies),
                    u.unyt_array(kinetic_energies),
                ],
                header=("SNAPNUM\tTIME\tTFALLBACK\tEORB\tERAD\tEINT\tEGRAV\tEKIN"),
            )

            os.replace(f"{output_file}.tmp", output_file)


if __name__ == "__main__":
    app()
