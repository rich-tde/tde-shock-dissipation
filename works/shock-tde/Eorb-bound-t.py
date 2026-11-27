r"""Total orbital energy and mass of bound gas versus time, as in Steinberg & Stone (2024).

For each BH-frame snapshot, ``epsilon = v**2/2 + Phi`` with Phi the Paczynski-Wiita
potential ``-G*Mbh/(r - 2GMbh/c**2)``, continuously matched to a harmonic core inside
RICH's softening radius ``dev.tde.softening_radius`` (0.5 r_p for 1e6, 0.6 r_p otherwise;
for ss24 this is the definition of ``SS24-circularization-t.py``). Cells with
``epsilon < 0`` are summed: ``EORB_BOUND = sum(epsilon*m)``, ``MBOUND = sum(m)``.
Star-frame snapshots (``dev.tde.needs_frame_switch``) are skipped; all lie before 0.1 t_fb.
Extracted NPY snapshot folders (``dev.datapaths.NPY_RUNS``) are read through ``dev.positions_and_time``.

Output
------
``data/processed/EorbBound/Eorb-bound-t-<run>.txt``: tab-separated, ``#`` header, shape
``(N, 5)`` sorted by time. Columns: SNAPNUM, TIME [code_time], TFALLBACK [t/t_fb],
EORB_BOUND [code_mass*code_length**2/code_time**2], MBOUND [code_mass].
Rows are checkpointed after each snapshot and reused on rerun.

Usage
-----
    python works/shock-tde/Eorb-bound-t.py --run 1e6
"""

import os

import numpy as np
import typer
import unyt as u
from dev.datapaths import DATAPATHS, snapshot_number
from dev.helpers import positions_and_time
from dev.tde import needs_frame_switch, softening_radius, tde_basics
from loguru import logger

import richio

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "data/processed/EorbBound",
)
COLUMNS = {
    "SNAPNUM": "dimensionless",
    "TIME": "code_time",
    "TFALLBACK": "dimensionless",
    "EORB_BOUND": "code_mass*code_length**2/code_time**2",
    "MBOUND": "code_mass",
}


def potential(radius, m_bh, softening):
    r_g = 2 * u.G * m_bh / u.c**2
    phi = -u.G * m_bh / (np.maximum(radius, softening) - r_g)
    omega_squared = u.G * m_bh / (softening * (softening - r_g) ** 2)
    inner = radius < softening
    phi[inner] += 0.5 * omega_squared * (radius[inner] ** 2 - softening**2)
    return phi


def bound_energy_and_mass(snap, m_bh, softening):
    x, y, z, _ = positions_and_time(snap)
    radius = np.sqrt(x**2 + y**2 + z**2)
    epsilon = 0.5 * (snap.vx**2 + snap.vy**2 + snap.vz**2) + potential(
        radius, m_bh, softening
    )
    mass = snap.density * snap.volume
    bound = epsilon < 0
    return np.sum(epsilon[bound] * mass[bound]), np.sum(mass[bound])


def main(
    run: str = typer.Option(..., help="Run key in dev.datapaths, e.g. 1e4, 1e6, ss24."),
):
    tde = tde_basics(run)
    softening = softening_radius(run)
    output = os.path.join(OUTPUT_DIR, f"Eorb-bound-t-{run}.txt")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    rows = list(np.loadtxt(output, ndmin=2)) if os.path.exists(output) else []
    done = {int(row[0]) for row in rows}
    for path in DATAPATHS(run):
        number = snapshot_number(path)
        if number in done or needs_frame_switch(run, path):
            continue
        snap = richio.load(path)
        time = positions_and_time(snap)[3]
        energy, mass = bound_energy_and_mass(snap, tde.Mbh, softening)
        row = [
            number,
            time.to_value("code_time"),
            (time / tde.t_fb).to_value("dimensionless"),
            energy.to_value(COLUMNS["EORB_BOUND"]),
            mass.to_value("code_mass"),
        ]
        rows = sorted(rows + [row], key=lambda r: r[1])
        u.savetxt(
            output,
            [
                u.unyt_array(column, unit, registry=richio.units.registry)
                for column, unit in zip(np.transpose(rows), COLUMNS.values())
            ],
            header="\t".join(COLUMNS),
        )
        logger.info("{} snap {}: t/t_fb={:.3f} {}", run, number, row[2], row[3:])


if __name__ == "__main__":
    typer.run(main)
