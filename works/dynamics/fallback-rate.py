r"""Measure the signed pericenter flux at every available snapshot.

Integrate ``-rho*vy`` over ``y=0, 0<x<5*r_p, -5*r_p<z<5*r_p`` in the
BH frame, including all gas and both velocity signs. Use nearest native-cell
values on a uniform ``nx`` by ``2*nx`` plane and lower-edge rectangular
quadrature. The default ``nx=1024`` gives ``dx=dz=5*r_p/1024``. The richio
default proximity filter is ``abs(y-y_plane)<Volume**(1/3)``. Outside the
stored simulation Box, extend the flux by zero rather than extrapolating cells.

Input files
-----------
``dev.DATAPATHS(run)``
    All snapshots from the current cooling runs; modes 1/2/3 select 1e4/1e5/1e6.
    Prefer ``snap_full_N.h5`` when both names exist. Full snapshots are in the
    BH frame; plain ``snap_N.h5`` use a translating frame in all three runs.
    Its parabolic orbit uses the simulation's G=1 and rs=4.21*M_BH/1e6.
    Required fields are Time, Box, X/Y/Z, Volume, Density and Vy.

Output files
------------
``data/processed/FallbackRate/flux-<run>.txt``
    ``np.loadtxt(path, ndmin=2)`` returns float64 shape ``(Nsnap, 7)``.
    Columns, numbered from zero: 0 snapshot number (integer-valued),
    1 time [code_time], 2 time/t_fb, 3 signed rate [Msun/yr],
    4 half-resolution signed rate [Msun/yr], 5 stored BH-frame flag
    (1=full, 0=plain), 6 fraction of sampled plane covered by the stored Box.
    Values are linear; negative rates mean net upward passage. Rows are sorted
    by time. This measures repeat crossings as well as first-return material.
    The earliest outgoing stellar passage is also retained.

``data/processed/FallbackRate/energy-<run>.txt``
    ``dev.tde.ballistic_fallback`` writes the two-column ballistic
    prediction from the snapshot nearest 0.5 t_fb: T/t_fb and mdot [Msun/yr].
    Its header records the native stellar-cell energy histogram and potential.

Usage
-----
From the repository root in the richanalysis environment::

    python works/dynamics/fallback-rate.py --mode 1
    sbatch jobs/submit-fallback-rate.sh

The batch job distributes each run across three nodes using ``--part`` and
``--parts`` (defaults 0 and 1 for a serial run). Each worker takes every
``parts``-th snapshot. Small checkpoints live under ``FallbackRate/parts``;
the job merges them into the same complete, time-sorted text tables above.

Rows are saved atomically after every snapshot. Reruns resume matching cached
rows and process newly available snapshots. Changed grid/source settings
require ``--overwrite`` or a separate ``--output-dir``. A newly available full
snapshot replaces a cached plain snapshot of the same number. The energy
cache independently checks its source/settings. For a small calculation::

    python works/dynamics/fallback-rate.py --mode 1 --end-snapshot 1

Loading examples
----------------
Restore units at the text boundary, then plot in the notebook::

    import numpy as np
    import unyt as u

    table = np.loadtxt("data/processed/FallbackRate/flux-1e4.txt", ndmin=2)
    time = u.unyt_array(table[:, 2], "dimensionless")
    rate = u.unyt_array(table[:, 3], "Msun/yr")
"""

# The local dev path and plotting-cache environment must precede dev imports.
# ruff: noqa: E402

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "dev"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, ".cache", "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import typer
import unyt as u
from dev.datapaths import DATADIRS, TDE_PARAMETERS
from dev.tde import ballistic_fallback, star_frame_offset

import dev
import richio

app = typer.Typer(add_completion=False)


def moving_frame_offset(run, time):
    """Return (dx, dy, dvy) of the star frame in the BH frame."""
    x, y, _, vy = star_frame_offset(run, float(time.to_value("code_time")))
    return (
        x * richio.units.lscale,
        y * richio.units.lscale,
        vy * richio.units.lscale / richio.units.tscale,
    )


def save_flux(output, rows, header):
    """Atomically replace the small time-sorted text checkpoint."""
    table = sorted(rows.values(), key=lambda row: row[1])
    np.savetxt(
        output + ".partial",
        table,
        header=header,
        fmt=["%d", "%.16e", "%.16e", "%.16e", "%.16e", "%d", "%.16e"],
    )
    os.replace(output + ".partial", output)


@app.command()
def main(
    mode: int = typer.Option(..., min=1, max=3),
    nx: int = typer.Option(1024, min=2, help="Plane resolution is (nx, 2*nx)."),
    output_dir: str = os.path.join(REPO, "data", "processed", "FallbackRate"),
    start_snapshot: int = 0,
    end_snapshot: int | None = None,
    overwrite: bool = False,
    part: int = typer.Option(0, min=0),
    parts: int = typer.Option(1, min=1),
):
    """Calculate the complete flux curve and the 0.5-t_fb energy prediction."""
    if nx % 2:
        raise typer.BadParameter("nx must be even for the half-resolution check")
    if part >= parts:
        raise typer.BadParameter("part must be smaller than parts")
    run = {1: "1e4", 2: "1e5", 3: "1e6"}[mode]
    mbh, mstar, rstar = TDE_PARAMETERS[run]
    M_BH = mbh * richio.units.mscale
    M_STAR = mstar * richio.units.mscale
    R_STAR = rstar * richio.units.lscale
    r_p = R_STAR * (M_BH / M_STAR) ** (1 / 3)
    t_fb = (
        np.pi
        / np.sqrt(2)
        * np.sqrt(R_STAR**3 / (M_STAR * u.G))
        * np.sqrt(M_BH / M_STAR)
    )
    header = (
        f"run={run}; nx={nx}; nz={2 * nx}; plane=y=0; x/rp=(0,5); z/rp=(-5,5)\n"
        f"sources={','.join(os.fspath(path) for path in DATADIRS[run])}\n"
        f"M_BH={mbh:g}; M_STAR={mstar:g}; R_STAR={rstar:g}; beta=1; G=unyt.G\n"
        "nearest_cell; volume_selection=True; lower_edge_rectangle; all_gas; outside_Box=0\n"
        "plain_frame=parabolic_unytG_rs4.21M/1e6; full_frame=BH\n"
        f"rp_code_length={float(r_p):.16e}; tfb_code_time={float(t_fb):.16e}\n"
        "snapshot time_code time_tfb mdot_Msunyr mdot_halfres_Msunyr stored_BH_frame coverage"
    )
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, f"flux-{run}.txt")
    rows = {}
    if os.path.exists(output) and not overwrite:
        expected = "".join(f"# {line}\n" for line in header.splitlines())
        with open(output) as handle:
            if handle.read(len(expected)) != expected:
                raise ValueError(
                    f"{output}: settings changed; use --overwrite or --output-dir"
                )
        rows = {int(row[0]): row.tolist() for row in np.loadtxt(output, ndmin=2)}

    ballistic_fallback(run, os.path.join(output_dir, f"energy-{run}.txt"))
    paths = dev.DATAPATHS(run)[part::parts]
    if parts > 1:
        numbers = {
            int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])
            for path in paths
        }
        rows = {number: row for number, row in rows.items() if number in numbers}
    for path in paths:
        number = int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])
        if number < start_snapshot or (
            end_snapshot is not None and number > end_snapshot
        ):
            continue
        full = os.path.basename(path).startswith("snap_full_")
        if number in rows and rows[number][5] == int(full):
            continue
        snap = richio.load(path)
        time = snap.time.squeeze()
        bounds = np.array([0, -5, 5, 5]) * r_p
        y_plane = 0 * richio.units.lscale
        dvy = 0 * richio.units.lscale / richio.units.tscale
        options = {}
        if not full:
            dx, dy, dvy = moving_frame_offset(run, time)
            bounds[[0, 2]] -= dx
            y_plane = -dy
            options["slice_coord"] = y_plane

        box = snap.box
        mdot = coarse = 0 * richio.units.mscale / richio.units.tscale
        coverage = 0.0
        if box[1] <= y_plane <= box[4]:
            spacing = 5 * r_p / nx
            x = np.linspace(bounds[0], bounds[2], nx, endpoint=False)
            z = np.linspace(bounds[1], bounds[3], 2 * nx, endpoint=False)
            ix = np.flatnonzero((x >= box[0]) & (x <= box[3]))
            iz = np.flatnonzero((z >= box[2]) & (z <= box[5]))
            coverage = len(ix) * len(iz) / (2 * nx**2)
            if len(ix) and len(iz):
                # Query only the covered portion of the same global grid.
                # Extrapolated samples outside Box have zero weight.
                query_bounds = bounds.copy()
                query_bounds[0], query_bounds[1] = x[ix[0]], z[iz[0]]
                query_bounds[2] = x[ix[-1]] + spacing
                query_bounds[3] = z[iz[-1]] + spacing
                indices, _, _ = snap.to_2dgrid(
                    (len(ix), len(iz)), plane="xz", box_size=query_bounds, **options
                )
                indices = indices.reshape(len(ix), len(iz))
                flux = -snap.density[indices] * (snap.vy[indices] + dvy)
                mdot = flux.sum() * spacing**2
                # Keep even GLOBAL indices for the half-resolution check.
                coarse = flux[ix[0] % 2 :: 2, iz[0] % 2 :: 2].sum() * 4 * spacing**2
                del indices, flux
            del x, z, ix, iz

        rows[number] = [
            number,
            float(time.to_value("code_time")),
            float((time / t_fb).to_value("dimensionless")),
            float(mdot.to_value("Msun/yr")),
            float(coarse.to_value("Msun/yr")),
            int(full),
            coverage,
        ]
        save_flux(output, rows, header)
        print(
            f"{run} snapshot {number}: t/tfb={rows[number][2]:.6f}, "
            f"mdot={rows[number][3]:.8g} Msun/yr; {len(rows)}/{len(paths)} saved",
            flush=True,
        )
        del snap
    if rows:
        save_flux(output, rows, header)
    print(f"Saved {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    app()
