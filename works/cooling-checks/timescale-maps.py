"""Make cooling-timescale diagnostic maps for the three TDE runs.

This is the batch-job version of ``0.1-timescales.ipynb``.  It keeps the
notebook's snapshot selections and adds surface-density, scale-height, and
photon-escape/vertical-compression-ratio panels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import dev  # noqa: F401  # isort: skip  # Configure style before pyplot.
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from loguru import logger

import richio

app = typer.Typer()
A_RAD = 4 * u.stefan_boltzmann_constant / u.c
OUTPUT_ROOT = Path("/home/hey4/rich_tde/data/processed/CoolingChecks/timescale-maps")


def mode_settings(mode: int):
    """Return data locations, selected snapshots, and stellar parameters."""
    if mode == 1:
        return {
            "label": "1e4",
            "runs": (
                (
                    (
                        "/data1/projects/pi-rossiem/TDE_data/NewSnellius/"
                        "R0.47M0.5BH10000beta1S60ComptonHiRes"
                    ),
                    range(21, 151, 10),
                ),
            ),
            "rstar": 0.47 * richio.units.lscale,
            "mstar": 0.5 * richio.units.mscale,
            "mbh": 1e4 * richio.units.mscale,
            "xy": 30,
        }
    if mode == 2:
        return {
            "label": "1e5",
            "runs": (
                (
                    (
                        "/data1/projects/pi-rossiem/TDE_data/YujieSnellius/"
                        "R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR"
                    ),
                    range(70, 150, 10),
                ),
            ),
            "rstar": 0.47 * richio.units.lscale,
            "mstar": 0.5 * richio.units.mscale,
            "mbh": 1e5 * richio.units.mscale,
            "xy": 100,
        }
    if mode == 3:
        return {
            "label": "1e6",
            "runs": (
                (
                    "/data1/projects/pi-rossiem/TDE_data/SS24/TEMPTDE4",
                    range(720, 820, 30),
                ),
                (
                    "/data1/projects/pi-rossiem/TDE_data/SS24/TEMPTDE4_new",
                    range(820, 1000, 50),
                ),
            ),
            "rstar": 1 * richio.units.lscale,
            "mstar": 1 * richio.units.mscale,
            "mbh": 1e6 * richio.units.mscale,
            "xy": 300,
        }
    raise ValueError("Invalid mode. Please choose 1, 2, or 3.")


def find_snapshot(datadir: str, snapnum: int) -> Path:
    """Resolve either snapshot naming convention used by the runs."""
    for filename in (f"snap_full_{snapnum}.h5", f"snap_{snapnum}.h5"):
        path = Path(datadir) / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No snapshot {snapnum} in {datadir}")


def plot_map(ax, field, xspace, yspace, label, **kwargs):
    richio.plots.scalar_map(
        field,
        xspace,
        yspace,
        ax=ax,
        cmap="magma",
        colorbar_label=label,
        **kwargs,
    )
    ax.set_xlabel(r"$x$ [code length]")
    ax.set_ylabel(r"$y$ [code length]")


def make_figure(snap_path: Path, snapnum: int, settings, output_dir: Path):
    snap = richio.load(str(snap_path))
    rstar = settings["rstar"]
    mstar = settings["mstar"]
    mbh = settings["mbh"]
    r_p = float((rstar * (mbh / mstar) ** (1 / 3)).in_units("code_length"))
    t_dyn = np.sqrt(rstar**3 / (u.G * mstar))

    alpha_ross_flat = richio.opacity.rosseland_alpha(snap.T, snap.rho)
    alpha_planck_flat = richio.opacity.planck_alpha(snap.T, snap.rho)

    xy = settings["xy"]
    box = [-xy, -xy, -10 * r_p, xy, xy, 10 * r_p]
    indices, xspace, yspace, zspace = snap.to_3dgrid(res=(256, 256, 512), box_size=box)

    cells = np.s_[:-1, :-1, :-1]
    rho = snap.rho[indices][cells]
    temperature = snap.T[indices][cells]
    vz = snap.vz[indices][cells]
    sie = snap.sie[indices][cells]
    alpha_ross = alpha_ross_flat[indices][cells]
    alpha_planck = alpha_planck_flat[indices][cells]

    dz = zspace[1:] - zspace[:-1]
    z = np.abs(zspace[:-1])
    surface_density = np.sum(rho * dz, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_height = np.sum(rho * z * dz, axis=-1) / surface_density
        t_cool = np.sum(rho * sie * dz, axis=-1) / np.sum(
            alpha_planck * A_RAD * temperature**4 * u.c * dz, axis=-1
        )
        tau = np.sum(alpha_ross * dz, axis=-1)
        t_escape = scale_height * tau / u.c

        kbin = np.argmin(
            np.abs(zspace[:-1][None, None, :] - scale_height[:, :, None]),
            axis=-1,
        )
        vz_scale_height = np.abs(
            np.take_along_axis(vz, kbin[:, :, None], axis=-1)[:, :, 0]
        )
        t_vertical = scale_height / vz_scale_height
        escape_vertical_ratio = t_escape / t_vertical

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    time_panels = (
        (axes[0, 0], t_cool / t_dyn, r"$\log_{10}(t_c/t_\mathrm{dyn,*})$"),
        (axes[0, 1], t_vertical / t_dyn, r"$\log_{10}(t_v/t_\mathrm{dyn,*})$"),
        (
            axes[0, 2],
            t_escape / t_dyn,
            r"$\log_{10}(t_\mathrm{es}/t_\mathrm{dyn,*})$",
        ),
        (axes[1, 2], escape_vertical_ratio, r"$\log_{10}(t_\mathrm{es}/t_v)$"),
    )
    for ax, field, label in time_panels:
        plot_map(ax, field, xspace, yspace, label, vmin=-1, vmax=1)

    plot_map(
        axes[1, 0],
        surface_density.in_units("g/cm**2"),
        xspace,
        yspace,
        r"$\log_{10}(\Sigma/[\mathrm{g\,cm^{-2}}])$",
    )
    plot_map(
        axes[1, 1],
        scale_height / rstar,
        xspace,
        yspace,
        r"$\log_{10}(H/R_*)$",
    )

    output_file = output_dir / f"timescales-{settings['label']}-snap-{snapnum:04d}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {output_file}")


@app.command()
def main(
    mode: int = typer.Option(..., help="Run 1e4, 1e5, or 1e6"),
    overwrite: bool = typer.Option(False, help="Replace figures that already exist"),
):
    settings = mode_settings(mode)
    output_dir = OUTPUT_ROOT / settings["label"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for datadir, snapnums in settings["runs"]:
        logger.info(f"Processing directory: {datadir}")
        for snapnum in snapnums:
            output_file = output_dir / (
                f"timescales-{settings['label']}-snap-{snapnum:04d}.png"
            )
            if output_file.exists() and not overwrite:
                logger.info(f"Skipping existing {output_file}")
                continue
            try:
                snap_path = find_snapshot(datadir, snapnum)
            except FileNotFoundError as error:
                logger.warning(str(error))
                continue
            make_figure(snap_path, snapnum, settings, output_dir)


if __name__ == "__main__":
    app()
