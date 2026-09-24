"""Render broad-aperture cooling and vertical-flow maps from RICH snapshots.

Sample a Cartesian nearest-cell grid (default ``256x256x512``), integrate rho,
internal energy and opacities along z, and plot emission time tc, density-weighted
height H, surface density Sigma, vertical time ``tv=H/|vz(z=+H)|``, diffusion
estimate ``tes=H*tau_R/c``, and ``tes/tv``. Times are normalized by the stellar
dynamical time ``sqrt(R_star**3/(G*M_star))``. These preserve the original
``0.1-timescales`` notebook definitions: the velocity is sampled at +H and the
diffusion time has no light-crossing floor. This broad box diagnostic differs
from the density-weighted wedge calculation in ``nozzle_timescales.py``. A short
emission time alone does not establish effective cooling or a causal effect on
dissipation.

Input files
-----------
``--mode 1/2/3`` selects ``1e4/1e5/1e6``. Required ``richio`` snapshot quantities
are rho, T, vz and sie, with coordinates for nearest-cell gridding; opacity comes
from ``richio.opacity``. Prefer ``snap_full_<n>.h5``, then ``snap_<n>.h5`` in:

``--mode 1``
    ``/data1/projects/pi-rossiem/TDE_data/NewSnellius/``
    ``R0.47M0.5BH10000beta1S60ComptonHiRes``; snapshots 21..141, step 10.
``--mode 2``
    ``/data1/projects/pi-rossiem/TDE_data/YujieSnellius/``
    ``R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR``; 70..140, step 10.
``--mode 3``
    ``/data1/projects/pi-rossiem/TDE_data/SS24/TEMPTDE4`` for 720,750,780,810;
    ``.../SS24/TEMPTDE4_new`` for 820,870,920,970.

Adjacent path fragments above form one directory. ``--input-dir`` overrides
the root and repeated ``--snapshot-number`` chooses exact snapshots. Stored
coordinates are used directly, without moving-frame correction. Default x/y
half-widths are 30/100/300 code lengths by mode; default z half-width is
``10*r_p``. ``--xy-half-width`` and ``--z-half-width-rp`` override them.

Output files
------------
The only output is
``<output-root>/<run>/timescales-<run>-snap-<NNNN>.png``. Default
``--output-root`` is ``data/processed/CoolingChecks/timescale-maps``. There is
no NPY/NPZ/table or recoverable physical array in the output: load the PNG as an
image, not as a numerical timescale map.

``timescales-<run>-snap-<NNNN>.png`` : PNG raster
    Six panels saved at 200 dpi. Top row, left to right: log10(tc/t_dyn),
    log10(tv/t_dyn), log10(tes/t_dyn). Bottom row: log10(Sigma/[g/cm**2]),
    log10(H/R_star), log10(tes/tv). Axes are x/y in code lengths. Time/ratio
    panels use color limits [-1,1] in log10 space; colors outside those limits
    saturate. Undefined/non-positive values cannot be interpreted as finite
    logarithms. ``matplotlib.pyplot.imread`` returns float32 RGB/RGBA pixels in
    [0,1], shape ``(height, width, channels)`` with 3 or 4 channels. These image
    axes are row, column, color, not physical x, y.

Internally, sampling uses ``(Nx,Ny,Nz)`` grid points and drops the final point on
each axis before z integration, so physical column maps are
``(Nx-1,Ny-1)`` in ``(x,y)`` order; they are not persisted. No wedge mask or
source metadata table is saved.

Usage
-----
Run from ``/home/hey4/rich_tde`` in the richanalysis environment::

    python works/cooling-checks/timescale-maps.py --mode 1
    python works/cooling-checks/timescale-maps.py --mode 1 --snapshot-number 108 --resolution-xy 128 --resolution-z 256
    python works/cooling-checks/timescale-maps.py --mode 1 --input-dir data/external/my-run --snapshot-number 108

Existing figures are skipped unless ``--overwrite``; missing snapshots are
logged and skipped. Use overwrite or a distinct output root when changing grids,
inputs or extents, because filenames do not encode these settings. Large grids
should use ``jobs/submit-timescale-maps.sh``. For saved physical cooling arrays,
use ``nozzle-timescale-validation.py`` or ``nozzle-timescale-series.py`` and
account for their different scientific definitions.

Loading examples
----------------
Open the rendered figure as an image; no array transpose is needed::

    from pathlib import Path
    import dev
    import matplotlib.pyplot as plt

    path = Path("data/processed/CoolingChecks/timescale-maps/1e4")
    image = plt.imread(path / "timescales-1e4-snap-0108.png")
    print(image.shape, image.dtype)  # (height,width,RGB/RGBA), not a physical grid.
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_axis_off()
    plt.show()
"""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

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


def make_figure(
    snap_path: Path,
    snapnum: int,
    settings,
    output_dir: Path,
    resolution_xy: int = 256,
    resolution_z: int = 512,
    z_half_width_rp: float = 10,
    workers: int = 8,
):
    snap = richio.load(str(snap_path))
    rstar = settings["rstar"]
    mstar = settings["mstar"]
    mbh = settings["mbh"]
    r_p = float((rstar * (mbh / mstar) ** (1 / 3)).in_units("code_length"))
    t_dyn = np.sqrt(rstar**3 / (u.G * mstar))

    alpha_ross_flat = richio.opacity.rosseland_alpha(snap.T, snap.rho)
    alpha_planck_flat = richio.opacity.planck_alpha(snap.T, snap.rho)

    xy = settings["xy"]
    box = [-xy, -xy, -z_half_width_rp * r_p, xy, xy, z_half_width_rp * r_p]
    indices, xspace, yspace, zspace = snap.to_3dgrid(
        res=(resolution_xy, resolution_xy, resolution_z), box_size=box, workers=workers
    )

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
    mode: int = typer.Option(..., min=1, max=3, help="1: 1e4, 2: 1e5, 3: 1e6"),
    input_dir: Path | None = typer.Option(
        None, help="Override the mode's HDF5 directory"
    ),
    snapshot_number: list[int] | None = typer.Option(
        None, help="Exact snapshot number; repeat to select several"
    ),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Root for per-run PNG directories"
    ),
    resolution_xy: int = typer.Option(
        256, min=2, help="Grid points along each x/y axis"
    ),
    resolution_z: int = typer.Option(512, min=2, help="Grid points along z"),
    xy_half_width: float | None = typer.Option(
        None, help="x/y half-width in code lengths"
    ),
    z_half_width_rp: float = typer.Option(10, min=0.001, help="z half-width in r_p"),
    workers: int = typer.Option(8, min=1, help="Nearest-cell query threads"),
    overwrite: bool = typer.Option(False, help="Replace figures that already exist"),
):
    """Render six-panel legacy cooling maps; see the opening docstring for definitions."""
    settings = mode_settings(mode)
    if xy_half_width is not None:
        settings["xy"] = xy_half_width
    if input_dir is not None:
        selected = snapshot_number or [n for _, nums in settings["runs"] for n in nums]
        settings["runs"] = ((input_dir, selected),)
    elif snapshot_number:
        # Preserve the established mode-3 restart transition at snapshot 820.
        if mode == 3:
            settings["runs"] = tuple(
                (directory, [n for n in snapshot_number if (n < 820) == (i == 0)])
                for i, (directory, _) in enumerate(settings["runs"])
            )
        else:
            settings["runs"] = ((settings["runs"][0][0], snapshot_number),)
    output_dir = output_root / settings["label"]
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
            make_figure(
                snap_path,
                snapnum,
                settings,
                output_dir,
                resolution_xy,
                resolution_z,
                z_half_width_rp,
                workers,
            )


if __name__ == "__main__":
    app()
