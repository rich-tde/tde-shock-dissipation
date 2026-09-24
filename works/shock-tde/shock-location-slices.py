#!/usr/bin/env python3
"""Compare projected gas/dissipation structure with detected shock locations.

Produce XY and YZ three-panel figures: gas column density, line-of-sight
integrated volumetric dissipation, and a histogram of shock-surface cells.
Projection bounds match the conference movie windows in ``r_amin`` units.
Counts depend on resolution; they are not shock power or volume-weighted
integrals. Counts include all line-of-sight surface cells within the plotted
2-D bounds, while gas projections integrate a finite line-of-sight depth.

Input files
-----------
Let ``ROOT=/home/hey4/rich_tde/data/processed/ShockFinderEdissSelection``.

``ROOT/RUN/shockfinder_snap_NNNN.npz``
    Produced by ``shock-finder-ediss-selection.py``. Required keys are
    scalar ``run``, ``snapnum``, ``time_tfb``, ``time_code``, ``is_last``,
    ``snap_path``, and integer ``surf_idx``. ``snap_path`` locates the raw
    HDF5 used for density/dissipation projections when building a cache.
``ROOT/analysis/per-cell/RUN_shock_dissipation_snap_NNNN.npz``
    Produced by ``0.5-shock-finder-ediss-analysis.ipynb``. Required aligned
    1-D arrays are ``x_Rsun``, ``y_Rsun``, ``z_Rsun`` (positions in physical
    unyt solar radii), ``mach_T`` (dimensionless), ``shock_power_erg_s``
    (erg/s), plus scalar string ``snap_path``. Required at render time even
    if the projection cache already exists. Positions are corrected to the
    BH frame where needed before filtering/histogramming.

Override roots with ``--result-root`` and ``--per-cell-root``. ``--result``
selects an explicit detector result and also sets the detector root to its
parent's parent; ``--task-index`` instead selects from the sorted catalogue.

Output files
------------
Under ``ROOT/analysis/shock-locations/`` unless ``--output-root`` is supplied:

``RUN/grids/shock_locations_snap_NNNN.npz``
    Compressed NumPy archive; load with ``np.load``. ``NNNN`` is the
    zero-padded snapshot number. The following scalar metadata keys have
    shape ``()`` and use ``.item()``:

    ``run``, ``projection_method`` : Unicode
        Run label and literal ``richio.project``.
    ``snapnum``, ``projection_nz`` : int64
        Snapshot number and number of line-of-sight integration intervals
        (128, from 129 sample planes).
    ``time_tfb``, ``r_p_rsun`` : float64
        Dimensionless snapshot time/fallback time and pericentre radius in
        code-length units (``richio.units.lscale=7e10 cm``), respectively.
        Despite its name, ``r_p_rsun`` is not an exact unyt ``Rsun`` conversion.
    ``is_last`` : bool
        Whether the detector input was selected as the final snapshot.

    Each key below occurs twice, with ``PLANE=xy`` or ``PLANE=yz``. Let
    ``(A, B)=(x, y)`` for XY and ``(A, B)=(y, z)`` for YZ. Arrays use
    ``[i, j]=(A-bin i, B-bin j)``; transpose for ``pcolormesh``. ``Nx, Ny``
    count projection pixels, ``Hx, Hy`` count histogram bins. Resolutions
    depend on window aspect ratio; projection longest axis is 512 pixels
    and shock histogram longest axis is 256 bins.

    ``PLANE_x_edges``, ``PLANE_y_edges`` : float64, shapes ``(Nx+1,)``, ``(Ny+1,)``
        Bin edges for projected fields in BH-frame ``A/r_p`` and ``B/r_p``.
        The suffixes x/y mean horizontal/vertical axes, even for YZ.
    ``PLANE_density_column`` : float64, shape ``(Nx, Ny)``
        Linear column density [g/cm^2], integrated along the normal axis.
    ``PLANE_dissipation_column`` : float64, shape ``(Nx, Ny)``
        Linear projected dissipation [erg/s/cm^2]. Negative/nonfinite raw
        dissipation is replaced by zero before integration.
    ``PLANE_shock_x_edges``, ``PLANE_shock_y_edges`` : float64, shapes ``(Hx+1,)``, ``(Hy+1,)``
        Histogram edges in the same dimensionless coordinate convention.
    ``PLANE_shock_count`` : float64, shape ``(Hx, Hy)``
        Unfiltered number of detector surface cells per bin, stored as
        integer-valued floats. No Mach/power filter is applied to this key.
    ``PLANE_surface_total``, ``PLANE_surface_in_view`` : int64, shape ``()``
        Surface cells with finite projected coordinates, and their count
        inside the 2-D histogram bounds, respectively.

    All saved numerical grids are linear, not log10; there is no validity
    mask. Nonfinite shock coordinates are excluded. Earlier caches without
    ``projection_method``/histogram edges are rebuilt by the current CLI.

``RUN/xy/shock_locations_snap_NNNN.png``, ``RUN/yz/shock_locations_snap_NNNN.png``
    Raster figures; dimensions depend on view and ``--dpi``. Unlike the
    cached count grid, the displayed histogram is recomputed from the
    per-cell input with finite coordinates/Mach/power, ``mach_T >= 1.8``
    and ``shock_power_erg_s >= 0`` by default. Change thresholds with
    ``--shock-mach-min``/``--shock-power-min``. Filtered grids are not saved.
    The per-cell renderer divides physical-Rsun coordinates by legacy
    ``r_p_rsun`` code-length values; this introduces an approximately 0.6%
    scale difference from the cached histogram. No correction is applied
    here. Figures use logarithmic colour normalization, not logged data.

Usage
-----
Run from ``/home/hey4/rich_tde`` with the richanalysis Python environment::

    python works/shock-tde/shock-location-slices.py --list-only
    python works/shock-tde/shock-location-slices.py --task-index 0 --workers 8
    python works/shock-tde/shock-location-slices.py --task-index 0 --shock-mach-min 2 --rerender

Complete caches and figures are skipped; ``--overwrite`` rebuilds both and
``--rerender`` reuses projections but reapplies display filters. Use
``--rerender`` after changing filters/DPI and a distinct root to keep variants.

Loading examples
----------------
Load a current-schema cache and plot its unfiltered XY surface-cell counts::

    from pathlib import Path
    import dev
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path("data/processed/ShockFinderEdissSelection/analysis/shock-locations")
    with np.load(root / "1e4/grids/shock_locations_snap_0151.npz") as data:
        x = data["xy_shock_x_edges"]
        y = data["xy_shock_y_edges"]
        count = data["xy_shock_count"]
        in_view = data["xy_surface_in_view"].item()
    assert count.shape == (len(x) - 1, len(y) - 1)
    print(count.sum(), in_view)  # Matching unfiltered cell counts.
    fig, ax = plt.subplots()
    image = ax.pcolormesh(x, y, count.T, shading="flat")
    ax.set(xlabel="x/r_p", ylabel="y/r_p", aspect="equal")
    fig.colorbar(image, ax=ax, label="Surface cells per bin (unfiltered)")
    plt.show()
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev

# Apply the repository plotting style before importing pyplot.
# isort: split

import matplotlib.pyplot as plt
import numpy as np
import typer
from dev.datapaths import TDE_PARAMETERS
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import richio

RESULT_ROOT = Path("/home/hey4/rich_tde/data/processed/ShockFinderEdissSelection")
OUTPUT_ROOT = RESULT_ROOT / "analysis" / "shock-locations"
PER_CELL_ROOT = RESULT_ROOT / "analysis" / "per-cell"
DEFAULT_SHOCK_MACH_MIN = 1.8


@dataclass(frozen=True)
class RunConfig:
    run: str
    m_bh: float
    m_star: float
    r_star: float

    @property
    def r_p(self) -> float:
        return self.r_star * (self.m_bh / self.m_star) ** (1 / 3)

    @property
    def r_a(self) -> float:
        return self.r_star * (self.m_bh / self.m_star) ** (2 / 3)


RUNS = {run: RunConfig(run, *TDE_PARAMETERS[run]) for run in ("1e4", "1e5", "1e6")}


def selected_results() -> list[Path]:
    return sorted(RESULT_ROOT.glob("1e*/shockfinder_snap_*.npz"))


# Keep these synchronized with the three-mass conference movies in
# works/movies/faceon_density.py. Values are in units of r_amin and ordered as
# (xmin, xmax, ymin, ymax, zmin, zmax).
MOVIE_WINDOW = (-1.5, 0.5, -0.7, 0.7, -0.7, 0.7)
MOVIE_WINDOW_OVERRIDES = {
    "1e4": (-2.0, 0.5, -0.875, 0.875, -0.7, 0.7),
}


def needs_reference_frame(run: str, path: Path) -> bool:
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def cache_path(run: str, snapnum: int) -> Path:
    return OUTPUT_ROOT / run / "grids" / f"shock_locations_snap_{snapnum:04d}.npz"


def cache_complete(path: Path) -> bool:
    required = {
        f"{plane}_{field}"
        for plane in ("xy", "yz")
        for field in (
            "x_edges",
            "y_edges",
            "shock_x_edges",
            "shock_y_edges",
            "density_column",
            "dissipation_column",
            "shock_count",
            "surface_total",
            "surface_in_view",
        )
    }
    try:
        with np.load(path) as data:
            return (
                required.issubset(data.files)
                and str(data["projection_method"].item()) == "richio.project"
            )
    except (FileNotFoundError, OSError, ValueError):
        return False


def projection_resolution(
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bounds
    aspect = (xmax - xmin) / (ymax - ymin)
    if aspect >= 1:
        return 512, max(300, round(512 / aspect))
    return max(300, round(512 * aspect)), 512


def shock_histogram_resolution(
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bounds
    aspect = (xmax - xmin) / (ymax - ymin)
    if aspect >= 1:
        return 256, max(150, round(256 / aspect))
    return max(150, round(256 * aspect)), 256


def movie_window_bounds(
    config: RunConfig,
) -> tuple[float, float, float, float, float, float]:
    window = MOVIE_WINDOW_OVERRIDES.get(config.run, MOVIE_WINDOW)
    r_amin_over_r_p = config.r_a / config.r_p
    return tuple(value * r_amin_over_r_p for value in window)


def projected_view(
    snap,
    first,
    second,
    normal,
    surface_indices: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    r_p,
    workers: int,
) -> dict[str, np.ndarray]:
    xmin, ymin, _, xmax, ymax, _ = bounds
    nx, ny = projection_resolution((xmin, ymin, xmax, ymax))
    projection_kwargs = {
        "res": (nx + 1, ny + 1, 129),
        "X": first,
        "Y": second,
        "Z": normal,
        "box_size": tuple(value * r_p for value in bounds),
        "unit_system": "cgs",
        "workers": workers,
        "spacing": ("linear", "linear", "sinh"),
        "sinh_scale": (None, None, 0.1 * richio.units.lscale),
    }
    density_column, xspace, yspace = snap.project(
        snap.density,
        **projection_kwargs,
    )

    dissipation = snap.dissipation.copy()
    dissipation[(dissipation < 0) | ~np.isfinite(dissipation)] = 0
    dissipation_column, dissipation_xspace, dissipation_yspace = snap.project(
        dissipation,
        **projection_kwargs,
    )
    if not (
        np.allclose(xspace, dissipation_xspace)
        and np.allclose(yspace, dissipation_yspace)
    ):
        raise RuntimeError("Density and dissipation projection grids do not match")

    x_edges = np.asarray((xspace / r_p).to_value(), dtype="float64")
    y_edges = np.asarray((yspace / r_p).to_value(), dtype="float64")
    shock_nx, shock_ny = shock_histogram_resolution((xmin, ymin, xmax, ymax))
    shock_x_edges = np.linspace(x_edges[0], x_edges[-1], shock_nx + 1)
    shock_y_edges = np.linspace(y_edges[0], y_edges[-1], shock_ny + 1)

    shock_first = np.asarray((first[surface_indices] / r_p).to_value(), dtype="float64")
    shock_second = np.asarray(
        (second[surface_indices] / r_p).to_value(), dtype="float64"
    )
    finite_surface = np.isfinite(shock_first) & np.isfinite(shock_second)
    shock_count, _, _ = np.histogram2d(
        shock_first[finite_surface],
        shock_second[finite_surface],
        bins=(shock_x_edges, shock_y_edges),
    )
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "shock_x_edges": shock_x_edges,
        "shock_y_edges": shock_y_edges,
        "density_column": np.asarray(
            density_column.to_value("g/cm**2"), dtype="float64"
        ),
        "dissipation_column": np.asarray(
            dissipation_column.to_value("erg/s/cm**2"), dtype="float64"
        ),
        "shock_count": shock_count,
        "surface_total": np.asarray(int(finite_surface.sum())),
        "surface_in_view": np.asarray(int(shock_count.sum())),
    }


def build_cache(result_path: Path, destination: Path, workers: int) -> None:
    with np.load(result_path) as result:
        run = str(result["run"].item())
        snapnum = int(result["snapnum"])
        snap_path = Path(str(result["snap_path"].item()))
        time_tfb = float(result["time_tfb"])
        is_last = bool(result["is_last"])
        surface_indices = np.asarray(result["surf_idx"], dtype=np.intp)

    config = RUNS[run]
    snap = richio.load(str(snap_path))
    x, y, z = snap.X, snap.Y, snap.Z
    if needs_reference_frame(run, snap_path):
        time = snap.t.reshape(-1)[0] if getattr(snap.t, "ndim", 0) else snap.t
        offset = dev.reference_frame_offset(
            t=time,
            Mbh=config.m_bh * richio.units.mscale,
            Mstar=config.m_star * richio.units.mscale,
            Rstar=config.r_star * richio.units.lscale,
            beta=1,
        )
        x = x + offset[0]
        y = y + offset[1]

    r_p = config.r_p * richio.units.lscale
    xmin, xmax, ymin, ymax, zmin, zmax = movie_window_bounds(config)

    arrays: dict[str, np.ndarray] = {
        "run": np.asarray(run),
        "snapnum": np.asarray(snapnum),
        "time_tfb": np.asarray(time_tfb),
        "is_last": np.asarray(is_last),
        "r_p_rsun": np.asarray(config.r_p),
        "projection_method": np.asarray("richio.project"),
        "projection_nz": np.asarray(128),
    }
    for plane, first, second, normal, bounds in (
        ("xy", x, y, z, (xmin, ymin, zmin, xmax, ymax, zmax)),
        ("yz", y, z, x, (ymin, zmin, xmin, ymax, zmax, xmax)),
    ):
        print(
            f"[{run} snap {snapnum}] projecting density/dissipation with richio "
            f"and binning shock cells in {plane}",
            flush=True,
        )
        view = projected_view(
            snap,
            first,
            second,
            normal,
            surface_indices,
            bounds,
            r_p,
            workers,
        )
        for field, values in view.items():
            arrays[f"{plane}_{field}"] = values

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def top_decades(values: np.ndarray, decades: float = 6) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    vmax = 10 ** (math.ceil(2 * np.log10(positive).max()) / 2)
    return vmax / 10**decades, vmax


def filtered_shock_count(
    cache_data,
    plane: str,
    mach_min: float,
    power_min: float,
) -> tuple[np.ndarray, int, float]:
    run = str(cache_data["run"].item())
    snapnum = int(cache_data["snapnum"])
    r_p_rsun = float(cache_data["r_p_rsun"])
    per_cell_path = PER_CELL_ROOT / f"{run}_shock_dissipation_snap_{snapnum:04d}.npz"
    if not per_cell_path.is_file():
        raise FileNotFoundError(
            f"Missing per-cell shock product needed for filtering: {per_cell_path}"
        )

    with np.load(per_cell_path) as cells:
        mach = np.asarray(cells["mach_T"], dtype="float64")
        power = np.asarray(cells["shock_power_erg_s"], dtype="float64")
        x = np.asarray(cells["x_Rsun"], dtype="float64")
        y = np.asarray(cells["y_Rsun"], dtype="float64")
        z = np.asarray(cells["z_Rsun"], dtype="float64")
        snap_path = Path(str(cells["snap_path"].item()))

    if needs_reference_frame(run, snap_path):
        result_path = RESULT_ROOT / run / f"shockfinder_snap_{snapnum:04d}.npz"
        with np.load(result_path) as result:
            time = float(result["time_code"]) * richio.units.tscale
        config = RUNS[run]
        offset = dev.reference_frame_offset(
            t=time,
            Mbh=config.m_bh * richio.units.mscale,
            Mstar=config.m_star * richio.units.mscale,
            Rstar=config.r_star * richio.units.lscale,
            beta=1,
        )
        x += offset[0].to_value("Rsun")
        y += offset[1].to_value("Rsun")

    first, second = (x, y) if plane == "xy" else (y, z)
    selected = (
        np.isfinite(first)
        & np.isfinite(second)
        & np.isfinite(mach)
        & np.isfinite(power)
        & (mach >= mach_min)
        & (power >= power_min)
    )
    shock_count, _, _ = np.histogram2d(
        first[selected] / r_p_rsun,
        second[selected] / r_p_rsun,
        bins=(
            cache_data[f"{plane}_shock_x_edges"],
            cache_data[f"{plane}_shock_y_edges"],
        ),
    )
    retained_power = float(power[selected].sum() / power.sum())
    return shock_count, int(selected.sum()), retained_power


def compact_colorbar(
    fig,
    ax,
    image,
    label: str,
    text_color: str = "black",
) -> None:
    cax = inset_axes(ax, width="50%", height="5%", loc="lower left", borderpad=0.8)
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label(label, fontsize=9, labelpad=1, color=text_color)
    bar.outline.set_edgecolor(text_color)
    cax.tick_params(
        which="both",
        labelsize=8,
        length=2,
        pad=1,
        colors=text_color,
    )
    cax.xaxis.set_label_position("top")


def render_plane(
    cache: Path,
    plane: str,
    destination: Path,
    dpi: int,
    shock_mach_min: float,
    shock_power_min: float,
) -> None:
    with np.load(cache) as data:
        shock_x_edges = np.asarray(data[f"{plane}_shock_x_edges"])
        shock_y_edges = np.asarray(data[f"{plane}_shock_y_edges"])
        x_edges = np.asarray(data[f"{plane}_x_edges"])
        y_edges = np.asarray(data[f"{plane}_y_edges"])
        density_column = np.asarray(data[f"{plane}_density_column"])
        dissipation_column = np.asarray(data[f"{plane}_dissipation_column"])
        shock_count, selected_shocks, retained_power = filtered_shock_count(
            data,
            plane,
            shock_mach_min,
            shock_power_min,
        )

    print(
        f"  {plane}: {selected_shocks} cells after M_T >= {shock_mach_min:g}, "
        f"power >= {shock_power_min:.3g} erg/s; "
        f"retained shock power {retained_power:.3%}",
        flush=True,
    )

    axis_names = ("x", "y") if plane == "xy" else ("y", "z")
    aspect = (x_edges[-1] - x_edges[0]) / (y_edges[-1] - y_edges[0])
    panel_width = 3.9
    figure_height = max(2.2, panel_width / aspect + 0.55)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3 * panel_width + 0.4, figure_height),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.045},
    )

    density_cmap = plt.get_cmap("twilight").copy()
    density_cmap.set_bad(density_cmap(0.0))
    density_image = axes[0].pcolormesh(
        x_edges,
        y_edges,
        density_column.T,
        cmap=density_cmap,
        norm=LogNorm(*top_decades(density_column)),
        shading="flat",
        rasterized=True,
    )
    compact_colorbar(
        fig,
        axes[0],
        density_image,
        "Column density",
        text_color="white",
    )

    visible_shocks = np.ma.masked_where(shock_count <= 0, shock_count)
    shock_cmap = plt.get_cmap("inferno").copy()
    shock_cmap.set_bad("white")
    shock_vmax = max(2.0, float(np.max(shock_count)))
    shock_image = axes[1].pcolormesh(
        shock_x_edges,
        shock_y_edges,
        visible_shocks.T,
        cmap=shock_cmap,
        norm=LogNorm(vmin=1, vmax=shock_vmax),
        shading="flat",
        rasterized=True,
    )
    compact_colorbar(
        fig,
        axes[1],
        shock_image,
        r"Shock surface cells per pixel",
    )

    dissipation_cmap = plt.get_cmap("viridis").copy()
    dissipation_cmap.set_bad(dissipation_cmap(0.0))
    dissipation_image = axes[2].pcolormesh(
        x_edges,
        y_edges,
        dissipation_column.T,
        cmap=dissipation_cmap,
        norm=LogNorm(*top_decades(dissipation_column)),
        shading="flat",
        rasterized=True,
    )
    compact_colorbar(
        fig,
        axes[2],
        dissipation_image,
        "Column dissipation",
        text_color="white",
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(x_edges[[0, -1]])
        ax.set_ylim(y_edges[[0, -1]])
        ax.tick_params(labelsize=10)
        ax.set_xlabel(rf"${axis_names[0]}/r_p$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    axes[0].set_ylabel(rf"${axis_names[1]}/r_p$", labelpad=2)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.985)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(fig)


def main(
    task_index: int | None = typer.Option(
        None,
        min=0,
        help="Zero-based index in sorted result files; use --list-only to inspect.",
    ),
    result: Path | None = typer.Option(
        None, help="Single shockfinder_snap_NNNN.npz, instead of --task-index."
    ),
    result_root: Path = typer.Option(
        RESULT_ROOT,
        help="Shock-finder input root; contains RUN/shockfinder_snap_NNNN.npz.",
    ),
    output_root: Path | None = typer.Option(
        None, help="Output root; defaults to RESULT_ROOT/analysis/shock-locations."
    ),
    per_cell_root: Path | None = typer.Option(
        None, help="Per-cell input root; defaults to RESULT_ROOT/analysis/per-cell."
    ),
    workers: int = typer.Option(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        min=1,
        help="Workers for richio projection queries",
    ),
    dpi: int = typer.Option(240, min=100, help="Output PNG resolution"),
    overwrite: bool = typer.Option(False, help="Rebuild histograms and figures"),
    rerender: bool = typer.Option(False, help="Redraw figures from an existing cache"),
    shock_mach_min: float = typer.Option(
        DEFAULT_SHOCK_MACH_MIN,
        min=1,
        help="Minimum thermal Mach number shown in the shock-location panel",
    ),
    shock_power_min: float = typer.Option(
        0, min=0, help="Optional minimum per-cell shock-finder power in erg/s"
    ),
    list_only: bool = typer.Option(False, help="Print the selected result and exit"),
) -> None:
    """Plot XY/YZ gas projections and filtered shock-surface cell counts."""
    global RESULT_ROOT, OUTPUT_ROOT, PER_CELL_ROOT
    result_root = result.parent.parent if result is not None else result_root
    RESULT_ROOT = result_root
    OUTPUT_ROOT = output_root or result_root / "analysis/shock-locations"
    PER_CELL_ROOT = per_cell_root or result_root / "analysis/per-cell"
    if result is not None and task_index is not None:
        raise typer.BadParameter("Choose --result or --task-index, not both.")
    results = [result] if result is not None else selected_results()
    if list_only and task_index is None:
        for index, path in enumerate(results):
            print(f"[{index:02d}] {path}")
        return
    if task_index is None:
        if len(results) != 1:
            raise typer.BadParameter(
                "Choose --result or --task-index; use --list-only to inspect results."
            )
        task_index = 0
    if task_index >= len(results):
        raise typer.BadParameter(
            f"--task-index must be between 0 and {len(results) - 1}"
        )
    result_path = results[task_index]
    with np.load(result_path) as result:
        run = str(result["run"].item())
        snapnum = int(result["snapnum"])
        time_tfb = float(result["time_tfb"])
    print(
        f"[{task_index:02d}] {run} snap {snapnum}, t/t_fb={time_tfb:.4f}: {result_path}"
    )
    if list_only:
        return

    cache = cache_path(run, snapnum)
    rebuilt = overwrite or not cache_complete(cache)
    if rebuilt:
        build_cache(result_path, cache, workers)
    else:
        print(f"Using cached histograms {cache}")

    for plane in ("xy", "yz"):
        destination = (
            OUTPUT_ROOT / run / plane / f"shock_locations_snap_{snapnum:04d}.png"
        )
        if (
            destination.is_file()
            and destination.stat().st_size > 0
            and not (overwrite or rerender or rebuilt)
        ):
            print(f"Exists {destination}")
            continue
        print(f"Rendering {destination}", flush=True)
        render_plane(
            cache,
            plane,
            destination,
            dpi,
            shock_mach_min,
            shock_power_min,
        )


if __name__ == "__main__":
    typer.run(main)
