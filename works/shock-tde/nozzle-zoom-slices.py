#!/usr/bin/env python3
"""Plot four-panel nozzle-region midplane slices at selected fallback times.

The snapshot selection exactly follows ``0.1-plot-Ediss-distribution.ipynb``.
Each run uses common colour limits across its selected snapshots.  Interpolated
logarithmic grids are cached so interrupted Slurm jobs can resume cheaply.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rich-tde-nozzle-slices")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev
import matplotlib.pyplot as plt
import numpy as np
import typer

import richio
from dev import DATAPATHS, SNAPSHOT_TFB
from dev.datapaths import TDE_PARAMETERS
from richio.plots import scalar_map


REPO = Path("/home/hey4/rich_tde")
OUTPUT_ROOT = REPO / "data/processed/NozzleZoomSlices"
WINDOW = (-1.0, 2.0, -1.5, 1.5)
REQUESTED_TFBS = {
    "1e4": (0.5, 1.0, 1.5, 2.0),
    "1e5": (0.3, 0.5),
    "1e6": (1.0, 1.2, 1.4, 1.5),
}


@dataclass(frozen=True)
class RunConfig:
    mode: int
    run: str
    m_bh: float
    m_star: float
    r_star: float

    @property
    def r_p(self) -> float:
        return self.r_star * (self.m_bh / self.m_star) ** (1.0 / 3.0)

    @property
    def t_fb(self) -> float:
        return (
            math.pi
            / math.sqrt(2.0)
            * math.sqrt(self.r_star**3 / self.m_star)
            * math.sqrt(self.m_bh / self.m_star)
        )


RUNS = {
    mode: RunConfig(mode, run, *TDE_PARAMETERS[run])
    for mode, run in enumerate(("1e4", "1e5", "1e6"), start=1)
}

FIELDS = (
    ("density", "density", "Density", r"$\log_{10}(\rho/[\mathrm{g\,cm^{-3}}])$", "twilight"),
    ("pressure", "pressure", "Gas pressure", r"$\log_{10}(P/[\mathrm{dyn\,cm^{-2}}])$", "rainbow"),
    ("temperature", "temperature", "Gas temperature", r"$\log_{10}(T_\mathrm{gas}/\mathrm{K})$", "inferno"),
    (
        "dissipation",
        "dissipation",
        "Dissipation",
        r"$\log_{10}(\dot{e}_\mathrm{diss}/[\mathrm{erg\,s^{-1}\,cm^{-3}}])$",
        "viridis",
    ),
)


def mode_settings(mode: int) -> RunConfig:
    try:
        return RUNS[mode]
    except KeyError as exc:
        raise ValueError("mode must be 1 (1e4), 2 (1e5), or 3 (1e6)") from exc


def selected_snapshots(run: str) -> list[tuple[int, Path, bool]]:
    selected = [
        (*SNAPSHOT_TFB(run, requested_tfb), False)
        for requested_tfb in REQUESTED_TFBS[run]
    ]
    snapnums, paths = DATAPATHS(run)
    selected.append((snapnums[-1], paths[-1], True))
    return [(snapnum, Path(path), is_last) for snapnum, path, is_last in selected]


def needs_reference_frame(run: str, path: Path) -> bool:
    """Match the dataset-specific coordinate switch used by ``E-t.py``."""
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def scalar_time(snapshot):
    return snapshot.t.reshape(-1)[0] if getattr(snapshot.t, "ndim", 0) else snapshot.t


def positive_log10(values) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def cache_complete(path: Path, resolution: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return all(
                name in data and data[name].shape == (resolution, resolution)
                for name, *_ in FIELDS
            )
    except (OSError, ValueError):
        return False


def cache_snapshot(
    path: Path,
    output: Path,
    config: RunConfig,
    resolution: int,
    workers: int,
) -> None:
    snap = richio.load(str(path))
    time = scalar_time(snap)
    x, y, z = snap.X, snap.Y, snap.Z
    if needs_reference_frame(config.run, path):
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
    xmin, xmax, ymin, ymax = WINDOW
    indices, xspace, yspace = snap.to_2dgrid(
        res=(resolution, resolution),
        X=x,
        Y=y,
        Z=z,
        plane="xy",
        slice_coord=0 * richio.units.lscale,
        box_size=(xmin * r_p, ymin * r_p, xmax * r_p, ymax * r_p),
        volume_selection=True,
        workers=workers,
    )

    arrays = {
        "x_rp": np.asarray(xspace / r_p, dtype="float64"),
        "y_rp": np.asarray(yspace / r_p, dtype="float64"),
        "time_tfb": np.asarray(float(time / config.t_fb)),
    }
    for output_name, attribute, *_ in FIELDS:
        field = getattr(snap, attribute)[indices].in_cgs()
        arrays[output_name] = positive_log10(field)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)


def common_limits(cache_paths: list[Path]) -> dict[str, tuple[float, float]]:
    limits = {}
    for name, *_ in FIELDS:
        finite_parts = []
        for path in cache_paths:
            with np.load(path) as data:
                values = data[name]
                finite_parts.append(values[np.isfinite(values)])
        finite = np.concatenate(finite_parts)
        if finite.size == 0:
            raise ValueError(f"all cached {name} values are non-positive or non-finite")
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        if data_max - data_min < 3.0:
            limits[name] = (data_min, data_max)
        else:
            vmax = math.ceil(2.0 * data_max) / 2.0
            vmin = max(math.floor(2.0 * data_min) / 2.0, vmax - 6.0)
            limits[name] = (vmin, vmax)
    return limits


def render_figure(
    cache_path: Path,
    destination: Path,
    snapnum: int,
    is_last: bool,
    config: RunConfig,
    limits: dict[str, tuple[float, float]],
    dpi: int,
) -> None:
    with np.load(cache_path) as data:
        x_rp = data["x_rp"]
        y_rp = data["y_rp"]
        time_tfb = float(data["time_tfb"])
        grids = {name: np.array(data[name]) for name, *_ in FIELDS}

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.0), sharex=True, sharey=True)
    for ax, (name, _, title, colorbar_label, cmap_name) in zip(axes.flat, FIELDS):
        scalar_map(
            grids[name],
            x_rp,
            y_rp,
            ax=ax,
            cmap=cmap_name,
            colorbar_label=colorbar_label,
            log_scale=False,
            vmin=limits[name][0],
            vmax=limits[name][1],
            shading="auto",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(WINDOW[:2])
        ax.set_ylim(WINDOW[2:])

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x/r_p$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y/r_p$")

    last_label = ", last snapshot" if is_last else ""
    exponent = int(round(math.log10(config.m_bh)))
    fig.suptitle(
        rf"$M_{{\rm BH}}=10^{{{exponent}}}\,M_\odot$, snap {snapnum}, "
        rf"$t/t_{{\rm fb}}={time_tfb:.3f}${last_label}"
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def main(
    mode: int = typer.Option(..., help="1: 1e4, 2: 1e5, 3: 1e6 solar-mass BH"),
    resolution: int = typer.Option(768, min=16, help="Pixels along each slice axis"),
    workers: int = typer.Option(8, min=1, help="KD-tree query threads"),
    dpi: int = typer.Option(240, min=50, help="Output PNG resolution"),
    output_root: Path = typer.Option(OUTPUT_ROOT, help="Study output directory"),
    overwrite: bool = typer.Option(False, help="Recompute cached grids and figures"),
    rerender: bool = typer.Option(False, help="Redraw figures from existing cached grids"),
    list_only: bool = typer.Option(False, help="Print selected snapshots without loading them"),
) -> None:
    config = mode_settings(mode)
    selected = selected_snapshots(config.run)
    output_dir = output_root / config.run
    cache_dir = output_dir / "grids"

    for snapnum, path, is_last in selected:
        print(f"[{config.run}] snap {snapnum}: {path}{' (last)' if is_last else ''}")
    if list_only:
        return

    cache_paths = []
    for snapnum, path, _ in selected:
        cache_path = cache_dir / f"nozzle_zoom_snap_{snapnum:04d}_{resolution}.npz"
        cache_paths.append(cache_path)
        if overwrite or not cache_complete(cache_path, resolution):
            print(f"[{config.run}] gridding snap {snapnum}", flush=True)
            cache_snapshot(path, cache_path, config, resolution, workers)
        else:
            print(f"[{config.run}] cached snap {snapnum}", flush=True)

    limits = common_limits(cache_paths)
    for (snapnum, _, is_last), cache_path in zip(selected, cache_paths):
        destination = output_dir / f"nozzle_zoom_snap_{snapnum:04d}.png"
        if (
            destination.is_file()
            and destination.stat().st_size > 0
            and not overwrite
            and not rerender
        ):
            print(f"[{config.run}] exists {destination.name}", flush=True)
            continue
        print(f"[{config.run}] rendering snap {snapnum}", flush=True)
        render_figure(cache_path, destination, snapnum, is_last, config, limits, dpi)


if __name__ == "__main__":
    typer.run(main)
