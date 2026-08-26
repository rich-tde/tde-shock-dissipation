#!/usr/bin/env python3
"""Render fixed-r_amin, face-on projection movies for the three TDE runs.

The image plane is sampled densely while the line of sight uses a cheaper sinh
grid concentrated around z=0.  All movies use one fixed six-decade colour range
and richio's established presentation layer.  Frames persist for safe restarts.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rich-tde-movies")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev
import numpy as np
import richio
import richio.render as richrender
import unyt as u
from dev.datapaths import DATAPATHS, SNAPSHOT_TIMES, TDE_PARAMETERS


REPO = Path("/home/hey4/rich_tde")
OUTPUT_ROOT = REPO / "reports/movies/crete"
COLOR_VMAX = 10.0**6.5  # rounded above the legacy faceon_A scan maximum (2.22e6)
AMBIENT_FACTOR = 1e-8
AMBIENT_UNTIL_TFB = 0.3
INITIAL_BOX_VOLUME = (10 * richio.units.lscale) ** 3
WINDOWS = {
    "proposed": (-1.5, 0.5, -0.7, 0.7, -0.7, 0.7),
    "taller": (-1.5, 0.5, -0.9, 0.9, -0.7, 0.7),
    "wider": (-2.0, 0.5, -0.9, 0.9, -0.7, 0.7),
}
RUN_WINDOW_OVERRIDES = {
    "1e4": {"proposed": (-2.0, 0.5, -0.875, 0.875, -0.7, 0.7)},
}


@dataclass(frozen=True)
class RunConfig:
    mode: int
    run: str
    m_bh: float
    m_star: float
    r_star: float
    fps: int

    @property
    def r_amin(self):
        return (
            self.r_star * (self.m_bh / self.m_star) ** (2.0 / 3.0)
            * richio.units.lscale
        )

    @property
    def t_fb(self):
        return (
            math.pi
            / math.sqrt(2.0)
            * math.sqrt(self.r_star**3 / self.m_star)
            * math.sqrt(self.m_bh / self.m_star)
            * richio.units.tscale
        )


@dataclass(frozen=True)
class Quality:
    name: str
    nx: int
    ny: int
    nz: int
    dpi: int
    canvas: tuple[int, int]


@dataclass(frozen=True)
class FieldConfig:
    name: str
    unit: str
    label: str
    cmap: str
    vmax: float
    decades: int
    suppress_ambient: bool = False
    floor_nonpositive: bool = False


RUNS = {
    1: RunConfig(1, "1e4", *TDE_PARAMETERS["1e4"], fps=8),
    2: RunConfig(2, "1e5", *TDE_PARAMETERS["1e5"], fps=16),
    3: RunConfig(3, "1e6", *TDE_PARAMETERS["1e6"], fps=24),
}
QUALITIES = {
    "preview": Quality("preview", 256, 180, 128, 128, (1220, 720)),
    "production": Quality("production", 2048, 1434, 256, 300, (2440, 1434)),
}
FIELDS = {
    "density": FieldConfig(
        "density", "g/cm**2", r"Column density $[\mathrm{g/cm^2}]$",
        "twilight", COLOR_VMAX, 6, suppress_ambient=True,
    ),
    "dissipation": FieldConfig(
        "dissipation", "erg/s/cm**2",
        r"Column dissipation $[\mathrm{erg/s/cm^2}]$", "viridis", 1e18, 4,
        floor_nonpositive=True,
    ),
}


def mode_settings(mode: int) -> RunConfig:
    try:
        return RUNS[mode]
    except KeyError as exc:
        raise ValueError("mode must be 1 (1e4), 2 (1e5), or 3 (1e6)") from exc


def needs_reference_frame(run: str, path: Path) -> bool:
    """Match the dataset-specific switch used by works/shock-tde/E-t.py."""
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def window_bounds(config: RunConfig, window: str = "proposed") -> tuple:
    return RUN_WINDOW_OVERRIDES.get(config.run, {}).get(window, WINDOWS[window])


def fixed_box(config: RunConfig, window: str = "proposed") -> tuple:
    # WINDOWS is stored as (xmin, xmax, ymin, ymax, zmin, zmax) for plotting;
    # richio expects (xmin, ymin, zmin, xmax, ymax, zmax).
    xmin, xmax, ymin, ymax, zmin, zmax = window_bounds(config, window)
    return tuple(
        value * config.r_amin
        for value in (xmin, ymin, zmin, xmax, ymax, zmax)
    )


def bh_axes_position(config: RunConfig, window: str = "proposed") -> tuple[float, float]:
    xmin, xmax, ymin, ymax, _, _ = window_bounds(config, window)
    return -xmin / (xmax - xmin), -ymin / (ymax - ymin)


def grid_samples(quality: Quality) -> tuple[int, int, int]:
    """Samples required for exactly nx*ny projected pixels and nz z intervals."""
    return quality.nx + 1, quality.ny + 1, quality.nz + 1


def color_limits(vmax: float, decades: int) -> tuple[float, float]:
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be positive and finite")
    if decades <= 0:
        raise ValueError("decades must be positive")
    return vmax / 10**decades, vmax


def frame_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _time_scalar(snapshot):
    return snapshot.t.reshape(-1)[0] if getattr(snapshot.t, "ndim", 0) else snapshot.t


def _time_days(time) -> float:
    """Convert a snapshot's registered code-time quantity to physical days."""
    return float(time.to_value("day"))


def time_annotation(time, config: RunConfig) -> str:
    """Presentation timestamp; mass labels are added separately in post."""
    return (
        f"$t = {float(time / config.t_fb):.2f}\\,t_{{\\rm fb}}$\n"
        f"$t = {_time_days(time):.2f}\\,\\mathrm{{d}}$"
    )


def projection_density(snapshot, time, config: RunConfig,
                       ambient_factor: float = AMBIENT_FACTOR):
    """Suppress the ambient tracer before 0.3 fallback times without mutating data."""
    if not 0 < ambient_factor <= 1:
        raise ValueError("ambient_factor must be in (0, 1]")
    density = snapshot.density
    if time / config.t_fb >= AMBIENT_UNTIL_TFB:
        return density
    lengths = snapshot.box[3:] - snapshot.box[:3]
    volume_ratio = (np.prod(lengths) / INITIAL_BOX_VOLUME).to_value("dimensionless")
    ambient_factor = min(1.0, ambient_factor * volume_ratio)
    density = density.copy()
    density[np.asarray(snapshot["tracers/Star"]) < 0.99] *= ambient_factor
    return density


def frame_durations(times, fps: int):
    """Hold each snapshot in proportion to the following physical-time gap."""
    intervals = np.diff(times)
    if len(intervals) == 0 or np.any(intervals <= 0 * times.units):
        raise ValueError("snapshot times must be strictly increasing")
    intervals = np.concatenate((intervals, np.atleast_1d(np.median(intervals))))
    target = (len(times) - 1) / fps * u.s
    return (intervals / intervals.sum() * target).in_units("s")


def project_snapshot(path: Path, config: RunConfig, quality: Quality, window: str,
                     workers: int, field: FieldConfig,
                     ambient_factor: float = AMBIENT_FACTOR):
    snap = richio.load(str(path))
    time = _time_scalar(snap)
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

    data = getattr(snap, field.name)
    if field.suppress_ambient:
        data = projection_density(snap, time, config, ambient_factor)
    elif field.floor_nonpositive:
        data = data.copy()
        data[(data < 0) | ~np.isfinite(data)] = 0

    projected, _, _ = snap.project(
        data,
        res=grid_samples(quality),
        X=x,
        Y=y,
        Z=z,
        box_size=fixed_box(config, window),
        unit_system="cgs",
        workers=workers,
        spacing=("linear", "linear", "sinh"),
        sinh_scale=(None, None, 0.1 * richio.units.lscale),
    )
    return projected, time


def render_projection(projected, time, config: RunConfig, quality: Quality,
                      field: FieldConfig, output: Path, window: str,
                      vmax: float) -> None:
    vmin, vmax = color_limits(vmax, field.decades)
    if field.floor_nonpositive:
        projected = projected.copy()
        floor = u.unyt_quantity(vmin, field.unit).in_units(projected.units)
        projected[(projected <= 0) | ~np.isfinite(projected)] = floor
    annotation = time_annotation(time, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    richrender.projection_image(
        projected, output, field=field.name, unit=field.unit,
        label=field.label, cmap=field.cmap,
        norm="log", vmin=vmin, vmax=vmax, annotate=annotation,
        azimuth=0.0, elevation=-90.0, axis_triad=True, flip_x=False,
        scalebar_frac=0.25, scalebar_label=r"$0.5\,r_{\rm amin}$",
        output_size=quality.canvas, dpi=quality.dpi,
        annotation_color="black", scalebar_color="black",
        points=[bh_axes_position(config, window)], point_color="black", point_size=12,
    )


_WORKER_STATE: dict = {}


def _render_task(task) -> int:
    index, path = task
    state = _WORKER_STATE
    destination = state["frames_dir"] / f"frame_{index:05d}.png"
    if frame_complete(destination) and not state["overwrite"]:
        return index
    projected, time = project_snapshot(
        path, state["config"], state["quality"], state["window"], state["workers"],
        state["field"], state["ambient_factor"],
    )
    render_projection(
        projected,
        time,
        state["config"],
        state["quality"],
        state["field"],
        destination,
        state["window"],
        state["vmax"],
    )
    return index


def _contact_sheet(paths: list[Path], titles: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.1), dpi=180)
    axes = np.atleast_1d(axes)
    for ax, path, title in zip(axes, paths, titles):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_preview_examples(config: RunConfig, quality: Quality, field: FieldConfig,
                          snapnums, paths, frames_dir: Path, output_dir: Path,
                          workers: int, overwrite: bool, vmax: float,
                          ambient_factor: float) -> None:
    indices = [0, len(paths) // 2, len(paths) - 1]
    _contact_sheet(
        [frames_dir / f"frame_{index:05d}.png" for index in indices],
        [f"early: snap {snapnums[indices[0]]}", f"middle: snap {snapnums[indices[1]]}",
         f"late: snap {snapnums[indices[2]]}"],
        output_dir / "timeline.png",
    )

    middle = indices[1]
    comparison_paths = []
    for window in WINDOWS:
        destination = output_dir / "window-examples" / f"{window}.png"
        comparison_paths.append(destination)
        if frame_complete(destination) and not overwrite:
            continue
        projected, time = project_snapshot(
            paths[middle], config, quality, window, workers, field, ambient_factor
        )
        render_projection(
            projected, time, config, quality, field, destination, window, vmax
        )
    _contact_sheet(
        comparison_paths,
        ["proposed", "taller", "wider"],
        output_dir / "window-comparison.png",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, required=True, choices=RUNS)
    parser.add_argument("--field", choices=FIELDS, default="density")
    parser.add_argument("--quality", choices=QUALITIES, default="preview")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=-1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--ambient-factor", type=float, default=AMBIENT_FACTOR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = mode_settings(args.mode)
    quality = QUALITIES[args.quality]
    field = FIELDS[args.field]
    vmax = field.vmax if args.vmax is None else args.vmax
    snapnums, paths = DATAPATHS(config.run)
    paths = [Path(path) for path in paths]
    output_dir = args.output_root / quality.name / config.run
    field_dir = output_dir if field.name == "density" else output_dir / field.name
    frames_dir = field_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    start = max(0, args.frame_start)
    stop = len(paths) if args.frame_stop < 0 else min(args.frame_stop, len(paths))
    tasks = [(index, paths[index]) for index in range(start, stop)]
    print(
        f"[{config.run}] {len(paths)} total snapshots; rendering [{start}, {stop}) "
        f"at {quality.nx}x{quality.ny}x{quality.nz}; fps={config.fps}; "
        f"field={field.name}; r_amin={config.r_amin:.6g}",
        flush=True,
    )

    if not args.encode_only:
        _WORKER_STATE.update(
            config=config,
            quality=quality,
            field=field,
            window="proposed",
            workers=args.workers,
            frames_dir=frames_dir,
            overwrite=args.overwrite,
            vmax=vmax,
            ambient_factor=args.ambient_factor,
        )
        jobs = min(args.n_jobs, max(1, len(tasks)))
        try:
            if jobs > 1 and tasks:
                with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
                    for done, index in enumerate(pool.imap_unordered(_render_task, tasks), 1):
                        print(f"[{config.run}] frame {index} complete ({done}/{len(tasks)})", flush=True)
            else:
                for done, task in enumerate(tasks, 1):
                    index = _render_task(task)
                    print(f"[{config.run}] frame {index} complete ({done}/{len(tasks)})", flush=True)
        finally:
            _WORKER_STATE.clear()

    complete_run = start == 0 and stop == len(paths)
    if quality.name == "preview" and complete_run and not args.no_examples:
        make_preview_examples(
            config, quality, field, snapnums, paths, frames_dir, field_dir,
            args.workers, args.overwrite, vmax, args.ambient_factor
        )
    if not args.no_encode and complete_run:
        movie = field_dir / f"faceon_{field.name}_{config.run}.mp4"
        durations = frame_durations(SNAPSHOT_TIMES(config.run), config.fps)
        richrender.encode_movie(
            frames_dir, len(paths), movie, config.fps, durations=durations
        )
        print(f"[{config.run}] encoded {movie}", flush=True)
    elif not args.no_encode:
        print(f"[{config.run}] partial frame window: skipping encode", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
