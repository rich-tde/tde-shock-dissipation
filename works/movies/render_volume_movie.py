"""Render a rotating-camera volume movie of a RICH snapshot.

Depth-cued 3-D volume rendering via :mod:`richio.render` (yt backend). Frames
are rendered across local CPU workers (``--n-jobs``) — yt's renderer is
single-threaded, so this is the cheap way to use a multi-core node (e.g. a
``gpu_strw`` node with 48 cores). For multi-node, see ``render_movie_mpi.py``.

Examples
--------
Quick local preview (low res / few frames)::

    python render_volume_movie.py SNAP.h5 --field density \
        --res 96 --resolution 384 --nframes 36 --out preview.mp4

Higher-quality, parallel across 24 workers::

    python render_volume_movie.py \
        /data1/projects/pi-rossiem/TDE_data/...HiResNewAMR/snap_18/snap_18.h5 \
        --field density --res 256 --resolution 1024 --nframes 180 \
        --n-jobs 24 --out reports/gifs/density_spin.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("snapshot", help="Path to a RICH snapshot (.h5 or NPY dir).")
    p.add_argument("--field", default="density")
    p.add_argument("--res", type=int, default=256, help="Resampling grid resolution.")
    p.add_argument("--resolution", type=int, default=1024, help="Image side in px.")
    p.add_argument("--nframes", type=int, default=180)
    p.add_argument("--total-angle", type=float, default=360.0)
    p.add_argument("--elevation", type=float, default=20.0)
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--zoom", type=float, default=1.4)
    p.add_argument("--rot-axis", default="z", choices=["x", "y", "z"])
    p.add_argument(
        "--box",
        default="auto",
        choices=["auto", "full"],
        help="'auto' fits a tight box around the dense region.",
    )
    p.add_argument(
        "--mode",
        default="volume",
        choices=["volume", "projection"],
        help="'volume' (transfer function) or 'projection' (line integral, no occlusion).",
    )
    p.add_argument(
        "--weight",
        default=None,
        help="Projection weight field (default: column density).",
    )
    p.add_argument(
        "--tf-mode",
        default="map",
        choices=["map", "layers"],
        help="'map' continuous colormap (shows core); 'layers' shells.",
    )
    p.add_argument(
        "--alpha", type=float, default=20.0, help="Opacity scale (map mode)."
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=2.5,
        help="Opacity ramp exponent; lower=more haze.",
    )
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--cmap", default="magma")
    p.add_argument(
        "--colorbar", action="store_true", help="Annotate frames with a colorbar."
    )
    p.add_argument("--sigma-clip", type=float, default=4.0)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--no-log", action="store_true", help="Render in linear space.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=1, help="Local worker processes.")
    p.add_argument("--out", default="render.mp4")
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")

    from richio.render import to_uniform_grid, volume_movie

    import richio

    axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[
        args.rot_axis
    ]

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render_volume_movie] loading {args.snapshot}", flush=True)
    snap = richio.load(args.snapshot)

    box_size = "auto" if args.box == "auto" else None

    grid_started = time.time()
    grid = to_uniform_grid(snap, args.field, res=args.res, box_size=box_size)
    print(
        f"[render_volume_movie] grid {grid.dims} built in "
        f"{time.time() - grid_started:.1f}s",
        flush=True,
    )

    render_started = time.time()
    result = volume_movie(
        snap,
        field=args.field,
        grid=grid,
        n_frames=args.nframes,
        total_angle=args.total_angle,
        rot_axis=axis,
        azimuth=args.azimuth,
        elevation=args.elevation,
        zoom=args.zoom,
        log=not args.no_log,
        vmin=args.vmin,
        vmax=args.vmax,
        mode=args.mode,
        weight=args.weight,
        tf_mode=args.tf_mode,
        alpha=args.alpha,
        gamma=args.gamma,
        n_layers=args.n_layers,
        cmap=args.cmap,
        colorbar=args.colorbar,
        sigma_clip=args.sigma_clip,
        resolution=args.resolution,
        fps=args.fps,
        n_jobs=args.n_jobs,
        filename=str(output_path),
        frames_dir=args.frames_dir,
        keep_frames=args.keep_frames,
    )
    elapsed = time.time() - render_started
    print(
        f"[render_volume_movie] {args.nframes} frames in {elapsed:.1f}s "
        f"({elapsed / max(args.nframes, 1):.2f}s/frame) -> {result['filename']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
