r"""Render a rotating-camera movie of one RICH snapshot using local workers.

Nearest-neighbour resampling builds one uniform grid, then ``richio.render``
uses a rotating camera to make volume-rendered or projected frames. Volume
opacity is a visual transfer function, not radiative transfer; an unweighted
projection is a line integral and a weighted projection is a mean. Workers
parallelize independent frames. Use ``render_movie_mpi.py`` for multiple nodes.

Input files
-----------
``snapshot`` (positional argument)
    RICH HDF5 file or extracted NPY directory readable by ``richio.load``.
    Requires ``--field`` (default density) and stored coordinates. No BH-frame
    correction is applied. ``--box auto`` fits the dense region; ``--box full``
    uses the snapshot domain. ``--res`` sets the 3-D interpolation resolution.

Output files
------------
``--out`` (default ``render.mp4`` in the working directory)
    H.264 movie of ``--nframes`` camera angles at ``--fps`` (default 30).
    Frames depict the same simulation time. ``imageio`` decodes each as
    ``uint8 (H, W, 3)``: row, column and RGB channels with values 0--255.
``<frames-dir>/frame_<index:05d>.png``
    Persistent PNGs; the default directory is ``<out-stem>_frames`` beside the
    movie. ``index`` is a zero-based camera position. PIL conversion to RGBA
    gives ``uint8 (H, W, 4)``, channel 3 alpha. ``--resolution`` sets render
    sampling; optional colourbars add width, and encoding can round dimensions
    to even pixels. Pixel values are colours, not field values; cgs physical
    units appear on the optional ``--colorbar`` (density volume:
    ``g/cm**3``; density projection: ``g/cm**2``). Bounds are linear physical
    values even for logarithmic display. No numerical map or cube is saved.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/movies/render_volume_movie.py /path/to/snap_21.h5 \
        --field density --res 96 --resolution 384 --nframes 36 --colorbar \
        --out data/processed/Movies/preview.mp4
    python works/movies/render_volume_movie.py /path/to/snap_21.h5 \
        --field density --res 256 --resolution 1024 --nframes 180 \
        --n-jobs 24 --colorbar --out data/processed/Movies/density_spin.mp4

Existing movies are skipped unless ``--overwrite`` is supplied. Interrupted
or overwritten runs render all requested frames again. PNGs persist after
encoding; ``--keep-frames`` remains accepted for compatibility. Use a distinct
output path for each scientific case/settings combination.

Loading examples
----------------
Inspect one movie frame and the corresponding PNG::

    import imageio.v2 as imageio
    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/Movies")
    with imageio.get_reader(root / "preview.mp4") as movie:
        rgb = movie.get_data(0)  # (H, W, 3), uint8
        print(movie.get_meta_data(), rgb.shape)
    with Image.open(root / "preview_frames/frame_00000.png") as image:
        rgba = np.array(image.convert("RGBA"))  # (H, W, 4), uint8
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")


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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing movie and rerender its frames.",
    )
    args = p.parse_args(argv)

    if Path(args.out).exists() and not args.overwrite:
        print(f"Skipping existing {args.out}; use --overwrite to replace.", flush=True)
        return 0

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
        frames_dir=args.frames_dir
        or str(output_path.with_name(output_path.stem + "_frames")),
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
