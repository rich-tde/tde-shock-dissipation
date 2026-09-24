r"""Render wide-debris and close-up rotating-camera views of one TDE snapshot.

The snapshot is loaded once. A wide automatic box surrounds cells above the 98th
percentile of the selected field (about the top two per cent); the close-up is
a cube centred on the largest ``--field`` value.
That centre is a geometric choice, not evidence of circularization. Frames
are distributed over local workers. Volume opacity is a visual transfer
function; projection mode integrates the field along the camera ray.

Input files
-----------
``snapshot`` (positional argument)
    RICH HDF5 file or extracted NPY directory readable by ``richio.load``.
    Requires ``--field`` (default density) and stored cell coordinates. No
    reference-frame correction is applied. ``--disk-radius`` is a cube
    half-width in RICH code lengths (solar-radius scale; default 250).

Output files
------------
``<outdir>/tde_wide.mp4``, ``<outdir>/tde_disk.mp4``
    H.264 movies for the selected ``--which`` views. Projection mode inserts
    ``_proj`` before ``.mp4``. Default ``outdir`` is ``reports/gifs``.
    ``--nframes`` camera angles (default 120) depict one snapshot; ``--fps``
    sets playback rate (default 24). Decoded frames are ``uint8 (H, W, 3)``:
    image rows, columns and RGB channels in 0--255.
``<frames-root>/tde_<view>[_proj]/frame_<index:05d>.png``
    Persistent PNGs; default root is ``<outdir>/frames``. ``index`` is the
    zero-based camera-frame position. PIL RGBA conversion gives
    ``uint8 (H, W, 4)``, alpha at channel 3. The colourbar encodes physical
    cgs field values on a log scale (density: ``g/cm**3`` for volume rendering
    or ``g/cm**2`` for projection). ``--resolution`` controls render sampling;
    colourbars add width and encoding can round dimensions to even pixels.
    No numerical map, grid or metadata table is saved.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/movies/render_tde_movies.py /path/to/snap_21.h5 --which disk \
        --mode projection --disk-radius 250 --n-jobs 4 \
        --outdir data/processed/Movies/snap21-density

Existing movies are skipped unless ``--overwrite`` is given. An overwritten
or incomplete movie rerenders its frames. Use a separate output directory for
each snapshot/field/settings combination to retain distinct products.

Loading examples
----------------
Inspect the close-up projection created above::

    import imageio.v2 as imageio
    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/Movies/snap21-density")
    with imageio.get_reader(root / "tde_disk_proj.mp4") as movie:
        rgb = movie.get_data(0)  # (H, W, 3), uint8
        print(movie.get_meta_data(), rgb.shape)
    with Image.open(root / "frames/tde_disk_proj/frame_00000.png") as image:
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

import numpy as np


def _densest_cell_box(snap, field, radius, coords=("X", "Y", "Z")):
    """Cubic box of half-width *radius* centred on the densest cell."""
    import unyt as u

    field_values = np.asarray(snap._get_data(field))
    densest_index = int(np.nanargmax(field_values))
    x_coordinates = snap._get_data(coords[0])
    center = np.array(
        [
            float(np.asarray(x_coordinates)[densest_index]),
            float(np.asarray(snap._get_data(coords[1]))[densest_index]),
            float(np.asarray(snap._get_data(coords[2]))[densest_index]),
        ]
    )
    return u.unyt_array(
        np.concatenate([center - radius, center + radius]), x_coordinates.units
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("snapshot")
    p.add_argument("--field", default="density")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--nframes", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument(
        "--disk-radius", type=float, default=250.0, help="Close-up half-width (R_sun)."
    )
    p.add_argument("--outdir", default="reports/gifs")
    p.add_argument(
        "--frames-root", type=Path, help="PNG storage (default: <outdir>/frames)."
    )
    p.add_argument("--which", default="both", choices=["both", "wide", "disk"])
    p.add_argument(
        "--mode",
        default="volume",
        choices=["volume", "projection"],
        help="'volume' (transfer function) or 'projection' (line integral).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing movie and rerender its frames.",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from richio.render import to_uniform_grid, volume_movie

    import richio

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = args.frames_root or output_dir / "frames"
    suffix = "_proj" if args.mode == "projection" else ""
    selected_views = ("wide", "disk") if args.which == "both" else (args.which,)
    if not args.overwrite and all(
        (output_dir / f"tde_{tag}{suffix}.mp4").exists() for tag in selected_views
    ):
        print(f"Skipping existing movies in {output_dir}; use --overwrite to replace.")
        return 0
    print(f"loading {args.snapshot}", flush=True)
    snap = richio.load(args.snapshot)

    common = {
        "field": args.field,
        "n_frames": args.nframes,
        "resolution": args.resolution,
        "cmap": "magma",
        "alpha": 20.0,
        "tf_mode": "map",
        "colorbar": True,
        "sigma_clip": 4.0,
        "fps": args.fps,
        "n_jobs": args.n_jobs,
        "mode": args.mode,
    }

    def render(tag, box_size, zoom, elevation, gamma):
        movie = output_dir / f"tde_{tag}{suffix}.mp4"
        if movie.exists() and not args.overwrite:
            print(f"Skipping existing {movie}; use --overwrite to replace.", flush=True)
            return
        grid_started = time.time()
        grid = to_uniform_grid(snap, args.field, res=args.res, box_size=box_size)
        print(
            f"[{tag}] grid {grid.dims} in {time.time() - grid_started:.1f}s",
            flush=True,
        )
        render_started = time.time()
        result = volume_movie(
            snap,
            grid=grid,
            zoom=zoom,
            elevation=elevation,
            gamma=gamma,
            filename=str(movie),
            frames_dir=str(frames_root / f"tde_{tag}{suffix}"),
            **common,
        )
        elapsed = time.time() - render_started
        print(
            f"[{tag}] {args.nframes} frames in {elapsed:.1f}s "
            f"({elapsed / args.nframes:.2f}s/frame) -> {result['filename']}",
            flush=True,
        )

    if args.which in ("both", "wide"):
        # Wide: a touch less gas (gamma 2.5) so the disk knot isn't obscured.
        render("wide", "auto", zoom=1.1, elevation=26, gamma=2.5)

    if args.which in ("both", "disk"):
        # Disk close-up: keep more gas as context (gamma 2.2).
        box = _densest_cell_box(snap, args.field, args.disk_radius)
        render("disk", box, zoom=1.15, elevation=35, gamma=2.2)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
