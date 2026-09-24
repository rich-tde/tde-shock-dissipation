#!/usr/bin/env python
#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO and distributed under the EUPL v1.2 or later.

r"""Render a rotating-camera volume movie of one snapshot across MPI ranks.

Rank zero builds and broadcasts a nearest-neighbour uniform grid. Ranks render
independent camera angles into a shared directory; rank zero encodes the movie
after a barrier. Every rank opens the snapshot. A volume transfer function
visualizes the selected field; its opacity is not physical radiative transfer.
The driver falls back to serial when MPI is unavailable.

Input files
-----------
``snapshot`` (positional argument)
    RICH HDF5 file or extracted NPY directory readable by ``richio.load``.
    Requires ``--field`` (default density) and stored cell coordinates.
    No BH-frame/orbital correction is applied. MPI execution requires
    ``mpi4py`` and input/frame paths accessible to every rank.

Output files
------------
``--out`` (default ``render.mp4`` in the working directory)
    H.264 movie of ``--nframes`` camera positions (default 180) at ``--fps``
    (default 30). All frames depict the same simulation time. ``imageio``
    decodes each frame as ``uint8 (H, W, 3)``: row, column, RGB channels
    (0--255). Image dimensions derive from ``--resolution`` (default 1024);
    video encoding may round them to even pixels.
``<frames-dir>/frame_<index:05d>.png``
    Zero-based camera-frame index. Default directory is ``<out-stem>_frames``
    beside the movie. PIL conversion to RGBA yields ``uint8 (H, W, 4)`` with
    alpha in channel 3. Channels are rendered colours, not physical density
    or dissipation. Fields are converted to cgs before rendering (density in
    ``g/cm**3``); logarithmic transfer-function bounds are specified as linear
    values. No numerical field/grid output is saved.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/movies/render_movie_mpi.py /path/to/snap_21.h5 \
        --res 96 --resolution 384 --nframes 36 --keep-frames \
        --out data/processed/Movies/spin.mp4
    srun -n 48 python works/movies/render_movie_mpi.py /path/to/snap_21.h5 \
        --nframes 360 --res 384 --resolution 1536 --keep-frames \
        --out data/processed/Movies/spin-mpi.mp4

Existing movies are skipped unless ``--overwrite`` is supplied. Otherwise all
assigned frames are rendered again; there is no per-frame resume. PNGs are
removed after successful encoding unless ``--keep-frames`` is set. Use separate
output paths for different scientific cases.

Loading examples
----------------
Inspect one movie frame and the retained source PNG::

    import imageio.v2 as imageio
    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/Movies")
    with imageio.get_reader(root / "spin.mp4") as movie:
        rgb = movie.get_data(0)  # (H, W, 3), uint8
        print(movie.get_meta_data(), rgb.shape)
    with Image.open(root / "spin_frames/frame_00000.png") as image:
        rgba = np.array(image.convert("RGBA"))  # (H, W, 4), uint8
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")


def _comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", help="Path to a RICH snapshot (.h5 or NPY dir).")
    p.add_argument("--field", default="density")
    p.add_argument("--res", type=int, default=256, help="Resampling grid resolution.")
    p.add_argument("--resolution", type=int, default=1024, help="Image side in px.")
    p.add_argument("--nframes", type=int, default=180)
    p.add_argument("--total-angle", type=float, default=360.0)
    p.add_argument("--elevation", type=float, default=20.0)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--sigma-clip", type=float, default=4.0)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="render.mp4")
    p.add_argument("--frames-dir", default=None, help="Shared dir for PNG frames.")
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

    comm = _comm()
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.frames_dir or str(
        output_path.with_name(output_path.stem + "_frames")
    )
    if rank == 0:
        os.makedirs(frames_dir, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    snap = richio.load(args.snapshot)

    # Build the uniform grid once on rank 0 (one file read, one k-d tree) and
    # broadcast it — the grid is just picklable numpy arrays. Avoids every rank
    # repeating the expensive resampling.
    if rank == 0:
        grid = to_uniform_grid(snap, args.field, res=args.res)
    else:
        grid = None
    if comm is not None:
        grid = comm.bcast(grid, root=0)

    my_frames = list(range(rank, args.nframes, size))
    if rank == 0:
        print(
            f"[render_movie_mpi] {size} rank(s), {args.nframes} frames, "
            f"grid={grid.dims}, image={args.resolution}px",
            flush=True,
        )

    common = dict(
        field=args.field,
        grid=grid,
        n_frames=args.nframes,
        total_angle=args.total_angle,
        elevation=args.elevation,
        zoom=args.zoom,
        n_layers=args.n_layers,
        cmap=args.cmap,
        sigma_clip=args.sigma_clip,
        vmin=args.vmin,
        vmax=args.vmax,
        resolution=args.resolution,
        frames_dir=frames_dir,
    )

    # Workers render only; nobody encodes yet.
    volume_movie(
        snap,
        frame_indices=my_frames,
        encode=False,
        verbose=(rank == 0),
        **common,
    )

    if comm is not None:
        comm.Barrier()

    # Rank 0 stitches the complete frame set into the movie.
    if rank == 0:
        from richio.render.yt_backend import _cleanup_frames, _encode_movie

        _encode_movie(frames_dir, args.nframes, args.out, args.fps)
        if not args.keep_frames:
            _cleanup_frames(frames_dir)
        print(f"[render_movie_mpi] done -> {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
