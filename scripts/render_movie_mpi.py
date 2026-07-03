#!/usr/bin/env python
#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO and distributed under the EUPL v1.2 or later.

"""Frame-parallel rotating-camera movie of a RICH snapshot.

``yt``'s volume renderer is CPU/software, but movie frames are independent, so
this script splits the frames across MPI ranks: each rank builds the (shared)
uniform grid once, renders its slice of frames to PNGs in a common directory,
and rank 0 stitches them into a movie after a barrier.

Run serially::

    python render_movie_mpi.py SNAP.h5 --field density --nframes 180 \
        --res 256 --resolution 1024 --out spin.mp4

Run frame-parallel on a cluster (one grid build per rank, frames split N ways)::

    srun -n 48 python render_movie_mpi.py SNAP.h5 --nframes 360 \
        --res 384 --resolution 1536 --out spin.mp4 --frames-dir ./frames

Falls back to serial automatically when mpi4py is unavailable.
"""

import argparse
import os
import sys


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
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")

    import richio
    from richio.render import to_uniform_grid, volume_movie

    comm = _comm()
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1

    frames_dir = args.frames_dir or os.path.abspath("richio_vr_frames")
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
