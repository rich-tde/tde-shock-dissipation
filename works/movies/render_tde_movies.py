#!/usr/bin/env python3
"""Render a wide whole-debris movie AND a disk close-up movie of a RICH TDE snapshot.

Loads the snapshot once, then renders two rotating-camera movies with
:mod:`richio.render`:

* **wide**  — cubic auto box around the densest 98% (whole stream + debris);
* **disk**  — small cube centred on the densest cell (the circularised disk).

Frames are split across local CPU workers (``--n-jobs``).
"""

import argparse
import os
import sys
import time

import numpy as np


def _densest_cell_box(snap, field, radius, coords=("X", "Y", "Z")):
    """Cubic box of half-width *radius* centred on the densest cell."""
    import unyt as u

    d = np.asarray(snap._get_data(field))
    ic = int(np.nanargmax(d))
    cx = snap._get_data(coords[0])
    c = np.array([
        float(np.asarray(cx)[ic]),
        float(np.asarray(snap._get_data(coords[1]))[ic]),
        float(np.asarray(snap._get_data(coords[2]))[ic]),
    ])
    return u.unyt_array(np.concatenate([c - radius, c + radius]), cx.units)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot")
    p.add_argument("--field", default="density")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--nframes", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--disk-radius", type=float, default=250.0, help="Close-up half-width (R_sun).")
    p.add_argument("--outdir", default="reports/gifs")
    p.add_argument("--which", default="both", choices=["both", "wide", "disk"])
    p.add_argument("--mode", default="volume", choices=["volume", "projection"],
                   help="'volume' (transfer function) or 'projection' (line integral).")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import richio
    from richio.render import to_uniform_grid, volume_movie

    os.makedirs(args.outdir, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    print(f"loading {args.snapshot}", flush=True)
    snap = richio.load(args.snapshot)

    common = dict(
        field=args.field, n_frames=args.nframes, resolution=args.resolution,
        cmap="magma", alpha=20.0, tf_mode="map", colorbar=True,
        sigma_clip=4.0, fps=args.fps, n_jobs=args.n_jobs, mode=args.mode,
    )

    suffix = "_proj" if args.mode == "projection" else ""

    def render(tag, box_size, zoom, elevation, gamma):
        t = time.time()
        grid = to_uniform_grid(snap, args.field, res=args.res, box_size=box_size)
        print(f"[{tag}] grid {grid.dims} in {time.time() - t:.1f}s", flush=True)
        t = time.time()
        out = volume_movie(
            snap, grid=grid, zoom=zoom, elevation=elevation, gamma=gamma,
            filename=os.path.join(args.outdir, f"tde_{tag}{suffix}.mp4"),
            frames_dir=f"/tmp/vr_{tag}{suffix}_{jid}", **common,
        )
        dt = time.time() - t
        print(f"[{tag}] {args.nframes} frames in {dt:.1f}s "
              f"({dt / args.nframes:.2f}s/frame) -> {out['filename']}", flush=True)

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
