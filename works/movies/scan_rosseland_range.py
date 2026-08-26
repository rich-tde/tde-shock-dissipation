#!/usr/bin/env python3
"""Study the Rosseland optical-depth (tau = int alpha_ross dr) value distribution
to choose a FIXED colorbar range for the movie.

Per-frame auto-scaling makes the colorbar flicker across a movie, so we want one
fixed ``[vmin, vmax]`` per ``(box, camera)`` that neither clips the bright disk
nor washes out the diffuse envelope over the whole time evolution.  This scans a
handful of snapshots spanning the run, projects tau for each ``(box, camera)``
exactly as the movie does (shared index map from ``render_evolution_multi``, the
opacity grid from ``render_rosseland_movie._opacity_grid``, and
``richio.render.yt_backend._make_projection`` with ``weight=None``), and reports
the pooled percentile distribution of the positive tau pixels.

Output: a JSON dump plus a printed table of percentiles and a recommended fixed
log range per ``(box, camera)`` (a low percentile for vmin, a high one for vmax,
rounded to clean 1/2/5 x 10^k decades).

Example::

    python scan_rosseland_range.py /data1/.../ComptonHiResNewAMR \
        --snaps 21,40,60,80,100,120,140,151 --res 512 \
        --out reports/movies/rosseland/tau_range.json
"""

import argparse
import gc
import json
import os
import sys

import numpy as np

import render_evolution  # BOX_PRESETS, find_snapshots
import render_evolution_multi as rem  # _index_map
import render_rosseland_movie as rrm  # _opacity_grid, CAMERAS
import tde_frame


def _nice_decade(x, direction):
    """Round *x* to a clean 1/2/5 x 10^k bound (down for vmin, up for vmax)."""
    if x <= 0:
        return x
    k = np.floor(np.log10(x))
    mant = x / 10.0**k
    steps = [1.0, 2.0, 5.0, 10.0]
    if direction == "up":
        for m in steps:
            if m >= mant - 1e-9:
                return float(m * 10.0**k)
        return float(10.0 ** (k + 1))
    else:  # down
        for m in reversed(steps):
            if m <= mant + 1e-9:
                return float(m * 10.0**k)
        return float(10.0**k)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir")
    p.add_argument("--boxes", default="A,B")
    p.add_argument("--cameras", default="faceon,side")
    p.add_argument(
        "--snaps",
        default="21,40,60,80,100,120,140,151",
        help="Snapshot indices to pool (span the run).",
    )
    p.add_argument("--coords", default="CMx,CMy,CMz")
    p.add_argument(
        "--res",
        type=int,
        default=512,
        help="Interpolation grid res for the study (production is 1024; "
        "percentile bounds are robust to this).",
    )
    p.add_argument("--resolution", type=int, default=512, help="Projection image px.")
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--m-bh", type=float, default=1e4)
    p.add_argument("--m-star", type=float, default=0.5)
    p.add_argument("--r-star", type=float, default=0.47)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21)
    # Recommended-bound percentiles: low for vmin (drop diffuse background),
    # high for vmax (keep the bright disk without letting a few hot pixels blow
    # out the scale).
    p.add_argument("--pmin", type=float, default=25.0)
    p.add_argument("--pmax", type=float, default=99.5)
    p.add_argument("--out", default="reports/movies/rosseland/tau_range.json")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import richio

    richio.load = tde_frame.make_bh_frame_loader(
        m_bh=args.m_bh,
        m_star=args.m_star,
        r_star=args.r_star,
        beta=args.beta,
        switch_snap=args.switch_snap,
    )
    from richio.render.yt_backend import _make_projection, to_yt

    coords = tuple(args.coords.split(","))
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    snaps = [int(s) for s in args.snaps.split(",") if s.strip()]

    pcts = [0.1, 1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    # Pool positive tau pixels across snapshots per (box, cam).
    pooled = {(b, c): [] for b in boxes for c in cams}

    for box_name in boxes:
        box = render_evolution.BOX_PRESETS[box_name]
        for snap_i in snaps:
            paths = render_evolution.find_snapshots(args.run_dir, snap_i, snap_i)
            if not paths:
                print(
                    f"[scan] WARNING snap_{snap_i} not found; skipping", file=sys.stderr
                )
                continue
            snap = richio.load(paths[0])
            print(
                f"[scan] box={box_name} snap_{snap_i}: index+opacity grid "
                f"(res={args.res}, workers={args.workers})...",
                flush=True,
            )
            i, bbox, dims, tval, tunit = rem._index_map(
                snap, coords, args.res, box, args.workers
            )
            grid = rrm._opacity_grid(snap, i, bbox, dims, tval, tunit, coords)
            ds = to_yt(grid)  # build once, reuse across cameras
            for cam in cams:
                az, el = rrm.CAMERAS[cam]
                arr, _u = _make_projection(
                    grid,
                    "rosseland_alpha",
                    azimuth=az,
                    elevation=el,
                    rot_axis=(0.0, 0.0, 1.0),
                    angle=0.0,
                    zoom=args.zoom,
                    resolution=args.resolution,
                    weight=None,
                    ds=ds,
                )
                v = np.asarray(arr).ravel()
                v = v[np.isfinite(v) & (v > 0)]
                pooled[(box_name, cam)].append(v)
                q = np.percentile(v, pcts)
                print(
                    f"[scan]   {cam:7s} tau: "
                    + " ".join(f"p{p}={qq:.3g}" for p, qq in zip(pcts, q)),
                    flush=True,
                )
                del arr, v
            del ds, grid, i
            gc.collect()

    # Aggregate pooled distribution and recommend fixed bounds per (box, cam).
    print("\n" + "=" * 78)
    print(
        f"Pooled tau distribution and recommended fixed log range "
        f"(vmin=p{args.pmin}, vmax=p{args.pmax}, rounded to decades)"
    )
    print("=" * 78)
    out = {
        "snaps": snaps,
        "res": args.res,
        "resolution": args.resolution,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "percentiles": pcts,
        "configs": {},
    }
    for box_name in boxes:
        for cam in cams:
            chunks = pooled[(box_name, cam)]
            if not chunks:
                continue
            allv = np.concatenate(chunks)
            q = np.percentile(allv, pcts)
            lo_raw = float(np.percentile(allv, args.pmin))
            hi_raw = float(np.percentile(allv, args.pmax))
            vmin = _nice_decade(lo_raw, "down")
            vmax = _nice_decade(hi_raw, "up")
            label = f"{box_name}_{cam}"
            out["configs"][label] = {
                "box": box_name,
                "camera": cam,
                "az_el": list(rrm.CAMERAS[cam]),
                "tau_min": float(allv.min()),
                "tau_max": float(allv.max()),
                "pctl": {str(pp): float(qq) for pp, qq in zip(pcts, q)},
                "vmin_raw": lo_raw,
                "vmax_raw": hi_raw,
                "vmin": vmin,
                "vmax": vmax,
            }
            print(f"\n# box {box_name}, camera {cam} (az,el={rrm.CAMERAS[cam]})")
            print("  " + " ".join(f"p{pp}={qq:.3g}" for pp, qq in zip(pcts, q)))
            print(
                f"  raw  vmin(p{args.pmin})={lo_raw:.4g}  vmax(p{args.pmax})={hi_raw:.4g}"
            )
            print(f"  -->  VMIN={vmin:.4g}  VMAX={vmax:.4g}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[scan] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
