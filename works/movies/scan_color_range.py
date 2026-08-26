"""Scan first+last snapshots for fixed projection colour limits.

For each ``(box, camera)`` and field, project the **first** and **last** snapshot
and take the **union** of their auto colour bounds, so the movie's fixed colour
scale won't clip the early or late state (the production driver otherwise fixes the
scale from the last snapshot alone).  Only the first+last frames are touched, so the
cost is ~one KDTree index build per ``(box, snap)`` — cheap relative to a full pass.

Everything heavy is **reused**: the geometry index + per-field grid from
``render_evolution_multi`` (``_index_map`` / ``_field_grid`` / ``FIELD_RECIPE``), the
off-axis projection and percentile-bound helpers from ``richio.render.yt_backend``
(``_make_projection`` / ``_auto_bounds`` / ``_sym_bounds``), and the BH-frame loader
from ``tde_frame`` — so the scanned numbers match what the movies will render.

Output: a JSON dump plus ready-to-paste ``--vmins``/``--vmaxs`` lines (aligned to the
field order) for each group config.

Example::

    python scan_color_range.py /data1/.../ComptonHiResNewAMR \
        --res 1024 --resolution 1024 --out reports/movies/color_ranges.json
"""

import argparse
import gc
import json
import os
import sys

import render_evolution  # BOX_PRESETS, find_snapshots
import render_evolution_multi as rem  # _index_map, _field_grid, FIELD_RECIPE
import tde_frame

# Field order MUST match jobs/submit_v2_groups.sh / render_multi.slurm FIELDS.
FIELDS = ["density", "dissipation", "temperature", "bernoulli"]

# Named cameras (azimuth, elevation), matching the submit scripts.
CAMERAS = {
    "canonical": (45.0, 26.0),  # g2 default 3/4 view
    "side": (0.0, 15.0),  # g3 side (along y)
    "top": (20.0, 72.0),  # g3 top (along z)
    "faceon": (0.0, 90.0),  # exact xy-plane projection (look down z), no rotation
}

# Which group configs to emit paste-ready lines for: (label, box, camera).
GROUP_CONFIGS = [
    ("g2_A", "A", "canonical"),
    ("g2_B", "B", "canonical"),
    ("g3_side_A", "A", "side"),
    ("g3_side_B", "B", "side"),
    ("g3_top_A", "A", "top"),
    ("g3_top_B", "B", "top"),
    ("faceon_A", "A", "faceon"),
    ("faceon_B", "B", "faceon"),
]


def _bounds_for_array(arr, norm):
    """(lo, hi) colour bounds for one projected field, using the movie's own rule."""
    from richio.render.yt_backend import _auto_bounds, _sym_bounds

    if norm == "symlog":
        v = _sym_bounds(arr, round_nice=True)
        return -float(v), float(v)
    lo, hi = _auto_bounds(arr, log=(norm == "log"))
    return float(lo), float(hi)


def _union(b1, b2, norm):
    """Union two (lo, hi) bounds; symmetric for symlog."""
    if norm == "symlog":
        v = max(abs(b1[0]), abs(b1[1]), abs(b2[0]), abs(b2[1]))
        return -v, v
    return min(b1[0], b2[0]), max(b1[1], b2[1])


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir")
    p.add_argument("--boxes", default="A,B")
    p.add_argument(
        "--snaps", default="21,151", help="First,last snapshot indices to union."
    )
    p.add_argument("--cameras", default="canonical,side,top")
    p.add_argument("--coords", default="CMx,CMy,CMz")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--m-bh", type=float, default=1e4)
    p.add_argument("--m-star", type=float, default=0.5)
    p.add_argument("--r-star", type=float, default=0.47)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21)
    p.add_argument("--out", default="reports/movies/color_ranges.json")
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
    snaps = [int(s) for s in args.snaps.split(",") if s.strip()]
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    derived_kw = {
        "m_bh": args.m_bh,
        "m_star": args.m_star,
        "r_star": args.r_star,
        "coords": coords,
    }

    # per-snap[box] -> {(field, cam): (lo, hi)}
    per_snap = {}
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
                f"[scan] box={box_name} snap_{snap_i}: building index "
                f"(res={args.res}, workers={args.workers})...",
                flush=True,
            )
            i, bbox, dims, tval, tunit = rem._index_map(
                snap, coords, args.res, box, args.workers
            )
            for field in FIELDS:
                rec = rem.FIELD_RECIPE[field]
                grid = rem._field_grid(
                    snap,
                    i,
                    bbox,
                    dims,
                    tval,
                    tunit,
                    coords,
                    field,
                    rec["weight"],
                    None,
                    derived_kw,
                    "cgs",
                )
                ds = to_yt(grid)  # build the yt dataset once; reuse across cameras
                for cam in cams:
                    az, el = CAMERAS[cam]
                    arr, _u = _make_projection(
                        grid,
                        field,
                        azimuth=az,
                        elevation=el,
                        rot_axis=(0.0, 0.0, 1.0),
                        angle=0.0,
                        zoom=args.zoom,
                        resolution=args.resolution,
                        weight=rec["weight"],
                        ds=ds,
                    )
                    lo, hi = _bounds_for_array(arr, rec["norm"])
                    per_snap.setdefault((box_name, snap_i), {})[(field, cam)] = (lo, hi)
                    print(
                        f"[scan]   {field:11s} {cam:9s} -> [{lo:.4g}, {hi:.4g}]",
                        flush=True,
                    )
                    del arr
                del ds, grid
                gc.collect()
            del i
            gc.collect()

    # Union first+last per (box, field, cam).
    first, last = snaps[0], snaps[-1]
    unioned = {}  # (box, field, cam) -> (lo, hi)
    for box_name in boxes:
        b_first = per_snap.get((box_name, first), {})
        b_last = per_snap.get((box_name, last), {})
        for field in FIELDS:
            norm = rem.FIELD_RECIPE[field]["norm"]
            for cam in cams:
                key = (field, cam)
                if key in b_first and key in b_last:
                    unioned[(box_name, field, cam)] = _union(
                        b_first[key], b_last[key], norm
                    )
                elif key in b_last:
                    unioned[(box_name, field, cam)] = b_last[key]
                elif key in b_first:
                    unioned[(box_name, field, cam)] = b_first[key]

    # Emit paste-ready lines per group config.
    print("\n" + "=" * 72)
    print("Paste-ready colour limits (field order: " + ",".join(FIELDS) + ")")
    print("=" * 72)
    out_json = {
        "fields": FIELDS,
        "snaps": snaps,
        "res": args.res,
        "resolution": args.resolution,
        "configs": {},
    }
    for label, box_name, cam in GROUP_CONFIGS:
        if box_name not in boxes or cam not in cams:
            continue
        vmins, vmaxs = [], []
        for field in FIELDS:
            lo, hi = unioned[(box_name, field, cam)]
            vmins.append(f"{lo:.6g}")
            vmaxs.append(f"{hi:.6g}")
        vmins_s, vmaxs_s = ",".join(vmins), ",".join(vmaxs)
        out_json["configs"][label] = {
            "box": box_name,
            "camera": cam,
            "vmins": vmins_s,
            "vmaxs": vmaxs_s,
        }
        print(f"\n# {label} (box {box_name}, camera {cam})")
        print(f"VMINS={vmins_s}")
        print(f"VMAXS={vmaxs_s}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"\n[scan] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
