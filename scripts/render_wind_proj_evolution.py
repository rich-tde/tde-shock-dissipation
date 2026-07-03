#!/usr/bin/env python3
"""Time-evolution movie of the unbound-wind density & dissipation projections.

Reproduces the 3-plane column projections from
``notebooks/windstudy/0.1-tde-wind.ipynb`` for *every* snapshot and stitches the
frames into a movie.  Each frame is a 2x3 panel:

* **top row** density, **bottom row** dissipation;
* **columns** are the xy / xz / yz projection planes.

Only the unbound, radially-outflowing wind is kept (``B>0 & v_r>0`` -- the
notebook's ``wind_mask``).  Following the notebook this is a *value* mask: cells
outside the wind have their field set to zero and still enter the column
integral (so non-wind sight-lines read as empty), rather than a ``selection``
mask, which would drop them from the nearest-neighbour grid and smear the wind
across the box.

The rendering path is :meth:`richio.data.Snapshot.plots.projection` (matplotlib
column integral), the same as the notebook -- *not* the yt off-axis projection
used by ``render_evolution.py``.  Frames are split across forked workers
(``--n-jobs``); the colour scale is fixed once from a reference snapshot so the
colorbars don't flicker.

Example::

    python render_wind_proj_evolution.py /data1/.../ComptonHiResNewAMR \
        --start 21 --end 151 --box-half 200 --res 512 --n-jobs 6 \
        --out reports/movies/wind_proj/wind_proj_wide.mp4
"""

import argparse
import os
import sys

import numpy as np


# scripts/ is on sys.path[0] when run as ``python scripts/...``; these are the
# sibling helper modules we reuse rather than re-implement.
import render_evolution  # find_snapshots
import tde_frame  # select_unbound_outflow

PLANES = ("xy", "xz", "yz")
# (xlabel, ylabel) for each plane's projected map.
PLANE_LABELS = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
# Axis permutation (axis1, axis2, normal) of the physical-(x,y,z) index cube for
# each plane, matching richio.data._parse_plane.  The third axis is integrated.
PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}

# Filled in main() before forking so the Pool workers inherit it copy-on-write.
_CFG: dict = {}


def _read_tfb(path):
    """Return the fallback time t_fb for a snapshot path, or None.

    Mirrors the lookup in ``render_evolution._spin_indices``: each snapshot dir
    holds ``tfb_<n>.txt`` with a single float.
    """
    d = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        n = int(os.path.basename(d).split("_")[1])
        with open(os.path.join(d, f"tfb_{n}.txt")) as fh:
            return float(fh.read().strip())
    except Exception:
        return None


def _masked_field(snap, field, mask):
    """Field array with non-wind cells zeroed (the notebook's value mask)."""
    arr = snap._get_data(field).copy()
    arr[~mask] = 0
    return arr


def _wind_mask(snap, coords):
    """Notebook ``wind_mask``: B>0 & v_r>0 (no cone / x-side cut)."""
    return tde_frame.select_unbound_outflow(
        snap, zr_max=None, x_sign=0, coords=coords
    )


def _project_all(snap, fields, mask, coords, box, res, tree_workers, unit_system="cgs"):
    """All field/plane projections from a **single** nearest-neighbour index map.

    ``project`` rebuilds the k-d tree on every call, so reproducing the notebook
    naively costs 6 tree builds per frame (2 fields x 3 planes).  For a symmetric
    cube at equal resolution all three planes share the same grid points, so we
    build the physical-(x,y,z) index cube once via ``to_3dgrid`` and obtain each
    plane by transposing and integrating along the right axis -- the identical
    ``sum(cube[:-1,:-1,:-1]*dz, axis=-1).in_base(...)`` formula ``project`` uses.

    :returns: ``{field: [(plane, proj2d, xspace, yspace), ...]}``.
    """
    i, xs, ys, zs = snap.to_3dgrid(
        res, coords[0], coords[1], coords[2], box, None, workers=tree_workers
    )
    spaces = (xs, ys, zs)  # physical x/y/z coordinate spaces (equal for a cube)
    out = {}
    for field in fields:
        vals = _masked_field(snap, field, mask)[i]  # (nx, ny, nz), field units
        panels = []
        for plane in PLANES:
            a1, a2, a3 = PLANE_AXES[plane]
            cube = vals.transpose(a1, a2, a3)
            sp = spaces[a3]
            dz = sp[1:] - sp[:-1]
            proj = np.sum(cube[:-1, :-1, :-1] * dz, axis=-1).in_base(unit_system)
            panels.append((plane, proj, spaces[a1], spaces[a2]))
        out[field] = panels
    return out


def _log_range(projected, lo_pct=1.0, hi_pct=99.5):
    """(vmin, vmax) in log10 space, percentile-clipped and rounded to half-ints.

    Matches ``scalar_map``'s half-integer rounding (plots.py) but uses
    percentiles instead of raw min/max so a single hot pixel can't blow the
    scale across a whole movie.
    """
    v = np.asarray(projected, dtype="float64")
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None, None
    lv = np.log10(v)
    lo = np.percentile(lv, lo_pct)
    hi = np.percentile(lv, hi_pct)
    return float(np.floor(lo * 2.0) / 2.0), float(np.ceil(hi * 2.0) / 2.0)


def _fix_color_scale(ref_path, fields, coords, res, box, tree_workers, overrides):
    """Per-field (vmin, vmax) in log10 space from the reference snapshot.

    Pools all three planes so one scale serves every panel/frame.  ``overrides``
    (field -> (vmin, vmax)) wins when set; one-sided overrides are honoured.
    """
    import richio

    # Skip the (expensive) reference projection if every scale is overridden.
    if all(None not in overrides.get(f, (None, None)) for f in fields):
        return {f: overrides[f] for f in fields}

    snap = richio.load(ref_path)
    mask = _wind_mask(snap, coords)
    panels = _project_all(snap, fields, mask, coords, box, res, tree_workers)
    scales = {}
    for field in fields:
        pooled = np.concatenate([np.asarray(p).ravel() for _, p, _, _ in panels[field]])
        vmin, vmax = _log_range(pooled)
        ov = overrides.get(field, (None, None))
        scales[field] = (ov[0] if ov[0] is not None else vmin,
                         ov[1] if ov[1] is not None else vmax)
    return scales


def render_frame(args):
    """Render one snapshot's 2x3 panel to a PNG.  Reads config from ``_CFG``."""
    idx, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import richio
    from richio.plots import scalar_map

    cfg = _CFG
    coords = cfg["coords"]
    box = cfg["box"]
    res = cfg["res"]
    fields = cfg["fields"]
    cmaps = cfg["cmaps"]
    scales = cfg["scales"]

    snap = richio.load(path)
    mask = _wind_mask(snap, coords)
    panels = _project_all(snap, fields, mask, coords, box, res, cfg["tree_workers"])

    fig, axes = plt.subplots(len(fields), 3, figsize=(17, 4 * len(fields)),
                             squeeze=False)
    for row, field in enumerate(fields):
        vmin, vmax = scales[field]
        for col, (plane, proj, xspace, yspace) in enumerate(panels[field]):
            ax = axes[row][col]
            scalar_map(
                proj, xspace, yspace, ax=ax, cmap=cmaps[row],
                label_latex=field, vmin=vmin, vmax=vmax,
            )
            xl, yl = PLANE_LABELS[plane]
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            if plane in ("xy", "xz"):
                ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            if row == 0:
                ax.set_title(f"{field}  [{plane}]")

    tfb = _read_tfb(path)
    title = os.path.basename(path if os.path.isdir(path) else os.path.dirname(path))
    if tfb is not None:
        title += rf"   $t/t_{{fb}} = {tfb:.2f}$"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = os.path.join(cfg["frames_dir"], f"frame_{idx:05d}.png")
    fig.savefig(out_png, dpi=cfg["dpi"])
    plt.close(fig)
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Run directory containing snap_<i>/ subdirs.")
    p.add_argument("--start", type=int, default=21,
                   help="First snapshot (default 21: first post-switch / BH-frame snap).")
    p.add_argument("--end", type=int, default=151)
    p.add_argument("--step", type=int, default=1, help="Use every Nth snapshot.")
    p.add_argument("--coords", default="CMx,CMy,CMz", help="Coordinate fields (comma-sep).")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--box-half", type=float, default=200.0,
                   help="Symmetric cube half-width in code length units (notebook used "
                        "200 and 30). Ignored when --box is given.")
    p.add_argument("--box", default="",
                   help="Explicit box 'x0,y0,z0,x1,y1,z1' (code length); overrides "
                        "--box-half. Use equal extents per axis so pixels stay square.")
    p.add_argument("--fields", default="density,dissipation",
                   help="Comma-separated fields, one per row.")
    p.add_argument("--cmaps", default="twilight,viridis",
                   help="Comma-separated colormaps, one per field row.")
    p.add_argument("--ref-index", type=int, default=-1,
                   help="Index into the snapshot list used to fix the colour scale.")
    p.add_argument("--vmin", default="",
                   help="Per-field log10 vmin overrides, comma-sep aligned with --fields "
                        "(blank entry = auto).")
    p.add_argument("--vmax", default="", help="Per-field log10 vmax overrides (see --vmin).")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--n-jobs", type=int, default=6, help="Forked frame workers (1 = serial).")
    p.add_argument("--tree-workers", type=int, default=2,
                   help="KDTree query threads per frame's single grid build "
                        "(keep low: frames are already process-parallel via --n-jobs).")
    p.add_argument("--out", default="wind_proj.mp4")
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from richio.render.yt_backend import _cleanup_frames, _encode_movie

    coords = tuple(args.coords.split(","))
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    cmaps = [c.strip() for c in args.cmaps.split(",") if c.strip()]
    if len(cmaps) != len(fields):
        print("--cmaps must have one entry per --fields", file=sys.stderr)
        return 1
    if args.box:
        box = tuple(float(v) for v in args.box.split(","))
        if len(box) != 6:
            print("--box must be 'x0,y0,z0,x1,y1,z1'", file=sys.stderr)
            return 1
    else:
        h = args.box_half
        box = (-h, -h, -h, h, h, h)

    def _parse_overrides(s):
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            out.append(float(tok) if tok else None)
        return out
    vmins = _parse_overrides(args.vmin) if args.vmin else [None] * len(fields)
    vmaxs = _parse_overrides(args.vmax) if args.vmax else [None] * len(fields)
    overrides = {f: (vmins[i] if i < len(vmins) else None,
                     vmaxs[i] if i < len(vmaxs) else None)
                 for i, f in enumerate(fields)}

    snaps = render_evolution.find_snapshots(args.run_dir, args.start, args.end)[:: args.step]
    if not snaps:
        print(f"No snapshots found in {args.run_dir}", file=sys.stderr)
        return 1

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    frames_dir = args.frames_dir or os.path.abspath("wind_proj_frames")
    os.makedirs(frames_dir, exist_ok=True)

    ref_path = snaps[args.ref_index]
    print(f"[wind_proj] {len(snaps)} snapshots, box={box}, res={args.res}, "
          f"fields={fields}, n_jobs={args.n_jobs}", flush=True)
    print(f"[wind_proj] fixing colour scale from {ref_path}", flush=True)
    scales = _fix_color_scale(ref_path, fields, coords, args.res, box,
                              args.tree_workers, overrides)
    for f in fields:
        print(f"[wind_proj]   {f}: log10 vmin/vmax = {scales[f]}", flush=True)

    _CFG.update(dict(coords=coords, box=box, res=args.res, fields=fields,
                     cmaps=cmaps, scales=scales, frames_dir=frames_dir, dpi=args.dpi,
                     tree_workers=args.tree_workers))

    work = list(enumerate(snaps))
    n_jobs = min(args.n_jobs, len(work))
    if n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, _ in enumerate(pool.imap_unordered(render_frame, work)):
                print(f"[wind_proj] frame {i + 1}/{len(work)}", flush=True)
    else:
        for j, item in enumerate(work):
            render_frame(item)
            print(f"[wind_proj] frame {j + 1}/{len(work)}", flush=True)

    _encode_movie(frames_dir, len(work), args.out, args.fps)
    if not args.keep_frames:
        _cleanup_frames(frames_dir)
    print(f"[wind_proj] done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
