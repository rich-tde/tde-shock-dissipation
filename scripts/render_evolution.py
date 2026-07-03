#!/usr/bin/env python3
"""Render a time-evolution movie of a RICH run (one frame per snapshot).

Auto-detects usable snapshots in a run directory (``snap_<i>/snap_<i>.h5`` if
present, else the ``snap_<i>/`` NPY directory) and feeds them to
:func:`richio.render.evolution_movie`.  Box and colour scale are fixed from a
reference snapshot, so only the data (and optionally the camera) change.

Example::

    python render_evolution.py /data1/.../ComptonHiResNewAMR \
        --mode projection --box disk --res 224 --resolution 1024 \
        --n-jobs 6 --out reports/gifs/evo_disk_proj.mp4
"""

import argparse
import os
import sys


# Fixed boxes (R☉, code length): A is a cube around the origin, B is the wide
# downstream/pericenter view.  Format is [x0, y0, z0, x1, y1, z1].
BOX_PRESETS = {
    "A": [-400.0, -400.0, -400.0, 400.0, 400.0, 400.0],
    "B": [-2000.0, -1400.0, -1400.0, 800.0, 1400.0, 1400.0],
    "C": [-200.0, -200.0, -200.0, 200.0, 200.0, 200.0],  # ±200 cube close-up
}


def _nice_n(target):
    """Nearest 'round' multiple to *target* for a tidy ``n r_t`` scale-bar label."""
    candidates = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
    return min(candidates, key=lambda c: abs(c - target))


def _scalebar_for_box(box, zoom, r_t, target_frac=0.2):
    """Scale-bar (fraction-of-width, label) for a ``n·r_t`` bar on a fixed *box*.

    The rendered field of view spans ``max(box extent) / zoom`` in length units
    (the off-axis camera width), so a bar of ``n·r_t`` is that long a fraction of
    the image.  *n* is rounded to a clean value near ``target_frac`` of the view.
    """
    extent = [box[3] - box[0], box[4] - box[1], box[5] - box[2]]
    fov = max(extent) / float(zoom)
    n = _nice_n(target_frac * fov / r_t)
    frac = n * r_t / fov
    return float(frac), rf"${n}\,r_t$"


def _spin_indices(snaps, spin_tfb):
    """Snapshot indices nearest each target time (in t_fb), read from tfb_<n>.txt."""
    tfbs = []
    for p in snaps:
        d = p if os.path.isdir(p) else os.path.dirname(p)
        try:
            n = int(os.path.basename(d).split("_")[1])
            with open(os.path.join(d, f"tfb_{n}.txt")) as fh:
                tfbs.append(float(fh.read().strip()))
        except Exception:
            tfbs.append(None)
    idxs = []
    for target in spin_tfb:
        cand = [(abs(t - target), i) for i, t in enumerate(tfbs) if t is not None]
        if cand:
            idxs.append(min(cand)[1])
    return sorted(set(idxs))


def find_snapshots(run_dir, start, end):
    snaps = []
    for i in range(start, end + 1):
        h5 = os.path.join(run_dir, f"snap_{i}", f"snap_{i}.h5")
        npy = os.path.join(run_dir, f"snap_{i}", f"Den_{i}.npy")
        if os.path.exists(h5):
            snaps.append(h5)
        elif os.path.exists(npy):
            snaps.append(os.path.join(run_dir, f"snap_{i}"))
    return snaps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Run directory containing snap_<i>/ subdirs.")
    p.add_argument("--field", default="density")
    p.add_argument("--mode", default="projection", choices=["projection", "volume"])
    p.add_argument("--weight", default=None,
                   help="Projection weight field (e.g. density for a weighted mean; "
                        "use for intensive fields like temperature).")
    p.add_argument("--box", default="wide", choices=["wide", "disk", "A", "B", "C"],
                   help="wide/disk derive a box from the data; A/B/C are fixed boxes "
                        "(A=cube +-400, B=x[-2000,800] y,z[-1400,1400], C=cube +-200, R_sun).")
    p.add_argument("--select", default="none", choices=["none", "wind"],
                   help="'wind' renders only the unbound disc-plane wind: B>0 & v_r>0 & "
                        "|z|/r<=--cone-zr (cells zeroed elsewhere).")
    p.add_argument("--cone-zr", type=float, default=0.5,
                   help="Equatorial-wedge cut |z|/r for --select wind (0.5 = +-30 deg of xy).")
    p.add_argument("--plane", default="none", choices=["none", "xy", "xz", "yz"],
                   help="Exact axis-aligned line-of-sight projection (integrate along the third "
                        "axis): xy=top-down, xz=along y, yz=along x. Forces no rotation.")
    p.add_argument("--scalebar", action="store_true",
                   help="Draw a map-style scale bar of ~n*r_t (fixed boxes only).")
    p.add_argument("--spin-tfb", default="",
                   help="Comma-separated t_fb times to pause and spin 360 (e.g. 0.5,1.0,1.5). "
                        "Empty keeps the single trailing spin.")
    p.add_argument("--final-turns", type=int, default=1,
                   help="Full turns at the final snapshot when --spin-tfb is set.")
    p.add_argument("--box-field", default=None,
                   help="Field defining the auto box (default: --field). "
                        "E.g. density to match a density movie's framing.")
    p.add_argument("--disk-radius", type=float, default=300.0)
    p.add_argument("--rotate", action="store_true", help="Orbit camera while time advances.")
    p.add_argument("--flip-x", action="store_true", help="Reverse the displayed x-axis.")
    p.add_argument("--total-angle", type=float, default=360.0)
    p.add_argument("--spin-frames", type=int, default=0,
                   help="Trailing frames orbiting the final state (0=off).")
    p.add_argument("--spin-angle", type=float, default=360.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=151)
    p.add_argument("--step", type=int, default=1, help="Use every Nth snapshot.")
    p.add_argument("--ref-index", type=int, default=-1)
    p.add_argument("--coords", default="CMx,CMy,CMz", help="Coordinate fields (comma-sep).")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8,
                   help="KDTree NN query threads per grid build (raise for high --res "
                        "with few --n-jobs so one build uses all cores).")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--vmin", type=float, default=None, help="Fixed colorbar vmin (cgs).")
    p.add_argument("--vmax", type=float, default=None, help="Fixed colorbar vmax (cgs).")
    p.add_argument("--norm", default="log", choices=["log", "linear", "symlog"],
                   help="Colour norm; symlog for signed fields (e.g. bernoulli).")
    p.add_argument("--linthresh", type=float, default=1.0, help="SymLogNorm linear threshold.")
    p.add_argument("--m-bh", type=float, default=1e4, help="BH mass (Msun) for derived fields.")
    p.add_argument("--m-star", type=float, default=0.5, help="Star mass (Msun).")
    p.add_argument("--r-star", type=float, default=0.47, help="Star radius (Rsun).")
    p.add_argument("--elevation", type=float, default=26.0)
    p.add_argument("--azimuth", type=float, default=45.0)
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--out", default="evolution.mp4")
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--bh-frame", action="store_true",
                   help="Transform pre-switch (star-frame) snapshots into the BH frame.")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21,
                   help="First snapshot already in the BH frame (earlier ones get shifted).")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import richio
    if args.bh_frame:
        # Scripts-only shim: pre-switch snapshots are returned already in the BH
        # frame.  Forked render workers inherit this monkeypatched loader.
        import tde_frame
        richio.load = tde_frame.make_bh_frame_loader(
            m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star, beta=args.beta,
            switch_snap=args.switch_snap,
        )
    from richio.render import evolution_movie

    snaps = find_snapshots(args.run_dir, args.start, args.end)[:: args.step]
    if not snaps:
        print(f"No snapshots found in {args.run_dir}", file=sys.stderr)
        return 1
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    coords = tuple(args.coords.split(","))

    # Fixed box preset (A/B) vs data-derived box (wide/disk).
    box_arg = BOX_PRESETS.get(args.box)
    box_kind_arg = "wide" if box_arg is not None else args.box

    # Cell selection: the unbound disc-plane wind (B>0 & v_r>0 & |z|/r<=cone_zr).
    selection_fn = None
    if args.select == "wind":
        import functools
        import tde_frame

        selection_fn = functools.partial(
            tde_frame.select_unbound_outflow, zr_max=args.cone_zr,
            m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star, coords=coords,
        )

    # Exact axis-aligned plane projection: override the camera to look straight
    # down the integration axis and disable all rotation (static plane map).
    azimuth, elevation, rotate, spin_frames = (
        args.azimuth, args.elevation, args.rotate, args.spin_frames)
    if args.plane != "none":
        azimuth, elevation = {"xy": (0.0, 90.0), "xz": (0.0, 0.0), "yz": (90.0, 0.0)}[args.plane]
        rotate, spin_frames = False, 0
        print(f"[render_evolution] plane={args.plane} -> azimuth={azimuth}, "
              f"elevation={elevation}, no rotation", flush=True)

    # Scale bar sized to the TDE tidal radius r_t = R*(M_BH/M*)^(1/3).
    scalebar_frac = scalebar_label = None
    if args.scalebar:
        r_t = args.r_star * (args.m_bh / args.m_star) ** (1.0 / 3.0)
        if box_arg is not None:
            scalebar_frac, scalebar_label = _scalebar_for_box(box_arg, args.zoom, r_t)
            print(f"[render_evolution] r_t = {r_t:.3f} R_sun -> scale bar "
                  f"{scalebar_label} (frac={scalebar_frac:.3f})", flush=True)
        else:
            print(f"[render_evolution] --scalebar needs a fixed box (A/B); skipping "
                  f"(r_t={r_t:.3f})", flush=True)

    # Multi-spin: pause and spin at the given t_fb times.
    spin_at = None
    spin_tfb = [float(s) for s in args.spin_tfb.split(",") if s.strip()]
    if spin_tfb:
        spin_at = _spin_indices(snaps, spin_tfb)
        print(f"[render_evolution] spin at t_fb {spin_tfb} -> snapshot indices {spin_at}, "
              f"final turns={args.final_turns}", flush=True)

    print(f"[render_evolution] {len(snaps)} snapshots, mode={args.mode}, box={args.box}, "
          f"select={args.select}, plane={args.plane}, rotate={rotate}, res={args.res}, "
          f"n_jobs={args.n_jobs}", flush=True)

    out = evolution_movie(
        snaps,
        field=args.field,
        mode=args.mode,
        weight=args.weight,
        box=box_arg,
        box_kind=box_kind_arg,
        box_field=args.box_field,
        disk_radius=args.disk_radius,
        coords=coords,
        ref_index=args.ref_index,
        res=args.res,
        resolution=args.resolution,
        workers=args.workers,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        norm=args.norm,
        linthresh=args.linthresh,
        m_bh=args.m_bh,
        m_star=args.m_star,
        r_star=args.r_star,
        azimuth=azimuth,
        elevation=elevation,
        zoom=args.zoom,
        rotate=rotate,
        flip_x=args.flip_x,
        total_angle=args.total_angle,
        spin_frames=spin_frames,
        spin_angle=args.spin_angle,
        spin_at=spin_at,
        final_spin_turns=args.final_turns,
        selection_fn=selection_fn,
        scalebar_frac=scalebar_frac,
        scalebar_label=scalebar_label,
        fps=args.fps,
        n_jobs=args.n_jobs,
        filename=args.out,
        frames_dir=args.frames_dir,
        keep_frames=args.keep_frames,
    )
    print(f"[render_evolution] done -> {out['filename']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
