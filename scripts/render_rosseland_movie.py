#!/usr/bin/env python3
"""Rosseland optical-depth movie: tau = int alpha_ross dr.

Self-contained sibling of :mod:`render_evolution_multi` for a single field that
is *not* stored on disk: alpha_ross(T, rho), the Rosseland extinction
coefficient [cm^-1] interpolated per cell from the STA opacity table via
:mod:`opacity_interpolator`. Reuses the shared-index-map machinery
(:func:`render_evolution_multi._index_map`) and BH-frame shim
(:mod:`tde_frame`), but builds its own :class:`~richio.render.grid.UniformGrid`
instead of going through richio's ``DERIVED_FIELDS`` registry, so nothing here
touches the richio package.

``mode="projection", weight=None`` makes ``richio.render.volume_image`` compute
a plain, unweighted ``int field dl`` along the camera ray -- with the field
tagged ``1/cm`` against a ``length_unit="cm"`` grid, that integral is exactly
the dimensionless Rosseland optical depth tau.

Example (face-on, box A)::

    python render_rosseland_movie.py /data1/.../ComptonHiResNewAMR \
        --camera faceon --box A --bh-frame --flip-x --scalebar \
        --outdir reports/movies/rosseland --tag rosseland
"""

import argparse
import os
import sys

import numpy as np

import render_evolution  # BOX_PRESETS, find_snapshots, _scalebar_for_box
import tde_frame  # make_bh_frame_loader
import opacity_interpolator

#: Named camera presets (azimuth, elevation), matching scan_color_range.py /
#: jobs/submit_movies.sh: "faceon" = exact xy-plane projection; "side" = the
#: g3 roughly-side-on view (15 deg above the midplane, not fully edge-on).
CAMERAS = {"faceon": (0.0, 90.0), "side": (0.0, 15.0)}

# Worker config (small, picklable); set in the parent before forking so the Pool
# children inherit it. Heavy per-snapshot data is loaded inside each worker.
_CFG: dict = {}


def _opacity_grid(snap, i, bbox, dims, tval, tunit, coords, unit_system="cgs"):
    """Per-cell alpha_ross resampled onto the shared uniform-grid index map.

    Mirrors ``render_evolution_multi._field_grid``'s unyt handling: keep the
    field as a real ``unyt_array`` (not a bare ndarray with a hand-written unit
    string) until the last step, and let unyt's own ``.in_base()`` conversion
    report the unit string, so a units bug would show up as a unyt error/wrong
    label instead of being silently masked by a hardcoded ``"1/cm"``.
    """
    from richio.render.grid import UniformGrid

    T_cgs = snap._get_data("temperature").in_cgs()
    rho_cgs = snap._get_data("density").in_cgs()
    sigma = opacity_interpolator.rosseland_alpha(
        np.asarray(T_cgs, dtype="float64"), np.asarray(rho_cgs, dtype="float64")
    )  # unyt_array, cm**-1
    cube = sigma[i].in_base(unit_system)
    fields = {"rosseland_alpha": np.ascontiguousarray(np.asarray(cube), dtype="float64")}
    units = {"rosseland_alpha": str(cube.units)}
    return UniformGrid(
        fields=fields, units=units,
        bbox=bbox, dims=dims, length_unit="cm", time=tval, time_unit=tunit, coords=coords,
    )


def _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg):
    from richio.render import volume_image

    grid = _opacity_grid(snap, i, bbox, dims, tval, tunit, cfg["coords"])
    out_png = os.path.join(cfg["frames_dir"], f"frame_{idx:05d}.png")
    volume_image(
        snap, "rosseland_alpha", grid=grid, mode="projection", weight=None,
        flip_x=cfg["flip_x"], log=True, norm="log", vmin=cfg["vmin"], vmax=cfg["vmax"],
        cmap="inferno", colorbar=True, resolution=cfg["resolution"],
        azimuth=cfg["azimuth"], elevation=cfg["elevation"], zoom=cfg["zoom"],
        rot_axis=(0.0, 0.0, 1.0), annotate=annotate, axis_triad=True,
        scalebar_frac=cfg["scalebar_frac"], scalebar_label=cfg["scalebar_label"],
        filename=out_png,
    )


def _render_evolution_frame(task):
    """One frame (own process loads its own snapshot, one tree build).

    Resume-safe: if the target PNG already exists and is non-empty (e.g. from a
    previous run that hit the walltime), skip the expensive load+tree+project and
    return immediately. Combined with a persistent --frames-root on the shared
    filesystem, a re-submit continues from where a timed-out run left off instead
    of re-rendering everything.
    """
    from richio.render.evolution import _evolution_label
    import richio
    import render_evolution_multi

    idx, path = task
    cfg = _CFG
    out_png = os.path.join(cfg["frames_dir"], f"frame_{idx:05d}.png")
    if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
        return idx  # already rendered — resume
    snap = richio.load(path)
    i, bbox, dims, tval, tunit = render_evolution_multi._index_map(
        snap, cfg["coords"], cfg["res"], cfg["box"], cfg["workers"]
    )
    annotate = _evolution_label(snap, path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg)
    return idx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--camera", default="faceon", choices=sorted(CAMERAS),
                   help="Named camera preset; overridden by --azimuth/--elevation if given.")
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--box", default="A", choices=sorted(render_evolution.BOX_PRESETS))
    p.add_argument("--coords", default="CMx,CMy,CMz")
    p.add_argument("--start", type=int, default=21)
    p.add_argument("--end", type=int, default=151)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--res", type=int, default=1024, help="Interpolation grid resolution.")
    p.add_argument("--resolution", type=int, default=1024, help="Output image px.")
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument("--vmin", type=float, default=None, help="Fixed colorbar vmin (tau, log).")
    p.add_argument("--vmax", type=float, default=None, help="Fixed colorbar vmax (tau, log).")
    p.add_argument("--scalebar", action="store_true")
    p.add_argument("--bh-frame", action="store_true")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--no-annotate", action="store_true", help="Drop the time/snap label.")
    p.add_argument("--m-bh", type=float, default=1e4)
    p.add_argument("--m-star", type=float, default=0.5)
    p.add_argument("--r-star", type=float, default=0.47)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21)
    p.add_argument("--days-per-tfb", type=float, default=None)
    p.add_argument("--workers", type=int, default=24, help="KDTree query threads per build.")
    p.add_argument("--n-jobs", type=int, default=2, help="Frame-parallel worker processes.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--outdir", default="reports/movies/rosseland")
    p.add_argument("--tag", default="rosseland")
    p.add_argument("--frames-root", default=None)
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import richio
    if args.bh_frame:
        richio.load = tde_frame.make_bh_frame_loader(
            m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star,
            beta=args.beta, switch_snap=args.switch_snap,
        )
    from richio.render.yt_backend import _cleanup_frames, _encode_movie

    azimuth, elevation = CAMERAS[args.camera]
    azimuth = args.azimuth if args.azimuth is not None else azimuth
    elevation = args.elevation if args.elevation is not None else elevation

    coords = tuple(args.coords.split(","))
    box = render_evolution.BOX_PRESETS[args.box]

    snaps = render_evolution.find_snapshots(args.run_dir, args.start, args.end)[:: args.step]
    if not snaps:
        print(f"No snapshots found in {args.run_dir}", file=sys.stderr)
        return 1

    # Scale bar sized to the tidal radius r_t = R*(M_BH/M*)^(1/3), as render_evolution.
    scalebar_frac = scalebar_label = None
    if args.scalebar:
        r_t = args.r_star * (args.m_bh / args.m_star) ** (1.0 / 3.0)
        scalebar_frac, scalebar_label = render_evolution._scalebar_for_box(box, args.zoom, r_t)

    os.makedirs(args.outdir, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    frames_root = args.frames_root or os.path.abspath(f"/tmp/{args.tag}_{args.camera}_{jid}")
    frames_dir = os.path.join(frames_root, "rosseland_alpha")
    os.makedirs(frames_dir, exist_ok=True)

    _CFG.update(dict(
        coords=coords, box=box, res=args.res, resolution=args.resolution,
        workers=args.workers, azimuth=azimuth, elevation=elevation, zoom=args.zoom,
        flip_x=args.flip_x, vmin=args.vmin, vmax=args.vmax,
        scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
        annotate_time=(not args.no_annotate), days_per_tfb=args.days_per_tfb,
        frames_dir=frames_dir,
    ))

    n = len(snaps)
    print(f"[rosseland] {n} snapshots, camera={args.camera} (az={azimuth} el={elevation}), "
          f"box={args.box}, res={args.res}, n_jobs={args.n_jobs} workers={args.workers}",
          flush=True)

    tasks = [(idx, snaps[idx]) for idx in range(n)]
    n_jobs = min(args.n_jobs, max(1, len(tasks)))
    if n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for k, _ in enumerate(pool.imap_unordered(_render_evolution_frame, tasks)):
                print(f"[rosseland] frame {k + 1}/{n}", flush=True)
    else:
        for k, t in enumerate(tasks):
            _render_evolution_frame(t)
            print(f"[rosseland] frame {k + 1}/{n}", flush=True)

    out = os.path.join(args.outdir, f"{args.tag}_{args.camera}_{args.box}.mp4")
    _encode_movie(frames_dir, n, out, args.fps)
    print(f"[rosseland] done -> {out}", flush=True)
    if not args.keep_frames:
        _cleanup_frames(frames_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
