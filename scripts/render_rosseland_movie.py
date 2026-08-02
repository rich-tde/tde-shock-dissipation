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
import functools
import os
import sys

import numpy as np

import movie_zoom  # pencil_box, camera_zoom_for, beam_selection
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


def _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg, view):
    from richio.render import volume_image

    grid = _opacity_grid(snap, i, bbox, dims, tval, tunit, cfg["coords"])
    out_png = os.path.join(view["frames_dir"], f"frame_{idx:05d}.png")
    volume_image(
        snap, "rosseland_alpha", grid=grid, mode="projection", weight=None,
        flip_x=cfg["flip_x"], log=True, norm="log", vmin=cfg["vmin"], vmax=cfg["vmax"],
        cmap="inferno", colorbar=True, resolution=cfg["resolution"],
        azimuth=cfg["azimuth"], elevation=cfg["elevation"], zoom=view["zoom"],
        rot_axis=(0.0, 0.0, 1.0), annotate=annotate, axis_triad=True,
        scalebar_frac=view["scalebar_frac"], scalebar_label=view["scalebar_label"],
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
    todo = []
    for view in cfg["views"]:
        p = os.path.join(view["frames_dir"], f"frame_{idx:05d}.png")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            todo.append(view)
    if not todo:
        return idx  # already rendered — resume
    snap = richio.load(path)
    annotate = _evolution_label(snap, path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    for view in todo:
        sel = view["selection_fn"](snap) if view["selection_fn"] is not None else None
        i, bbox, dims, tval, tunit = render_evolution_multi._index_map(
            snap, cfg["coords"], view["res"], view["box"], cfg["workers"], selection=sel
        )
        _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg, view)
        del i
    return idx


def _make_view(name, box, res, zoom, scalebar_frac, scalebar_label, selection_fn,
               frames_root):
    """One rendered view: a box + camera + its own frame directory.

    Single-field sibling of :func:`render_evolution_multi._make_view`; *name* is
    the output-stem suffix (``""`` wide, ``"zoom"`` close-up).
    """
    sub = f"rosseland_alpha_{name}" if name else "rosseland_alpha"
    frames_dir = os.path.join(frames_root, sub)
    os.makedirs(frames_dir, exist_ok=True)
    return dict(name=name, box=box, res=res, zoom=zoom, scalebar_frac=scalebar_frac,
                scalebar_label=scalebar_label, selection_fn=selection_fn,
                frames_dir=frames_dir)


def _zoom_views(args, box, elevation, frames_root, coords):
    """The close-up view list for ``--zoom-rp`` (empty when off or not face-on).

    Face-on only: the pencil-beam box preserves the full line-of-sight integral
    only when the line of sight is z, and a truncated integral would no longer be
    the optical depth.
    """
    if not args.zoom_rp:
        return []
    if abs(elevation - 90.0) > 1e-6:
        print(f"[rosseland] --zoom-rp ignored: needs a face-on camera "
              f"(elevation 90, got {elevation})", flush=True)
        return []

    r_p = movie_zoom.pericentre_radius(args.m_bh, args.m_star, args.r_star, args.beta)
    half_width = args.zoom_rp * r_p
    zbox = movie_zoom.pencil_box(box, half_width)
    cam_zoom = movie_zoom.camera_zoom_for(zbox, 2.0 * half_width)
    frac, label = movie_zoom.scalebar_in_rp(zbox, cam_zoom, r_p)
    sel = functools.partial(movie_zoom.beam_selection, coords=coords,
                            half_width=half_width)
    return [_make_view("zoom", zbox, args.zoom_res or args.res, cam_zoom,
                       frac if args.scalebar else None,
                       label if args.scalebar else None,
                       sel, frames_root)]


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
    p.add_argument("--zoom-rp", type=float, default=0.0,
                   help="Also render a face-on close-up reaching this many r_p in x and y "
                        "(e.g. 2.5); 0 = off.  Keeps the wide box's full line-of-sight "
                        "extent, so tau stays a true optical depth and the wide colour "
                        "limits still apply.  Face-on camera only.")
    p.add_argument("--zoom-res", type=int, default=None,
                   help="Interpolation resolution for the close-up (default: --res).")
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

    # NPY snapshots store no time, so the label's "t = ... d" line needs the
    # run-wide days-per-tfb factor; derive it once here (see render_evolution_multi).
    days_per_tfb = args.days_per_tfb
    if days_per_tfb is None:
        from richio.render.evolution import _calibrate_days_per_tfb

        days_per_tfb = _calibrate_days_per_tfb(snaps)
    print(f"[rosseland] days_per_tfb = {days_per_tfb}", flush=True)

    # Scale bar sized to the tidal radius r_t = R*(M_BH/M*)^(1/3), as render_evolution.
    scalebar_frac = scalebar_label = None
    if args.scalebar:
        r_t = args.r_star * (args.m_bh / args.m_star) ** (1.0 / 3.0)
        scalebar_frac, scalebar_label = render_evolution._scalebar_for_box(box, args.zoom, r_t)

    os.makedirs(args.outdir, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    frames_root = args.frames_root or os.path.abspath(f"/tmp/{args.tag}_{args.camera}_{jid}")

    views = [_make_view("", box, args.res, args.zoom, scalebar_frac, scalebar_label,
                        None, frames_root)]
    views += _zoom_views(args, box, elevation, frames_root, coords)
    for v in views:
        print(f"[rosseland] view '{v['name'] or 'wide'}': "
              f"box={[round(b, 2) for b in v['box']]} res={v['res']} "
              f"camera_zoom={v['zoom']:.4f} bar={v['scalebar_label']}", flush=True)

    _CFG.update(dict(
        coords=coords, box=box, res=args.res, resolution=args.resolution,
        workers=args.workers, azimuth=azimuth, elevation=elevation, zoom=args.zoom,
        flip_x=args.flip_x, vmin=args.vmin, vmax=args.vmax,
        scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
        annotate_time=(not args.no_annotate), days_per_tfb=days_per_tfb,
        views=views,
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

    for view in views:
        suffix = f"_{view['name']}" if view["name"] else ""
        out = os.path.join(args.outdir,
                           f"{args.tag}_{args.camera}_{args.box}{suffix}.mp4")
        _encode_movie(view["frames_dir"], n, out, args.fps)
        print(f"[rosseland] done -> {out}", flush=True)
    if not args.keep_frames:
        _cleanup_frames(frames_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
