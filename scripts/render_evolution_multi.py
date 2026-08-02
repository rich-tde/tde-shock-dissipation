#!/usr/bin/env python3
"""Shared-index, multi-field time-evolution projection movies.

A multi-field generalisation of :func:`richio.render.evolution_movie`.  The
nearest-neighbour index map that resamples the unstructured cells onto the
uniform grid is **geometry only** (it depends on cell positions, not field
values), so this driver builds it **once per snapshot** and projects *every*
requested field from it.  The single-threaded k-d-tree build is the dominant
per-frame cost, so sharing it across fields cuts the work by ~``len(fields)``
versus running one ``render_evolution.py`` job per field.

Memory is kept bounded by materialising **one field cube at a time** (the index
map plus a single cube, ~17 GB at res 1024) instead of all cubes at once, so a
node can still run ``--n-jobs 2`` for unweighted fields.

One movie is written per field: ``<outdir>/<tag>_<field>.mp4``.

Example (the conference batch: g3 side view, density+dissipation, box A)::

    python render_evolution_multi.py /data1/.../ComptonHiResNewAMR \
        --fields density,dissipation --box A --azimuth 0 --elevation 15 \
        --res 1024 --resolution 1024 --bh-frame --flip-x --scalebar \
        --vmins -, --vmaxs -,  --n-jobs 2 --workers 24 \
        --outdir reports/movies/angles --tag g3_side
"""

import argparse
import functools
import os
import sys

import numpy as np

# scripts/ is on sys.path[0]; reuse the existing single-field driver's helpers.
import movie_zoom  # pencil_box, camera_zoom_for, beam_selection
import render_evolution  # BOX_PRESETS, find_snapshots, _scalebar_for_box
import tde_frame  # make_bh_frame_loader, select_unbound_outflow


# Per-field rendering recipe (matches jobs/render_evolution_v2.slurm's case block):
# cmap, colour norm, projection weight field, symlog linear threshold.
FIELD_RECIPE = {
    "density":     dict(cmap="twilight", norm="log",    weight=None,      linthresh=1.0),
    "dissipation": dict(cmap="viridis",  norm="log",    weight=None,      linthresh=1.0),
    "temperature": dict(cmap="inferno",  norm="log",    weight="density", linthresh=1.0),
    "bernoulli":   dict(cmap="RdBu_r",   norm="symlog", weight="density", linthresh=1.0),
}

# Worker config (small, picklable); set in the parent before forking so the Pool
# children inherit it. Heavy per-snapshot data is loaded inside each worker.
_CFG: dict = {}


def _index_map(snap, coords, res, box, workers, selection=None):
    """Build the geometric NN index map once and return it with grid metadata.

    Mirrors the index-map + bbox/dims/time computation inside
    :func:`richio.render.grid.to_uniform_grid`, but stops *before* materialising
    any field cube so the caller can apply it to one field at a time.

    *selection* restricts which cells enter the k-d tree (the zoom views pass a
    pencil-beam mask so the tree is built from a small subset).  ``to_3dgrid``
    maps the result back to absolute particle indices, so the returned *i* still
    indexes full-length field arrays either way.
    """
    cx, cy, cz = coords
    i, xspace, yspace, zspace = snap.to_3dgrid(
        res, X=cx, Y=cy, Z=cz, box_size=box, workers=workers, selection=selection
    )
    dims = tuple(int(s) for s in i.shape)

    def _edges(space):  # to_3dgrid uses endpoint=False: true upper edge = last + d
        s = space.to("cm")
        d = s[1] - s[0]
        return float(s[0].value), float((s[-1] + d).value)

    bbox = np.array([_edges(xspace), _edges(yspace), _edges(zspace)], dtype="float64")

    t = getattr(snap, "time", None)
    time_val = time_unit = None
    if t is not None:
        t = np.atleast_1d(t)
        time_val = float(np.asarray(t)[0])
        time_unit = str(t.units) if hasattr(t, "units") else None
    return i, bbox, dims, time_val, time_unit


def _field_grid(snap, i, bbox, dims, time_val, time_unit, coords, field, weight,
                mask, derived_kw, unit_system):
    """A one-field (+optional weight) UniformGrid sharing the precomputed index *i*.

    Reuses :class:`richio.render.grid.UniformGrid`.  When *mask* is given the
    field is **zeroed** outside the selection (not dropped) before sampling, so
    the k-d tree can't smear selected values into the emptied space — the same
    rule as :func:`richio.render.evolution._build_grid`.
    """
    from richio.render.grid import UniformGrid
    from richio.render.derived import DERIVED_FIELDS

    def _source(name):
        if name in DERIVED_FIELDS:
            return DERIVED_FIELDS[name](snap, **derived_kw)
        return snap._get_data(name)

    names = [field] + ([weight] if weight and weight != field else [])
    fields, units = {}, {}
    for name in names:
        data = _source(name)
        if mask is not None:
            data = data * mask
        cube = data[i].in_base(unit_system)
        fields[name] = np.ascontiguousarray(np.asarray(cube), dtype="float64")
        units[name] = str(cube.units)
    return UniformGrid(fields=fields, units=units, bbox=bbox, dims=dims,
                       length_unit="cm", time=time_val, time_unit=time_unit,
                       coords=coords)


def _render_fields(snap, i, bbox, dims, tval, tunit, annotate, azimuth, idx, cfg, view,
                   mask=None):
    """Project and save every field for one camera angle from a shared index map.

    *view* carries the per-view geometry (box, camera zoom, scale bar, output
    directory) so the wide and close-up views share this code path.
    """
    from richio.render import volume_image

    for field in cfg["fields"]:
        rec = cfg["recipe"][field]
        grid = _field_grid(snap, i, bbox, dims, tval, tunit, cfg["coords"], field,
                           rec["weight"], mask, cfg["derived_kw"], cfg["unit_system"])
        out_png = os.path.join(view["frames_dir"][field], f"frame_{idx:05d}.png")
        volume_image(
            snap, field, grid=grid, mode="projection", weight=rec["weight"],
            flip_x=cfg["flip_x"], log=(rec["norm"] == "log"), norm=rec["norm"],
            linthresh=rec["linthresh"], vmin=cfg["vmins"][field], vmax=cfg["vmaxs"][field],
            cmap=rec["cmap"], colorbar=True, resolution=cfg["resolution"],
            azimuth=azimuth, elevation=cfg["elevation"], zoom=view["zoom"],
            rot_axis=(0.0, 0.0, 1.0), annotate=annotate, axis_triad=True,
            scalebar_frac=view["scalebar_frac"], scalebar_label=view["scalebar_label"],
            filename=out_png,
        )
        del grid


def _render_evolution_frame(task):
    """One evolution frame (own process loads its own snapshot, one tree build/view).

    Resume-safe: a view whose PNGs all exist already is skipped, so a re-submit
    after a walltime kill continues instead of redoing finished frames.
    """
    from richio.render.evolution import _evolution_label
    import richio

    idx, path = task
    cfg = _CFG
    todo = [v for v in cfg["views"] if not _view_done(v, idx, cfg["fields"])]
    if not todo:
        return idx
    snap = richio.load(path)
    annotate = _evolution_label(snap, path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    mask = cfg["mask_fn"](snap) if cfg["mask_fn"] is not None else None
    for view in todo:
        sel = view["selection_fn"](snap) if view["selection_fn"] is not None else None
        i, bbox, dims, tval, tunit = _index_map(snap, cfg["coords"], view["res"],
                                                view["box"], cfg["workers"], selection=sel)
        _render_fields(snap, i, bbox, dims, tval, tunit, annotate, cfg["azimuth"],
                       idx, cfg, view, mask)
        del i
    return idx


def _view_done(view, idx, fields):
    """True when every field's PNG for this frame already exists and is non-empty."""
    for field in fields:
        p = os.path.join(view["frames_dir"][field], f"frame_{idx:05d}.png")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False
    return True


def _render_spin(final_path, n_evo, cfg, k_lo=0, k_hi=None):
    """Trailing camera orbit on the final snapshot (one tree build, reused).

    Builds the index map once for the final snapshot and re-projects each field
    across the azimuth sweep, appending frames after the evolution frames.  Only
    spin-frame indices in ``[k_lo, k_hi)`` are rendered (default: all), so the
    sweep can be chunked across jobs while keeping the shared index build.
    """
    from richio.render import volume_image
    from richio.render.evolution import _evolution_label
    import richio

    if k_hi is None:
        k_hi = cfg["spin_frames"]
    if k_hi <= k_lo:
        return
    snap = richio.load(final_path)
    annotate = _evolution_label(snap, final_path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    mask = cfg["mask_fn"](snap) if cfg["mask_fn"] is not None else None
    angles = cfg["azimuth"] + np.linspace(0.0, cfg["spin_angle"], cfg["spin_frames"],
                                          endpoint=False)
    for view in cfg["views"]:
        sel = view["selection_fn"](snap) if view["selection_fn"] is not None else None
        i, bbox, dims, tval, tunit = _index_map(snap, cfg["coords"], view["res"],
                                                view["box"], cfg["workers"], selection=sel)
        for field in cfg["fields"]:
            rec = cfg["recipe"][field]
            grid = _field_grid(snap, i, bbox, dims, tval, tunit, cfg["coords"], field,
                               rec["weight"], mask, cfg["derived_kw"], cfg["unit_system"])
            for k in range(k_lo, k_hi):
                az = angles[k]
                out_png = os.path.join(view["frames_dir"][field],
                                       f"frame_{n_evo + k:05d}.png")
                volume_image(
                    snap, field, grid=grid, mode="projection", weight=rec["weight"],
                    flip_x=cfg["flip_x"], log=(rec["norm"] == "log"), norm=rec["norm"],
                    linthresh=rec["linthresh"], vmin=cfg["vmins"][field],
                    vmax=cfg["vmaxs"][field],
                    cmap=rec["cmap"], colorbar=True, resolution=cfg["resolution"],
                    azimuth=float(az), elevation=cfg["elevation"], zoom=view["zoom"],
                    rot_axis=(0.0, 0.0, 1.0), annotate=annotate, axis_triad=True,
                    scalebar_frac=view["scalebar_frac"],
                    scalebar_label=view["scalebar_label"],
                    filename=out_png,
                )
            del grid
        del i


def _make_view(name, box, res, zoom, scalebar_frac, scalebar_label, selection_fn,
               frames_root, fields):
    """One rendered view: a box + camera + its own frame directories.

    The wide view and any close-up are just different entries in this list, so
    they share the whole render path and differ only in geometry.  *name* is the
    output-stem suffix (``""`` for the wide view, ``"zoom"`` for the close-up).
    """
    sub = f"{{f}}_{name}" if name else "{f}"
    frames_dir = {f: os.path.join(frames_root, sub.format(f=f)) for f in fields}
    for d in frames_dir.values():
        os.makedirs(d, exist_ok=True)
    return dict(name=name, box=box, res=res, zoom=zoom, scalebar_frac=scalebar_frac,
                scalebar_label=scalebar_label, selection_fn=selection_fn,
                frames_dir=frames_dir)


def _zoom_views(args, box, frames_root, fields, coords):
    """The close-up view list for ``--zoom-rp`` (empty when off or not face-on).

    Restricted to face-on cameras: the pencil-beam box only preserves the wide
    view's column integral when the line of sight is z, so a tilted camera would
    silently produce a *different* quantity rather than a magnified one.
    """
    if not args.zoom_rp:
        return []
    if abs(args.elevation - 90.0) > 1e-6:
        print(f"[multi] --zoom-rp ignored: needs a face-on camera "
              f"(elevation 90, got {args.elevation})", flush=True)
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
                       sel, frames_root, fields)]


def _parse_aligned(s, fields, cast=str):
    """Parse a comma-separated per-field override list aligned with *fields*.

    Blank entries (or a shorter/empty list) mean "no override" (None).
    """
    toks = [t.strip() for t in s.split(",")] if s else []
    out = {}
    for j, f in enumerate(fields):
        tok = toks[j] if j < len(toks) else ""
        out[f] = cast(tok) if tok else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--fields", default="density,dissipation",
                   help="Comma-separated fields; one movie each (shared index map).")
    p.add_argument("--box", default="A", choices=sorted(render_evolution.BOX_PRESETS),
                   help="Fixed box preset (A/B/C).")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument("--res", type=int, default=1024, help="Interpolation grid resolution.")
    p.add_argument("--resolution", type=int, default=1024, help="Output image px.")
    p.add_argument("--start", type=int, default=21)
    p.add_argument("--end", type=int, default=151)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--coords", default="CMx,CMy,CMz")
    p.add_argument("--select", default="none", choices=["none", "wind"],
                   help="'wind' zeroes all but the unbound disc-plane outflow.")
    p.add_argument("--cone-zr", type=float, default=0.5)
    p.add_argument("--vmins", default="", help="Per-field log/symlog vmin (comma-sep, aligned).")
    p.add_argument("--vmaxs", default="", help="Per-field vmax (comma-sep, aligned).")
    p.add_argument("--scalebar", action="store_true")
    p.add_argument("--bh-frame", action="store_true")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--no-annotate", action="store_true", help="Drop the time/snap label.")
    p.add_argument("--zoom-rp", type=float, default=0.0,
                   help="Also render a face-on close-up reaching this many r_p in x and y "
                        "(e.g. 2.5); 0 = off.  The close-up keeps the wide box's full "
                        "line-of-sight extent, so it is a true magnification sharing the "
                        "wide colour limits.  Face-on cameras only (elevation 90).")
    p.add_argument("--zoom-res", type=int, default=None,
                   help="Interpolation resolution for the close-up (default: --res, which "
                        "makes its z spacing identical to the wide grid's).")
    p.add_argument("--spin-frames", type=int, default=90)
    p.add_argument("--spin-angle", type=float, default=360.0)
    p.add_argument("--m-bh", type=float, default=1e4)
    p.add_argument("--m-star", type=float, default=0.5)
    p.add_argument("--r-star", type=float, default=0.47)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21)
    p.add_argument("--days-per-tfb", type=float, default=None)
    p.add_argument("--workers", type=int, default=24, help="KDTree query threads per build.")
    p.add_argument("--n-jobs", type=int, default=2, help="Frame-parallel worker processes.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--outdir", default="reports/movies/angles")
    p.add_argument("--tag", default="g3_side", help="Output stem: <outdir>/<tag>_<field>.mp4.")
    p.add_argument("--name-template", default="{tag}_{field}",
                   help="Output stem template; '{tag}' and '{field}' are substituted. "
                        "Use e.g. 'g2_{field}_A' to place the field mid-name "
                        "(-> <outdir>/g2_density_A.mp4).")
    p.add_argument("--frames-root", default=None)
    p.add_argument("--keep-frames", action="store_true")
    # Chunked rendering (for short-walltime partitions): render a global frame-index
    # window into a shared frames-root across several jobs, then encode once.
    # Global index space: evolution frames are [0, n_snaps), spin frames are
    # [n_snaps, n_snaps + spin_frames).  A natural split is evolution vs spin.
    p.add_argument("--frame-start", type=int, default=0,
                   help="First global frame index to render (inclusive).")
    p.add_argument("--frame-stop", type=int, default=-1,
                   help="Stop global frame index (exclusive); -1 = all frames.")
    p.add_argument("--no-encode", action="store_true",
                   help="Render frames only (keep them); skip movie encode/cleanup.")
    p.add_argument("--encode-only", action="store_true",
                   help="Skip rendering; encode the full frames-root into movies.")
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import richio
    if args.bh_frame:
        import tde_frame as _tf
        richio.load = _tf.make_bh_frame_loader(
            m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star,
            beta=args.beta, switch_snap=args.switch_snap,
        )
    from richio.render.yt_backend import _cleanup_frames, _encode_movie

    coords = tuple(args.coords.split(","))
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    unknown = [f for f in fields if f not in FIELD_RECIPE]
    if unknown:
        print(f"unknown field(s): {unknown}; known: {list(FIELD_RECIPE)}", file=sys.stderr)
        return 1
    recipe = {f: dict(FIELD_RECIPE[f]) for f in fields}
    box = render_evolution.BOX_PRESETS[args.box]
    vmins = _parse_aligned(args.vmins, fields, float)
    vmaxs = _parse_aligned(args.vmaxs, fields, float)

    snaps = render_evolution.find_snapshots(args.run_dir, args.start, args.end)[:: args.step]
    if not snaps:
        print(f"No snapshots found in {args.run_dir}", file=sys.stderr)
        return 1

    # Most snapshots here are NPY directories, which store no time -- only the
    # early/late .h5 ones do.  Without this calibration the label's "t = ... d"
    # line silently vanishes on every NPY frame, so derive days-per-tfb once in
    # the parent (it is a run-wide constant) and hand it to the workers.
    days_per_tfb = args.days_per_tfb
    if days_per_tfb is None:
        from richio.render.evolution import _calibrate_days_per_tfb

        days_per_tfb = _calibrate_days_per_tfb(snaps)
    print(f"[multi] days_per_tfb = {days_per_tfb}", flush=True)

    # Scale bar sized to the tidal radius r_t = R*(M_BH/M*)^(1/3), as render_evolution.
    scalebar_frac = scalebar_label = None
    if args.scalebar:
        r_t = args.r_star * (args.m_bh / args.m_star) ** (1.0 / 3.0)
        scalebar_frac, scalebar_label = render_evolution._scalebar_for_box(box, args.zoom, r_t)

    mask_fn = None
    if args.select == "wind":
        mask_fn = functools.partial(
            tde_frame.select_unbound_outflow, zr_max=args.cone_zr,
            m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star, coords=coords,
        )

    os.makedirs(args.outdir, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    frames_root = args.frames_root or os.path.abspath(f"/tmp/{args.tag}_{jid}")

    views = [_make_view("", box, args.res, args.zoom, scalebar_frac, scalebar_label,
                        None, frames_root, fields)]
    views += _zoom_views(args, box, frames_root, fields, coords)
    for v in views:
        print(f"[multi] view '{v['name'] or 'wide'}': box={[round(b, 2) for b in v['box']]} "
              f"res={v['res']} camera_zoom={v['zoom']:.4f} bar={v['scalebar_label']}",
              flush=True)

    _CFG.update(dict(
        fields=fields, recipe=recipe, coords=coords, box=box, res=args.res,
        resolution=args.resolution, workers=args.workers, azimuth=args.azimuth,
        elevation=args.elevation, zoom=args.zoom, flip_x=args.flip_x,
        vmins=vmins, vmaxs=vmaxs, scalebar_frac=scalebar_frac,
        scalebar_label=scalebar_label, mask_fn=mask_fn, unit_system="cgs",
        derived_kw=dict(m_bh=args.m_bh, m_star=args.m_star, r_star=args.r_star,
                        coords=coords),
        annotate_time=(not args.no_annotate), days_per_tfb=days_per_tfb,
        views=views, spin_frames=args.spin_frames, spin_angle=args.spin_angle,
    ))

    n = len(snaps)
    total = n + args.spin_frames  # global frame count: evolution [0,n) + spin [n,total)
    fstart = max(0, args.frame_start)
    fstop = total if args.frame_stop < 0 else min(total, args.frame_stop)
    print(f"[multi] {n} snapshots, fields={fields}, box={args.box}, "
          f"az={args.azimuth} el={args.elevation}, res={args.res}, "
          f"n_jobs={args.n_jobs} workers={args.workers}, spin={args.spin_frames}, "
          f"frames[{fstart},{fstop}) of {total}, encode_only={args.encode_only}, "
          f"no_encode={args.no_encode}", flush=True)
    for f in fields:
        print(f"[multi]   {f}: cmap={recipe[f]['cmap']} norm={recipe[f]['norm']} "
              f"weight={recipe[f]['weight']} vmin={vmins[f]} vmax={vmaxs[f]}", flush=True)

    if not args.encode_only:
        # Evolution frames whose global index falls in the window.
        evo_lo, evo_hi = fstart, min(n, fstop)
        tasks = [(idx, snaps[idx]) for idx in range(evo_lo, evo_hi)]
        n_jobs = min(args.n_jobs, max(1, len(tasks)))
        if tasks and n_jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_jobs) as pool:
                for k, _ in enumerate(pool.imap_unordered(_render_evolution_frame, tasks)):
                    print(f"[multi] evolution frame {k + 1}/{len(tasks)} "
                          f"(window [{evo_lo},{evo_hi}))", flush=True)
        else:
            for k, t in enumerate(tasks):
                _render_evolution_frame(t)
                print(f"[multi] evolution frame {k + 1}/{len(tasks)} "
                      f"(window [{evo_lo},{evo_hi}))", flush=True)

        # Spin frames whose global index falls in the window (k = gid - n).
        if args.spin_frames > 0 and fstop > n:
            k_lo, k_hi = max(0, fstart - n), min(args.spin_frames, fstop - n)
            print(f"[multi] trailing spin: frames {k_lo}..{k_hi} on {snaps[-1]}", flush=True)
            _render_spin(snaps[-1], n, _CFG, k_lo=k_lo, k_hi=k_hi)

    if args.no_encode:
        print(f"[multi] rendered frames [{fstart},{fstop}); skipping encode "
              f"(frames kept in {frames_root})", flush=True)
        return 0

    for view in views:
        suffix = f"_{view['name']}" if view["name"] else ""
        for f in fields:
            stem = args.name_template.format(tag=args.tag, field=f)
            out = os.path.join(args.outdir, f"{stem}{suffix}.mp4")
            _encode_movie(view["frames_dir"][f], total, out, args.fps)
            print(f"[multi] done -> {out}", flush=True)
    if not args.keep_frames:
        _cleanup_frames(frames_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
