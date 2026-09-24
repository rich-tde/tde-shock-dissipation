#!/usr/bin/env python3
r"""Render a time-evolution movie of Rosseland column optical depth.

The per-cell Rosseland extinction coefficient ``alpha_ross(T, rho)`` in
``cm**-1`` is interpolated from STA tables, resampled onto a uniform grid and
integrated along the camera ray. The resulting ``tau = integral(alpha dl)``
is dimensionless and covers the whole chosen box. It is neither escaping
luminosity nor the optical depth from each individual cell to the surface.

Input files
-----------
``RUN_DIR`` (positional argument)
    Top-level or ``snap_<n>/`` HDF5 files named ``snap_<n>.h5`` or
    ``snap_full_<n>.h5``, or extracted ``snap_<n>/`` NPY directories. Requires
    density, temperature and coordinates (default ``CMx,CMy,CMz``). Optional
    ``tfb_<n>.txt`` files supply fallback-time labels. Snapshot selection uses
    inclusive ``--start``/``--end`` and ``--step``. ``--bh-frame`` and
    ``--switch-snap`` must match the run's frame transition and orbit.
``/home/hey4/RICH/data/STA/{T,rho,ross,scatter}.txt``
    Opacity tables read by ``scripts/opacity_interpolator.py``. Fixed A/B/C
    rendering boxes are in code lengths (solar-radius scale), not mass-scaled.

Output files
------------
``<outdir>/<tag>_<camera>_<box>.mp4``
    H.264 movie; defaults are ``outdir=reports/movies/rosseland``,
    ``tag=rosseland``. Frames follow selected snapshot order at ``--fps``.
    ``imageio`` decodes each as ``uint8 (H, W, 3)`` (row, column, RGB in
    0--255). These are colour pixels rather than optical-depth values.
``<frames-root>/rosseland_alpha/frame_<index:05d>.png``
    Default root: ``<outdir>/<tag>_<camera>_<box>_frames``. ``index`` is a
    zero-based movie position, not snapshot number. PIL RGBA conversion gives
    ``uint8 (H, W, 4)``, with channel 3 alpha. ``--resolution`` sets sampling;
    the colourbar adds image width, and encoding can round to even dimensions.
    Image axes follow the selected camera (``faceon`` looks along z;
    ``side`` is 15 degrees above the midplane). The colourbar is logarithmic
    dimensionless tau; ``--vmin``/``--vmax`` are linear tau values. No optical
    depth array, coordinate grid or opacity cube is saved.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/movies/render_rosseland_movie.py /path/to/run \
        --camera faceon --box A --bh-frame --flip-x --scalebar \
        --outdir data/processed/Movies/rosseland --tag rosseland --keep-frames

Nonempty frames are reused with the same arguments; use a new frame root for
changed inputs or settings. Movies are replaced on encoding; PNGs are removed
unless ``--keep-frames`` is set. ``scan_rosseland_range.py`` helps choose fixed
bounds before a full run; such limits alone establish neither convergence nor
coverage of every epoch. In parallel runs ``FRAME_TIMEOUT`` (default 3600 s)
limits how long the parent waits without a completed frame, leaving PNGs for
resume after failure.

Loading examples
----------------
Inspect a movie frame and its retained PNG::

    import imageio.v2 as imageio
    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/Movies/rosseland")
    with imageio.get_reader(root / "rosseland_faceon_A.mp4") as movie:
        rgb = movie.get_data(0)  # (H, W, 3), uint8
        print(movie.get_meta_data(), rgb.shape)
    path = root / "rosseland_faceon_A_frames/rosseland_alpha/frame_00000.png"
    with Image.open(path) as image:
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

import movie_zoom  # is_pencil, camera_zoom_for_box, box_selection
import numpy as np
import render_evolution  # BOX_PRESETS, find_snapshots, _scalebar_for_box
import render_evolution_multi  # _index_map, _box_geometry
import tde_frame  # make_bh_frame_loader

# Opacity interpolation is a general analysis helper and remains under scripts/.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
)
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
    fields = {
        "rosseland_alpha": np.ascontiguousarray(np.asarray(cube), dtype="float64")
    }
    units = {"rosseland_alpha": str(cube.units)}
    return UniformGrid(
        fields=fields,
        units=units,
        bbox=bbox,
        dims=dims,
        length_unit="cm",
        time=tval,
        time_unit=tunit,
        coords=coords,
    )


def _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg, view):
    from richio.render import volume_image

    grid = _opacity_grid(snap, i, bbox, dims, tval, tunit, cfg["coords"])
    out_png = os.path.join(view["frames_dir"], f"frame_{idx:05d}.png")
    volume_image(
        snap,
        "rosseland_alpha",
        grid=grid,
        mode="projection",
        weight=None,
        flip_x=cfg["flip_x"],
        log=True,
        norm="log",
        vmin=cfg["vmin"],
        vmax=cfg["vmax"],
        cmap="inferno",
        colorbar=True,
        resolution=cfg["resolution"],
        azimuth=cfg["azimuth"],
        elevation=cfg["elevation"],
        zoom=view["zoom"],
        rot_axis=(0.0, 0.0, 1.0),
        annotate=annotate,
        axis_triad=True,
        scalebar_frac=view["scalebar_frac"],
        scalebar_label=view["scalebar_label"],
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
    annotate = (
        _evolution_label(snap, path, cfg["days_per_tfb"])
        if cfg["annotate_time"]
        else None
    )
    for view in todo:
        sel = view["selection_fn"](snap) if view["selection_fn"] is not None else None
        i, bbox, dims, tval, tunit = render_evolution_multi._index_map(
            snap, cfg["coords"], view["res"], view["box"], cfg["workers"], selection=sel
        )
        _render_frame(snap, i, bbox, dims, tval, tunit, annotate, idx, cfg, view)
        del i
    return idx


def _make_view(
    name, box, res, zoom, scalebar_frac, scalebar_label, selection_fn, frames_root
):
    """One rendered view: a box + camera + its own frame directory.

    Single-field sibling of :func:`render_evolution_multi._make_view`; *name* is
    the output-stem suffix (``""`` wide, ``"zoom"`` close-up).
    """
    sub = f"rosseland_alpha_{name}" if name else "rosseland_alpha"
    frames_dir = os.path.join(frames_root, sub)
    os.makedirs(frames_dir, exist_ok=True)
    return dict(
        name=name,
        box=box,
        res=res,
        zoom=zoom,
        scalebar_frac=scalebar_frac,
        scalebar_label=scalebar_label,
        selection_fn=selection_fn,
        frames_dir=frames_dir,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir")
    p.add_argument(
        "--camera",
        default="faceon",
        choices=sorted(CAMERAS),
        help="Named camera preset; overridden by --azimuth/--elevation if given.",
    )
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--box", default="A", choices=sorted(render_evolution.BOX_PRESETS))
    p.add_argument("--coords", default="CMx,CMy,CMz")
    p.add_argument("--start", type=int, default=21)
    p.add_argument("--end", type=int, default=151)
    p.add_argument("--step", type=int, default=1)
    p.add_argument(
        "--res", type=int, default=1024, help="Interpolation grid resolution."
    )
    p.add_argument("--resolution", type=int, default=1024, help="Output image px.")
    p.add_argument("--zoom", type=float, default=1.1)
    p.add_argument(
        "--vmin", type=float, default=None, help="Fixed colorbar vmin (tau, log)."
    )
    p.add_argument(
        "--vmax", type=float, default=None, help="Fixed colorbar vmax (tau, log)."
    )
    p.add_argument("--scalebar", action="store_true")
    p.add_argument("--bh-frame", action="store_true")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument(
        "--no-annotate", action="store_true", help="Drop the time/snap label."
    )
    p.add_argument("--m-bh", type=float, default=1e4)
    p.add_argument("--m-star", type=float, default=0.5)
    p.add_argument("--r-star", type=float, default=0.47)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--switch-snap", type=int, default=21)
    p.add_argument("--days-per-tfb", type=float, default=None)
    p.add_argument(
        "--workers", type=int, default=24, help="KDTree query threads per build."
    )
    p.add_argument(
        "--n-jobs", type=int, default=2, help="Frame-parallel worker processes."
    )
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
            m_bh=args.m_bh,
            m_star=args.m_star,
            r_star=args.r_star,
            beta=args.beta,
            switch_snap=args.switch_snap,
        )
    from richio.render.yt_backend import _cleanup_frames, _encode_movie

    azimuth, elevation = CAMERAS[args.camera]
    azimuth = args.azimuth if args.azimuth is not None else azimuth
    elevation = args.elevation if args.elevation is not None else elevation

    coords = tuple(args.coords.split(","))
    box = render_evolution.BOX_PRESETS[args.box]

    snaps = render_evolution.find_snapshots(args.run_dir, args.start, args.end)[
        :: args.step
    ]
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

    # Camera zoom, scale bar and cell pre-selection all follow from the box; a
    # pencil box (preset C) is framed on its transverse extent, a cube is not
    # touched.  Reuses the multi-field driver's helper so both stay in step.
    args.elevation = elevation
    cam_zoom, scalebar_frac, scalebar_label, selection_fn = (
        render_evolution_multi._box_geometry(args, box, coords, tag="rosseland")
    )

    os.makedirs(args.outdir, exist_ok=True)
    frames_root = args.frames_root or str(
        Path(args.outdir) / f"{args.tag}_{args.camera}_{args.box}_frames"
    )

    views = [
        _make_view(
            "",
            box,
            args.res,
            cam_zoom,
            scalebar_frac,
            scalebar_label,
            selection_fn,
            frames_root,
        )
    ]
    for v in views:
        print(
            f"[rosseland] view: box={[round(b, 2) for b in v['box']]} res={v['res']} "
            f"camera_zoom={v['zoom']:.4f} pencil={movie_zoom.is_pencil(v['box'])} "
            f"bar={v['scalebar_label']}",
            flush=True,
        )

    _CFG.update(
        dict(
            coords=coords,
            box=box,
            res=args.res,
            resolution=args.resolution,
            workers=args.workers,
            azimuth=azimuth,
            elevation=elevation,
            zoom=args.zoom,
            flip_x=args.flip_x,
            vmin=args.vmin,
            vmax=args.vmax,
            scalebar_frac=scalebar_frac,
            scalebar_label=scalebar_label,
            annotate_time=(not args.no_annotate),
            days_per_tfb=days_per_tfb,
            views=views,
        )
    )

    n = len(snaps)
    print(
        f"[rosseland] {n} snapshots, camera={args.camera} (az={azimuth} el={elevation}), "
        f"box={args.box}, res={args.res}, n_jobs={args.n_jobs} workers={args.workers}",
        flush=True,
    )

    tasks = [(idx, snaps[idx]) for idx in range(n)]
    n_jobs = min(args.n_jobs, max(1, len(tasks)))
    if n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        # maxtasksperchild=1 retires each worker after one frame, so peak RSS is
        # one frame's worth rather than whatever the process has accumulated.
        #
        # The timeout is the important part.  When SLURM OOM-kills a worker,
        # imap_unordered waits forever for a result that will never arrive: this
        # driver has twice sat dead for 11 h and 31 h that way, burning walltime
        # while looking exactly like slow progress.  Frames are resumable, so
        # failing loudly and letting a re-submit continue is strictly better than
        # hanging.  A frame takes minutes, so an hour of total silence is a death.
        timeout = float(os.environ.get("FRAME_TIMEOUT", 3600))
        with ctx.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            it = pool.imap_unordered(_render_evolution_frame, tasks)
            for k in range(len(tasks)):
                try:
                    it.next(timeout=timeout)
                except mp.TimeoutError:
                    pool.terminate()
                    print(
                        f"[rosseland] ABORT: no frame completed in {timeout:.0f}s after "
                        f"{k}/{n} -- a worker was most likely OOM-killed. Frames on "
                        f"disk are kept; re-submit to resume (lower NJOBS if it "
                        f"recurs).",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                print(f"[rosseland] frame {k + 1}/{n}", flush=True)
    else:
        for k, t in enumerate(tasks):
            _render_evolution_frame(t)
            print(f"[rosseland] frame {k + 1}/{n}", flush=True)

    for view in views:
        suffix = f"_{view['name']}" if view["name"] else ""
        out = os.path.join(
            args.outdir, f"{args.tag}_{args.camera}_{args.box}{suffix}.mp4"
        )
        _encode_movie(view["frames_dir"], n, out, args.fps)
        print(f"[rosseland] done -> {out}", flush=True)
    if not args.keep_frames:
        _cleanup_frames(frames_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
