#!/usr/bin/env python3
r"""Render RICH projections or slices as a PNG sequence and an animated GIF.

Nearest-neighbour resampling produces a line integral (``projection``) or a
section at ``--slice-coordinate`` (``slice``). A fixed logarithmic colour range
is determined from the final selected snapshot, rounded outward to half
decades, unless ``--vmin``/``--vmax`` give log10 cgs bounds. Earlier frames can
clip; this is a visualization rather than a grid-convergence test. Coordinates
are used as stored, without a BH-frame correction.

Input files
-----------
``RUN_DIR`` (positional argument)
    Run directory with top-level or ``snap_<n>/`` HDF5 files named
    ``snap_<n>.h5``/``snap_full_<n>.h5``, or extracted ``snap_<n>/`` NPY
    directories. Requires the selected ``--field`` (repeatable; default
    ``dissipation``) and ``--coords`` (default ``CMx,CMy,CMz``).
``--box u0 v0 w0 u1 v1 w1``
    Bounds in RICH code lengths (solar-radius scale). ``u,v`` are the displayed
    axes from ``--plane``; ``w`` is the remaining integration/slice axis. The
    default is ``[-6,-4,-2,2.5,3,2] * R_* (M_BH/M_*)**(2/3)`` in that order.
    ``--m-bh``, ``--m-star`` and ``--r-star`` affect only this default box.

Output files
------------
``<out>/<field>_<proj|slice>_<plane>/<same-stem>_snap_<n:04d>.png``
    Annotated PNG with a quantitative colourbar; ``n`` is the snapshot number.
    Default ``out`` is ``data/processed/Movies/Gifs`` under the repository.
    Loading with ``PIL.Image.convert("RGBA")`` gives ``uint8 (H, W, 4)`` in
    row, column, red/green/blue/alpha order (channel values 0--255). Canvas size
    is 6 by 5 inches at 150 dpi before tight cropping, so ``H,W`` depend on
    labels. Image x/y axes follow the two letters in ``--plane``. The colourbar
    displays log10 cgs values: density is ``g/cm**2`` in projections or
    ``g/cm**3`` in slices; dissipation is ``erg/s/cm**2`` or ``erg/s/cm**3``.
``<out>/<field>_<proj|slice>_<plane>.gif``
    Looping, palette-encoded animation in increasing selected snapshot order.
    ``--duration`` is display seconds per frame (default 0.2). Decode each
    frame to RGB for a ``uint8 (H, W, 3)`` array. GIF palette quantization can
    alter colours. Neither format stores the scientific arrays or coordinates.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/movies/make_gif.py /path/to/run --start 21 --end 30 \
        --field density --plane yz --res 128
    python works/movies/make_gif.py /path/to/run --field density \
        --plane xy --plot-kind slice --slice-coordinate 0 --res 256 \
        --box -100 -100 -20 100 100 20 --out data/processed/Movies/xy-slices

``--start``/``--end`` are inclusive snapshot numbers; ``--step`` subsamples
available snapshots. Existing PNGs/GIFs are reused unless ``--overwrite`` is
set; ``--resume`` is a compatibility alias for that default. Use a new ``--out``
for changed inputs/settings. ``--dry-run`` lists paths without loading data.

Loading examples
----------------
Inspect a PNG and count animation frames for the first command above::

    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/Movies/Gifs")
    stem = "density_proj_yz"
    with Image.open(root / stem / f"{stem}_snap_0021.png") as image:
        rgba = np.array(image.convert("RGBA"))  # (H, W, 4), uint8
    with Image.open(root / f"{stem}.gif") as movie:
        print(movie.n_frames, movie.info.get("duration"))  # count, milliseconds
        movie.seek(0)
        first_rgb = np.array(movie.convert("RGB"))  # (H, W, 3), uint8
"""

import argparse
import logging
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import dev  # noqa: F401  # isort: skip  # Configure style before pyplot.
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from render_evolution import find_snapshots

import richio

LOG = logging.getLogger("make_gif")


def sample_map(snapshot, field, args, box):
    """Return a unitful map and axes with coordinates permuted into viewing order."""
    order = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[args.plane]
    coords = args.coords.split(",")
    coordinate_unit = snapshot._get_data(coords[order[0]]).units
    kwargs = {
        "res": args.res,
        "X": coords[order[0]],
        "Y": coords[order[1]],
        "Z": coords[order[2]],
        "box_size": box.to(coordinate_unit),
        "workers": args.workers,
    }
    if args.plot_kind == "slice":
        return snapshot.slice(
            field,
            slice_coord=(args.slice_coordinate * richio.units.lscale).to(
                coordinate_unit
            ),
            **kwargs,
        )
    return snapshot.project(field, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_dir", help="Run directory containing snapshots.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=150)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--field",
        action="append",
        help="Repeat for multiple fields (default: dissipation).",
    )
    parser.add_argument(
        "--plot-kind", choices=("projection", "slice"), default="projection"
    )
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="yz")
    parser.add_argument("--coords", default="CMx,CMy,CMz")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--box",
        type=float,
        nargs=6,
        help="u0 v0 w0 u1 v1 w1, in code lengths; see module docstring.",
    )
    parser.add_argument(
        "--m-bh", type=float, default=1e4, help="Solar masses; default box only."
    )
    parser.add_argument(
        "--m-star", type=float, default=0.5, help="Solar masses; default box only."
    )
    parser.add_argument(
        "--r-star", type=float, default=0.47, help="Solar radii; default box only."
    )
    parser.add_argument("--slice-coordinate", type=float, default=0.0)
    parser.add_argument("--cmap", default="cividis")
    parser.add_argument(
        "--vmin",
        type=float,
        help="Fixed log10 cgs colour minimum, shared by selected fields.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Fixed log10 cgs colour maximum, shared by selected fields.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.2, help="Seconds per GIF frame."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/processed/Movies/Gifs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Compatibility alias: reuse existing outputs (already the default).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute all PNGs and replace GIFs."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    snapshots = find_snapshots(args.base_dir, args.start, args.end)[:: args.step]
    if not snapshots:
        parser.error("No snapshots found in the requested range.")
    fields = args.field or ["dissipation"]
    kind = "proj" if args.plot_kind == "projection" else "slice"
    if args.dry_run:
        for path in snapshots:
            print(path)
        for field in fields:
            print(f"Output: {args.out / f'{field}_{kind}_{args.plane}.gif'}")
        return 0

    radius = args.r_star * (args.m_bh / args.m_star) ** (2 / 3)
    bounds = args.box or [value * radius for value in (-6, -4, -2, 2.5, 3, 2)]
    box = u.unyt_array(bounds, richio.units.lscale)
    reference = None
    for field in fields:
        name = f"{field}_{kind}_{args.plane}"
        gif = args.out / f"{name}.gif"
        if gif.exists() and not args.overwrite:
            LOG.info("Skipping existing %s; use --overwrite to replace", gif)
            continue
        if reference is None:
            reference = richio.load(snapshots[-1])
        data, _, _ = sample_map(reference, field, args, box)
        positive = np.asarray(data)[np.isfinite(data) & (data > 0)]
        lower = (
            args.vmin
            if args.vmin is not None
            else math.floor(np.log10(positive.min()) * 2) / 2
        )
        upper = (
            args.vmax
            if args.vmax is not None
            else math.ceil(np.log10(positive.max()) * 2) / 2
        )
        LOG.info("%s fixed log10 range: %g .. %g", field, lower, upper)
        images = []
        for path in snapshots:
            snapshot = richio.load(path)
            number = snapshot.snapnum
            outpath = args.out / name / f"{name}_snap_{number:04d}.png"
            if not outpath.exists() or args.overwrite:
                data, xspace, yspace = sample_map(snapshot, field, args, box)
                fig, ax = plt.subplots(figsize=(6, 5))
                richio.plots.scalar_map(
                    data,
                    xspace,
                    yspace,
                    ax=ax,
                    cmap=args.cmap,
                    label_latex=field,
                    vmin=lower,
                    vmax=upper,
                )
                ax.set_xlabel(f"{args.plane[0]} [{xspace.units}]")
                ax.set_ylabel(f"{args.plane[1]} [{yspace.units}]")
                outpath.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(outpath, dpi=150, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved %s (input %s)", outpath, path)
            images.append(imageio.imread(outpath))
        imageio.mimsave(
            gif, images, format="GIF", duration=1000 * args.duration, loop=0
        )
        LOG.info("Wrote %s", gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
