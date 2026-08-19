#!/usr/bin/env python3
"""
Make GIFs from a series of snapshots using richio plotting helpers.

Features:
- Iterate snapshot folders named `snap_<n>` in a numeric range (skips missing).
- For each configured plot (projection or slice), produce per-snapshot PNGs and a GIF.
- Colorbar range (vmin/vmax) is computed from the latest available snapshot per plot.
- Configurable `PLOTS` list at top makes adding/removing plots trivial.
- Adds a title with the snapshot time when available.

Edit the PLOTS list below to add/remove plot types/parameters.

Examples:
python make_gif.py /data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/ --start 1 --end 151
"""

from pathlib import Path
import argparse
import logging
import math

import matplotlib
matplotlib.use('Agg')
import dev  # noqa: F401  # Configure local imports and the plotting style.
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import unyt as u

import richio

LOG = logging.getLogger("make_gif")

# ---------------------- USER-CONFIGURABLE PLOTS ----------------------
# Each entry should be a dict with at least:
# - name: short name used for output files
# - type: 'projection' or 'slice'
# - data: data key or unyt array (e.g., 'dissipation')
# - kwargs: additional kwargs forwarded to snap.project or snap.slice
# Example: add/remove entries from this list to change outputs.
beta = 1
mstar = .5 * richio.units.mscale
rstar = .47 * richio.units.lscale
mbh = 10**4 * richio.units.mscale
rt = rstar * (mbh/mstar)**(1/3)

ra = rt**2 / rstar #2 * Rt * (Mbh/mstar)**(1/3)

nozzle_box = u.unyt_array([-3*rt, -3*rt, -2*rt, 3*rt, 3*rt, 2*rt])
big_box = u.unyt_array([-6*ra, -4*ra, -2*ra, 2.5*ra, 3*ra, 2*ra])
mid_box = u.unyt_array([-3*ra, -2*ra, -0.8*ra, 2*ra, 2*ra, 0.8*ra])

PLOTS = [
    {
        'name': 'dissipation_proj_yz',
        'type': 'projection',
        'data': 'dissipation',
        'kwargs': {
            'res': (512, 512, 512),
            'x': 'CMy',
            'y': 'CMz',
            'z': 'CMx',
            'box_size': big_box,
        },
        'plot_kwargs': {
            'cmap': 'cividis',
            'label_latex': r'\text{Energy Dissipation}',
            'unit_latex': r'\mathrm{(erg/s/cm^2)}',
        }
    },
]
# ---------------------------------------------------------------------


def find_available_snapshots(base: Path, start: int, end: int):
    """Return sorted list of (idx, path) for existing snap_<idx> folders."""
    snaps = []
    for i in range(start, end + 1):
        p = base / f"snap_{i}"
        if p.exists():
            snaps.append((i, p))
    return snaps


def load_snapshot(path: Path):
    LOG.info("Loading %s", path)
    return richio.load(str(path))


def compute_vrange_from_snapshot(snap, plot_cfg):
    """Compute vmin/vmax (in log10 space) for a given snapshot and plot config.

    Rounds outward to nearest half-integers (floor/ceil on *2)/2.
    Returns tuple (vmin, vmax) or (None, None) on failure.
    """
    try:
        if plot_cfg['type'] == 'projection':
            projected, xspace, yspace = snap.project(data=plot_cfg['data'], **plot_cfg['kwargs'])
            arr = np.asarray(np.log10(projected))
        else:
            sliced, xspace, yspace = snap.slice(data=plot_cfg['data'], **plot_cfg['kwargs'])
            arr = np.asarray(np.log10(sliced))

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            LOG.warning("No finite values found when computing vmin/vmax for %s", plot_cfg['name'])
            return None, None

        dmin = float(np.min(arr[finite_mask]))
        dmax = float(np.max(arr[finite_mask]))

        vmin = math.floor(dmin * 2.0) / 2.0
        vmax = math.ceil(dmax * 2.0) / 2.0
        return vmin, vmax
    except Exception:
        LOG.exception("Failed to compute vmin/vmax for config %s", plot_cfg['name'])
        return None, None


def title_from_snapshot(snap):
    # try a few common attributes for snapshot time
    for attr in ('tfb', 'time', 't'):
        if hasattr(snap, attr):
            return f"t = {getattr(snap, attr):.4f}"
    # fallback to nothing
    return ""


def make_plot_for_snapshot(snap, idx, plot_cfg, vrange, outpath: Path):
    """Create and save a plot for a single snapshot for one plot_cfg.

    vrange is (vmin, vmax) in log10 units (or None values to let scalar_map choose).
    """
    try:
        if plot_cfg['type'] == 'projection':
            data_arr, xspace, yspace = snap.project(data=plot_cfg['data'], **plot_cfg['kwargs'])
        else:
            data_arr, xspace, yspace = snap.slice(data=plot_cfg['data'], **plot_cfg['kwargs'])

        # build and save the figure using richio plotting helper
        fig, ax = plt.subplots(figsize=(6, 5))

        # prepare kwargs to pass to scalar_map
        plot_args = dict(
            f=data_arr,
            xspace=xspace,
            yspace=yspace,
            ax=ax,
            **plot_cfg.get('plot_kwargs', {}),
        )

        if vrange is not None and vrange[0] is not None and vrange[1] is not None:
            plot_args.update({'vmin': vrange[0], 'vmax': vrange[1]})

        # call scalar_map which handles the colorbar
        ax, im = richio.plots.scalar_map(**plot_args)

        # add title/time info
        title = title_from_snapshot(snap)
        if title:
            ax.set_title(f"{plot_cfg['name']} — {title}")
        else:
            ax.set_title(f"{plot_cfg['name']} — snap {idx}")

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        LOG.info("Saved %s", outpath)
        return True
    except Exception:
        LOG.exception("Failed to make plot for snap %s, config %s", idx, plot_cfg['name'])
        return False


def build_gif(image_paths, gif_path, duration=0.2):
    imgs = []
    for p in image_paths:
        imgs.append(imageio.imread(str(p)))
    imageio.mimsave(str(gif_path), imgs, duration=duration)
    LOG.info("Wrote GIF %s", gif_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make GIFs from snapshots using richio")
    parser.add_argument('base_dir', help='Base directory containing snap_N folders')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=150)
    parser.add_argument('--out', default='out_plots')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip creating outputs that already exist and continue')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    base = Path(args.base_dir)
    outdir = Path(args.out)

    snaps = find_available_snapshots(base, args.start, args.end)
    if not snaps:
        LOG.error("No snapshots found in %s in range %d..%d", base, args.start, args.end)
        return 1

    # pick latest available snapshot (highest index) to determine vrange per plot
    latest_idx, latest_path = max(snaps, key=lambda x: x[0])
    LOG.info("Using latest snapshot %s to compute color ranges", latest_path)

    latest_snap = load_snapshot(latest_path)

    vranges = {}
    for cfg in PLOTS:
        vranges[cfg['name']] = compute_vrange_from_snapshot(latest_snap, cfg)

    if args.dry_run:
        LOG.info("Dry run: vranges = %s", vranges)
        return 0

    # For each plot config, produce images then a gif
    for cfg in PLOTS:
        name = cfg['name']
        img_paths = []
        cfg_out = outdir / name
        for idx, path in snaps:
            outpath = cfg_out / f"{name}_snap_{idx:04d}.png"
            # if resuming and image already exists, keep it and skip generation
            if args.resume and outpath.exists():
                LOG.info("Resuming: skipping existing image %s", outpath)
                img_paths.append(outpath)
                continue

            try:
                snap = load_snapshot(path)
                ok = make_plot_for_snapshot(snap, idx, cfg, vranges.get(name), outpath)
                if ok:
                    img_paths.append(outpath)
            except Exception:
                LOG.exception("Skipping snap %s for plot %s", path, name)

        if img_paths:
            gif_path = outdir / f"{name}.gif"
            if args.resume and gif_path.exists():
                LOG.info("Resuming: skipping existing GIF %s", gif_path)
            else:
                build_gif(img_paths, gif_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
