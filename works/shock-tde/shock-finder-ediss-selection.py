"""Run the shock finder for the t/t_fb samples in 0.1-plot-Ediss-distribution.

The selection contains the requested fallback times and final snapshot for each
of the 1e4, 1e5, and 1e6 solar-mass black-hole runs.  One Slurm array task
creates one restartable compressed result.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import typer
from loguru import logger

import richio
from dev.datapaths import DATAPATHS, SNAPSHOT_TFB, TDE_PARAMETERS
from richio import shockfinder as sf


app = typer.Typer(add_completion=False)

OUTPUT_ROOT = Path("/home/hey4/rich_tde/data/processed/ShockFinderEdissSelection")
REQUESTED_TFBS = {
    "1e4": (0.5, 1.0, 1.5, 2.0),
    "1e5": (0.3, 0.5),
    "1e6": (1.0, 1.2, 1.4, 1.5),
}
K_NEIGHBOURS = 48
GAMMA = 5 / 3
MACH_MIN = 1.3


def fallback_time_code(run: str) -> float:
    """Fallback time in RICH code units (G=1)."""

    mbh, mstar, rstar = TDE_PARAMETERS[run]
    return float(np.pi / np.sqrt(2) * np.sqrt(rstar**3 / mstar) * np.sqrt(mbh / mstar))


def selection() -> list[dict]:
    """Resolve the notebook's requested fallback times plus final snapshots."""

    selected = []
    for run, requested_tfbs in REQUESTED_TFBS.items():
        for requested_tfb in requested_tfbs:
            snapnum, path = SNAPSHOT_TFB(run, requested_tfb)
            selected.append(
                {
                    "run": run,
                    "requested_tfb": requested_tfb,
                    "is_last": False,
                    "snapnum": snapnum,
                    "path": path,
                }
            )
        snapnums, paths = DATAPATHS(run)
        selected.append(
            {
                "run": run,
                "requested_tfb": np.nan,
                "is_last": True,
                "snapnum": snapnums[-1],
                "path": paths[-1],
            }
        )
    return selected


def snapshot_time(snap) -> float:
    """Return the simulation time in code units for scalar or length-one data."""

    return float(np.asarray(snap.time.to_value("code_time")).squeeze())


@app.command()
def main(
    task_index: int = typer.Option(
        ..., min=0, help="Zero-based position in the 13-snapshot selection."
    ),
    workers: int = typer.Option(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        min=1,
        help="Workers used by the k-nearest-neighbour query.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing result."),
):
    selected = selection()
    if task_index >= len(selected):
        raise typer.BadParameter(
            f"--task-index must be between 0 and {len(selected) - 1}"
        )
    item = selected[task_index]
    run = item["run"]
    snapnum = item["snapnum"]
    snap_path = item["path"]

    output_dir = OUTPUT_ROOT / run
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shockfinder_snap_{snapnum:04d}.npz"
    if output_path.exists() and not overwrite:
        logger.info(f"Output exists; skipping {output_path}")
        return

    logger.info(f"Loading {run} snapshot {snapnum}: {snap_path}")
    started = time.perf_counter()
    snap = richio.load(snap_path)
    time_code = snapshot_time(snap)
    time_tfb = time_code / fallback_time_code(run)

    candidates = sf.shock_candidates(snap, gamma=GAMMA)
    logger.info(
        f"Candidates: {len(candidates.candidates):,}/{len(snap):,} "
        f"({len(candidates.candidates) / len(snap):.1%})"
    )
    vor = sf.build_knn(
        snap,
        cells=candidates.candidates,
        k=K_NEIGHBOURS,
        workers=workers,
    )
    shock_zone = sf.find_shock_zone(snap, vor, gamma=GAMMA, mach_min=MACH_MIN)
    result = sf.find_shock_surface(snap, vor, shock_zone, gamma=GAMMA)

    elapsed = time.perf_counter() - started
    logger.info(
        f"Shock zone: {shock_zone.sum():,}; surface: {len(result.surf_idx):,}; "
        f"elapsed: {elapsed:.1f} s"
    )

    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{output_path.stem}.", suffix=".npz", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path,
            run=run,
            snap_path=str(snap_path),
            snapnum=snapnum,
            requested_tfb=item["requested_tfb"],
            is_last=item["is_last"],
            time_code=time_code,
            time_tfb=time_tfb,
            n_cells=len(snap),
            n_candidates=len(candidates.candidates),
            k_neighbours=K_NEIGHBOURS,
            gamma=GAMMA,
            mach_min=MACH_MIN,
            elapsed_seconds=elapsed,
            shock_zone=shock_zone,
            surface_mask=result.surface_mask,
            pre_mask=result.pre_mask,
            post_mask=result.post_mask,
            surf_idx=result.surf_idx,
            pre_idx=result.pre_idx,
            post_idx=result.post_idx,
            mach_T=result.mach_T,
            mach_P=result.mach_P,
            mach_rho=result.mach_rho,
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    app()
