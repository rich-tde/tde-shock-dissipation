"""Detect shock surfaces in selected TDE snapshots and save reusable cell masks.

Find compressive shock candidates, build their k-nearest-neighbour stencils,
apply shock-zone jump criteria, then identify shock-surface cells and their
upstream/downstream neighbours. Defaults are ``gamma=5/3``, 48 neighbours
and minimum zone Mach 1.3; inferred jump Mach numbers depend on these
assumptions and on resolution. This script does not integrate dissipation
or make figures. Indices always refer to the original snapshot cell order.

Input files
-----------
Raw ``snap_full_N.h5`` or ``snap_N.h5`` files resolved by
``dev.datapaths.DATAPATHS`` and ``SNAPSHOT_TFB``. Exact directories/restart
exclusions are in ``dev/dev/datapaths.py``; ``--list-only`` prints paths.
``richio.load`` supplies positions, density, pressure, velocity divergence,
and thermodynamic gradients required by ``richio.shockfinder``.
Modes ``1/2/3`` select ``1e4/1e5/1e6`` solar-mass BH models. Default
``t/t_fb`` samples are 1e4: ``0.5, 1, 1.5, 2``; 1e5: ``0.3, 0.5``;
1e6: ``1, 1.2, 1.4, 1.5``, plus each run's final snapshot.

Output files
------------
``data/processed/ShockFinderEdissSelection/RUN/shockfinder_snap_NNNN.npz``
    Compressed NumPy archive; the absolute default root is under
    ``/home/hey4/rich_tde`` and can be set with ``--output-root``. ``NNNN``
    is the zero-padded snapshot number. Load with ``np.load``; scalar
    metadata has shape ``()`` and is extracted with ``.item()``.
    Let ``N=n_cells`` and ``M=len(surf_idx)``. This is a collection of named
    arrays, not an ``(N, k)`` table. No cell coordinates or raw fields are
    duplicated in this archive; ``snap_path`` identifies their source.

    ``run``, ``snap_path`` : Unicode, shape ``()``
        Run label (``1e4``, ``1e5``, ``1e6``) and raw HDF5 snapshot path.
    ``snapnum`` : int64, shape ``()``
        Snapshot number, not a zero-based position in the run catalogue.
    ``requested_tfb``, ``time_tfb``, ``time_code`` : float64, shape ``()``
        Requested time/fallback time, actual time/fallback time, and actual
        time in code units (one code-time unit is 1603 s), respectively.
        ``requested_tfb`` is NaN for explicit snapshot-number and last-
        snapshot selections; it can differ from actual ``time_tfb``.
    ``is_last`` : bool, shape ``()``
        Whether this entry was selected as the final snapshot.
    ``n_cells``, ``n_candidates``, ``k_neighbours`` : int64, shape ``()``
        Total snapshot cells, compressive candidates and neighbour count.
    ``gamma``, ``mach_min`` : float64, shape ``()``
        Dimensionless assumed adiabatic index and shock-zone Mach cutoff.
    ``elapsed_seconds`` : float64, shape ``()``
        Wall time for loading and detection, in seconds.
    ``shock_zone``, ``surface_mask``, ``pre_mask``, ``post_mask`` : bool, shape ``(N,)``
        True for zone, surface, upstream and downstream cells, respectively.
        Pre/post masks are deduplicated sets; their number of True values
        can be smaller than ``M`` when several shocks share a cell.
    ``surf_idx``, ``pre_idx``, ``post_idx`` : int64, shape ``(M,)``
        Zero-based original snapshot indices, aligned per surface cell.
        Row ``j`` is the triplet surface/upstream/downstream of one shock;
        ``surf_idx`` equals ``np.flatnonzero(surface_mask)``. Upstream and
        downstream indices may repeat. Invalid ray traces are omitted,
        rather than represented with padded indices or NaNs.
    ``mach_T``, ``mach_P``, ``mach_rho`` : float64, shape ``(M,)``
        Linear dimensionless Mach numbers from temperature, pressure and
        density jumps for those same triplets. The detector uses zero for
        ``mach_rho`` when its inverse density-jump denominator is nonpositive;
        this is not evidence of a physical zero-Mach shock. No logarithms
        are stored and no additional finite-value mask is applied on save.

Usage
-----
Run from ``/home/hey4/rich_tde`` with the richanalysis Python environment::

    python works/shock-tde/shock-finder-ediss-selection.py --list-only
    python works/shock-tde/shock-finder-ediss-selection.py --task-index 0 --workers 8
    python works/shock-tde/shock-finder-ediss-selection.py --mode 1 --snapshot 77
    python works/shock-tde/shock-finder-ediss-selection.py --mode 2 --tfb 0.4

Each invocation processes one snapshot. ``--task-index`` is the zero-based
printed selection index and supports Slurm arrays. Repeated ``--snapshot``
(requires ``--mode``) or ``--tfb`` replaces default samples; ``--include-last``
adds the final snapshot. Existing NPZs are skipped without checking detector
settings: use ``--overwrite`` after changing parameters or a distinct output
root for comparisons. Files are written atomically beside their destination.

Loading examples
----------------
Find the original snapshot indices of surface cells with thermal Mach >= 2,
without loading the potentially much larger full-snapshot masks::

    from pathlib import Path
    import numpy as np

    root = Path("data/processed/ShockFinderEdissSelection/1e4")
    with np.load(root / "shockfinder_snap_0077.npz") as data:
        surface_index = data["surf_idx"]
        mach = data["mach_T"]
        snapshot_path = data["snap_path"].item()
        time_tfb = data["time_tfb"].item()
        upstream = data["pre_idx"]
        downstream = data["post_idx"]
    keep = np.isfinite(mach) & (mach >= 2.0)
    strong_surface_index = surface_index[keep]
    print(snapshot_path, time_tfb, strong_surface_index.size)
    triplets = np.column_stack((surface_index, upstream, downstream))
    # triplets.shape == (M, 3); columns 0=surface, 1=upstream, 2=downstream.
    # After richio.load(snapshot_path), snap.density[strong_surface_index]
    # selects the corresponding physical cells in the original snapshot.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import numpy as np
import typer
from dev.datapaths import DATAPATHS, SNAPSHOT_TFB, TDE_PARAMETERS
from loguru import logger

import richio
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


def selection(
    mode: int | None = None,
    snapshots: list[int] | None = None,
    tfbs: list[float] | None = None,
    include_last: bool = False,
) -> list[dict]:
    """Resolve the notebook's requested fallback times plus final snapshots."""

    selected = []
    runs = list(REQUESTED_TFBS) if mode is None else [list(REQUESTED_TFBS)[mode - 1]]
    use_defaults = not snapshots and not tfbs and not include_last
    for run in runs:
        requested_tfbs = REQUESTED_TFBS[run] if use_defaults else (tfbs or [])
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
        available = dict(zip(snapnums, paths))
        for number in snapshots or []:
            selected.append(
                {
                    "run": run,
                    "requested_tfb": np.nan,
                    "is_last": False,
                    "snapnum": number,
                    "path": available[number],
                }
            )
        if use_defaults or include_last:
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
    task_index: int | None = typer.Option(
        None,
        min=0,
        help="Zero-based selection index; optional when one snapshot is selected.",
    ),
    mode: int | None = typer.Option(
        None, min=1, max=3, help="Restrict selection to 1: 1e4, 2: 1e5, 3: 1e6."
    ),
    snapshot: list[int] | None = typer.Option(
        None, help="Snapshot number; requires --mode; repeat for several."
    ),
    tfb: list[float] | None = typer.Option(
        None, help="Nearest t/t_fb; repeat for several; replaces default times."
    ),
    include_last: bool = typer.Option(
        False, help="Include final snapshot; alone selects only final snapshots."
    ),
    output_root: Path = typer.Option(
        OUTPUT_ROOT, help="Write RUN/shockfinder_snap_NNNN.npz here."
    ),
    k_neighbours: int = typer.Option(
        K_NEIGHBOURS, min=1, help="Number of nearest neighbours in shock detection."
    ),
    gamma: float = typer.Option(
        GAMMA, min=1.000001, help="Assumed adiabatic index for jump conditions."
    ),
    mach_min: float = typer.Option(
        MACH_MIN, min=1, help="Minimum shock-zone Mach number."
    ),
    list_only: bool = typer.Option(
        False,
        help="List input/output paths without loading snapshot fields.",
    ),
    workers: int = typer.Option(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        min=1,
        help="Workers used by the k-nearest-neighbour query.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing result."),
):
    """Detect shocks in one selected snapshot and save masks and jump Mach numbers."""
    if snapshot and mode is None:
        raise typer.BadParameter("Use --mode with --snapshot to identify the run.")
    selected = selection(mode, snapshot, tfb, include_last)
    if list_only:
        for index, item in enumerate(selected):
            output = (
                output_root
                / item["run"]
                / f"shockfinder_snap_{item['snapnum']:04d}.npz"
            )
            print(
                f"[{index}] {item['run']} snap {item['snapnum']}: "
                f"{item['path']} -> {output}"
            )
        return
    if task_index is None:
        if len(selected) != 1:
            raise typer.BadParameter(
                "Choose --task-index; use --list-only to inspect the selection."
            )
        task_index = 0
    if task_index >= len(selected):
        raise typer.BadParameter(
            f"--task-index must be between 0 and {len(selected) - 1}"
        )
    item = selected[task_index]
    run = item["run"]
    snapnum = item["snapnum"]
    snap_path = item["path"]

    output_dir = output_root / run
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

    candidates = sf.shock_candidates(snap, gamma=gamma)
    logger.info(
        f"Candidates: {len(candidates.candidates):,}/{len(snap):,} "
        f"({len(candidates.candidates) / len(snap):.1%})"
    )
    vor = sf.build_knn(
        snap,
        cells=candidates.candidates,
        k=k_neighbours,
        workers=workers,
    )
    shock_zone = sf.find_shock_zone(snap, vor, gamma=gamma, mach_min=mach_min)
    result = sf.find_shock_surface(snap, vor, shock_zone, gamma=gamma)

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
            k_neighbours=k_neighbours,
            gamma=gamma,
            mach_min=mach_min,
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
