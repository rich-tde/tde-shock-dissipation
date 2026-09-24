r"""Compare linear and sinh line-of-sight sampling for one density projection.

All maps share linear X/Y sampling. A fine linear z reference is compared with
candidate linear/sinh grids, measuring dimensionless map errors and projection
runtime. Relative pixel percentiles use reference magnitudes greater than
1e-8 of the largest reference magnitude. This measures snapshot/grid accuracy,
not hydrodynamic convergence or a universal best spacing.

Input files
-----------
``--snapshot``
    HDF5 file or extracted NPY directory with X/Y/Z, density and box fields.
    Default: ``/data1/projects/pi-rossiem/TDE_data/NewSnellius/`` followed by
    ``R0.47M0.5BH10000beta1S60ComptonHiRes/snap_0.h5``. Coordinates are used
    as stored, with no BH-frame correction. ``--res-xy`` is the number of
    samples per transverse axis; ``--reference-nz`` defaults to 4096.
``--res-z``, ``--scale``
    Repeatable candidate z resolutions and sinh central scales. Defaults:
    64/128/256/512/1024 samples and 0.05/0.1/0.2/0.5/1.0 code lengths
    (solar-radius scale). One linear case and every scale are run per resolution.

Output files
------------
``<output-dir>/convergence.csv``
    UTF-8 comma-separated table with a header and shape ``(N, 8)`` when loaded
    as a table; ``N = len(res_z) * (1 + len(scale))``. Rows are grouped by
    resolution, with linear first and then sinh scales in requested order.
    Zero-based columns (also CSV names) are:

    0 ``spacing``
        String ``linear`` or ``sinh``.
    1 ``scale``
        Floating-point sinh central scale in code lengths; ``nan`` for linear.
    2 ``res_z``
        Integer number of z samples, not the number of integration intervals.
    3 ``seconds``
        Floating-point projection wall seconds, excluding initial field load.
    4 ``normalized_l1``
        Float ``sum(abs(candidate-reference))/sum(abs(reference))``.
    5 ``median_relative``, 6 ``p95_relative``
        Floating-point median and 95th percentile of
        ``abs(candidate-reference)/abs(reference)`` on significant pixels.
    7 ``total_relative``
        Float ``abs(sum(candidate)-sum(reference))/abs(sum(reference))``;
        this is a map-sum error, not a separately area-integrated mass error.

    Error values are dimensionless and linear, not logs. CSV contains no
    snapshot path, reference resolution or X/Y resolution; retain the command.
``<output-dir>/convergence.png``
    Two-panel figure: normalized L1 versus z resolution, and log10 relative
    error for the best L1 candidate. Default canvas is 1800 by 810 pixels.
    PIL conversion to RGBA gives ``uint8 (810, 1800, 4)`` (row, column, RGBA).
    The error-map panel displays x horizontally and y vertically. Numerical
    maps are ``(res_xy-1, res_xy-1)`` in x/y order internally but are not saved.

Default output directory is
``data/processed/RichioDevelopment/sinh-projection-convergence`` under the
repository. Stdout reports reference timing and the best candidate row.

Usage
-----
Run from ``/home/hey4/rich_tde`` (replace the input path)::

    python works/richio-development/sinh-projection-convergence.py \
        --snapshot /path/to/snap_21.h5 --res-xy 64 --reference-nz 512 \
        --res-z 64 --res-z 128 --scale 0.1 --scale 0.5 \
        --output-dir data/processed/RichioDevelopment/snap21-convergence

A completed CSV/PNG pair is skipped. An incomplete pair is recomputed;
``--overwrite`` recomputes and replaces both. Use separate directories for
separate snapshots or sweeps.

Loading examples
----------------
Load the mixed-type CSV as a structured NumPy array and select the best row::

    import numpy as np
    from pathlib import Path
    from PIL import Image

    root = Path("data/processed/RichioDevelopment/snap21-convergence")
    rows = np.atleast_1d(np.genfromtxt(
        root / "convergence.csv", delimiter=",", names=True,
        dtype=None, encoding="utf-8",
    ))  # shape (N,), named fields; not an ordinary numeric (N, 8) array
    best = rows[np.nanargmin(rows["normalized_l1"])]
    print(best["spacing"], best["scale"], best["res_z"], best["normalized_l1"])
    with Image.open(root / "convergence.png") as image:
        rgba = np.array(image.convert("RGBA"))  # uint8 (810, 1800, 4)
"""

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import dev  # noqa: F401  # isort: skip  # Configure style before pyplot.
import matplotlib.pyplot as plt
import numpy as np
import typer

import richio

DEFAULT_SNAPSHOT = Path(
    "/data1/projects/pi-rossiem/TDE_data/NewSnellius/"
    "R0.47M0.5BH10000beta1S60ComptonHiRes/snap_0.h5"
)
DEFAULT_OUTPUT = Path(
    "/home/hey4/rich_tde/data/processed/RichioDevelopment/sinh-projection-convergence"
)
RESOLUTIONS = (64, 128, 256, 512, 1024)
SCALES = (0.05, 0.1, 0.2, 0.5, 1.0)


def error_metrics(candidate, reference):
    """Return robust map errors relative to non-negligible reference pixels."""
    candidate = np.asarray(candidate, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    difference = np.abs(candidate - reference)
    significant = np.abs(reference) > 1e-8 * np.nanmax(np.abs(reference))
    relative = difference[significant] / np.abs(reference[significant])
    return {
        "normalized_l1": float(np.nansum(difference) / np.nansum(np.abs(reference))),
        "median_relative": float(np.nanmedian(relative)),
        "p95_relative": float(np.nanpercentile(relative, 95)),
        "total_relative": float(
            abs(np.nansum(candidate) - np.nansum(reference)) / abs(np.nansum(reference))
        ),
    }


def project(
    snapshot, arrays, resolution_xy, resolution_z, workers, spacing, scale=None
):
    """Time one density projection using preloaded cell arrays."""
    x, y, z, density = arrays
    started = perf_counter()
    projected, _, _ = snapshot.project(
        density,
        res=(resolution_xy, resolution_xy, resolution_z),
        X=x,
        Y=y,
        Z=z,
        workers=workers,
        spacing=("linear", "linear", spacing),
        sinh_scale=scale,
    )
    return projected, perf_counter() - started


def main(
    snapshot: Path = typer.Option(  # noqa: B008 - Typer declares options in defaults.
        DEFAULT_SNAPSHOT, exists=True, readable=True
    ),
    output_dir: Path = typer.Option(  # noqa: B008 - Typer declares options in defaults.
        DEFAULT_OUTPUT
    ),
    res_xy: int = typer.Option(128, min=2),
    reference_nz: int = typer.Option(4096, min=2),
    workers: int = typer.Option(16),
    res_z: list[int] | None = typer.Option(
        None,
        help="Candidate z samples; repeat this option (default: 64 128 256 512 1024).",
    ),
    scale: list[float] | None = typer.Option(
        None,
        help="Sinh central scales in RICH code lengths; repeat (default: 0.05 0.1 0.2 0.5 1.0).",
    ),
    overwrite: bool = typer.Option(False, help="Replace existing CSV and figure."),
):
    """Run the convergence comparison and write its table and figure."""
    resolutions = res_z or RESOLUTIONS
    scales = scale or SCALES
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "convergence.csv"
    figure_path = output_dir / "convergence.png"
    if not overwrite and csv_path.exists() and figure_path.exists():
        typer.echo(
            f"Skipping existing products in {output_dir}; use --overwrite to replace."
        )
        return

    loaded_snapshot = richio.load(str(snapshot))
    arrays = (
        loaded_snapshot.X,
        loaded_snapshot.Y,
        loaded_snapshot.Z,
        loaded_snapshot.density,
    )
    reference, reference_seconds = project(
        loaded_snapshot, arrays, res_xy, reference_nz, workers, "linear"
    )

    rows = []
    maps = {}
    for resolution_z in resolutions:
        linear, seconds = project(
            loaded_snapshot, arrays, res_xy, resolution_z, workers, "linear"
        )
        rows.append(
            {
                "spacing": "linear",
                "scale": np.nan,
                "res_z": resolution_z,
                "seconds": seconds,
                **error_metrics(linear, reference),
            }
        )
        maps[("linear", resolution_z, None)] = linear

        for scale in scales:
            sinh, seconds = project(
                loaded_snapshot,
                arrays,
                res_xy,
                resolution_z,
                workers,
                "sinh",
                scale * richio.units.lscale,
            )
            rows.append(
                {
                    "spacing": "sinh",
                    "scale": scale,
                    "res_z": resolution_z,
                    "seconds": seconds,
                    **error_metrics(sinh, reference),
                }
            )
            maps[("sinh", resolution_z, scale)] = sinh

    columns = tuple(rows[0])
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        for row in rows:
            stream.write(",".join(str(row[column]) for column in columns) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    linear_rows = [row for row in rows if row["spacing"] == "linear"]
    axes[0].loglog(
        [row["res_z"] for row in linear_rows],
        [row["normalized_l1"] for row in linear_rows],
        "o-",
        color="0.35",
        label="linear",
    )
    for scale in scales:
        selected = [
            row for row in rows if row["spacing"] == "sinh" and row["scale"] == scale
        ]
        axes[0].loglog(
            [row["res_z"] for row in selected],
            [row["normalized_l1"] for row in selected],
            "o-",
            label=rf"sinh $z_0={scale:g}$",
        )
    axes[0].set_xlabel(r"line-of-sight samples $n_z$")
    axes[0].set_ylabel("normalized L1 error")
    axes[0].grid(which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    best = min(rows, key=lambda row: row["normalized_l1"])
    best_key = (
        best["spacing"],
        best["res_z"],
        best["scale"] if best["spacing"] == "sinh" else None,
    )
    best_map = maps[best_key]
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(np.asarray(best_map - reference) / np.asarray(reference))
    image = axes[1].imshow(
        np.log10(np.clip(relative, 1e-12, None)).T,
        origin="lower",
        cmap="magma",
        vmin=-6,
        vmax=0,
    )
    axes[1].set_title(
        f"best: {best['spacing']}, nz={best['res_z']}, "
        f"z0={best['scale'] if best['spacing'] == 'sinh' else '-'}"
    )
    axes[1].set_xlabel("x pixel")
    axes[1].set_ylabel("y pixel")
    fig.colorbar(image, ax=axes[1], label=r"$\log_{10}$ relative error")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    typer.echo(f"Wrote {csv_path}")
    typer.echo(f"Wrote {figure_path}")
    typer.echo(
        f"Reference: {snapshot}, linear nz={reference_nz}, {reference_seconds:.1f} s"
    )
    typer.echo(f"Best candidate: {best}")


if __name__ == "__main__":
    typer.run(main)
