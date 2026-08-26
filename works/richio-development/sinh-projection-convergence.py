"""Compare linear and sinh line-of-sight grids for a TDE projection.

The reference uses a fine linear z grid. Candidate grids reuse the same x/y
resolution and compare several z resolutions and sinh central scales. Results
are saved as CSV plus a convergence figure; this script is intentionally not a
unit test because timings and convergence depend on the selected snapshot.
"""

from pathlib import Path
from time import perf_counter

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
    overwrite: bool = typer.Option(False, help="Replace existing CSV and figure."),
):
    """Run the convergence comparison and write its table and figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "convergence.csv"
    figure_path = output_dir / "convergence.png"
    if not overwrite and (csv_path.exists() or figure_path.exists()):
        raise FileExistsError("Outputs exist; pass --overwrite to replace them.")

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
    for resolution_z in RESOLUTIONS:
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

        for scale in SCALES:
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
    for scale in SCALES:
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
