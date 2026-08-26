#!/usr/bin/env python3
"""Render physical-field and diagnostic nozzle ``yz`` slices.

Both products use the same pericentre-scaled geometry. The geometry is the
original broad notebook box normalized by the 1e4-run pericentre and then
scaled by each run's own pericentre; plotted coordinates are shown in solar
radii. The physical-field product contains density, gas pressure, gas
temperature, and dissipation; density is overlaid with in-plane velocity
streamlines. The diagnostic product contains ``|v_z|/c_s``, cell entropy,
``|v_z|``, and sound speed.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rich-tde-nozzle-yz")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from scipy.integrate import solve_ivp

import richio
from dev import DATAPATHS, SNAPSHOT_TFB
from dev.datapaths import TDE_PARAMETERS
from richio.plots import scalar_map


REPO = Path("/home/hey4/rich_tde")
OUTPUT_ROOT = REPO / "data/processed/NozzleYZSlices"
REQUESTED_TFBS = {
    "1e4": (0.5, 1.0, 1.5, 2.0),
    "1e5": (0.3, 0.5),
    "1e6": (1.0, 1.2, 1.4, 1.5),
}

# Original broad notebook geometry for the 1e4 run, normalized by pericentre.
REFERENCE_RP_RSUN = 0.47 * (1.0e4 / 0.5) ** (1.0 / 3.0)
SLICE_X_RP = 13.0 / REFERENCE_RP_RSUN
YZ_WINDOW_RP = tuple(value / REFERENCE_RP_RSUN for value in (-10.0, 10.0, -5.0, 5.0))
MACH_PROXY_LIMITS = (0.0, 80.0)
PW_SCHWARZSCHILD_RSUN_1E6 = 4.21

BROAD_FIELDS = (
    ("density", "Density", r"$\log_{10}(\rho/[\mathrm{g\,cm^{-3}}])$", "twilight"),
    ("pressure", "Gas pressure", r"$\log_{10}(P/[\mathrm{dyn\,cm^{-2}}])$", "rainbow"),
    (
        "temperature",
        "Gas temperature",
        r"$\log_{10}(T_\mathrm{gas}/\mathrm{K})$",
        "inferno",
    ),
    (
        "dissipation",
        "Dissipation",
        r"$\log_{10}(\dot{e}_\mathrm{diss}/[\mathrm{erg\,s^{-1}\,cm^{-3}}])$",
        "viridis",
    ),
)
DIAGNOSTIC_FIELDS = (
    (
        "abs_vz_mach_proxy",
        r"Vertical Mach proxy $|v_z|/c_s$",
        r"$|v_z|/c_s$",
        "magma",
        False,
    ),
    (
        "cell_entropy",
        "Cell entropy",
        r"$\log_{10}(S_\mathrm{cell}/[\mathrm{erg\,K^{-1}}])$",
        "rainbow",
        True,
    ),
    (
        "abs_vz_kms",
        r"Vertical speed $|v_z|$",
        r"$\log_{10}(|v_z|/[\mathrm{km\,s^{-1}}])$",
        "plasma",
        True,
    ),
    (
        "sound_speed_kms",
        "Sound speed",
        r"$\log_{10}(c_s/[\mathrm{km\,s^{-1}}])$",
        "cividis",
        True,
    ),
)


@dataclass(frozen=True)
class RunConfig:
    mode: int
    run: str
    m_bh: float
    m_star: float
    r_star: float

    @property
    def r_p(self) -> float:
        return self.r_star * (self.m_bh / self.m_star) ** (1.0 / 3.0)

    @property
    def t_fb(self) -> float:
        return (
            math.pi
            / math.sqrt(2.0)
            * math.sqrt(self.r_star**3 / self.m_star)
            * math.sqrt(self.m_bh / self.m_star)
        )


RUNS = {
    mode: RunConfig(mode, run, *TDE_PARAMETERS[run])
    for mode, run in enumerate(("1e4", "1e5", "1e6"), start=1)
}


def mode_settings(mode: int) -> RunConfig:
    try:
        return RUNS[mode]
    except KeyError as exc:
        raise ValueError("mode must be 1 (1e4), 2 (1e5), or 3 (1e6)") from exc


def selected_snapshots(run: str) -> list[tuple[int, Path, bool]]:
    selected = [
        (*SNAPSHOT_TFB(run, requested_tfb), False)
        for requested_tfb in REQUESTED_TFBS[run]
    ]
    snapnums, paths = DATAPATHS(run)
    selected.append((snapnums[-1], paths[-1], True))
    return [(snapnum, Path(path), is_last) for snapnum, path, is_last in selected]


def needs_reference_frame(run: str, path: Path) -> bool:
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def scalar_time(snapshot):
    return snapshot.t.reshape(-1)[0] if getattr(snapshot.t, "ndim", 0) else snapshot.t


def positive_log10(values) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def sound_speed(density, pressure, internal_energy):
    """Return ``sqrt(gamma_eff P/rho)`` without assuming a fixed gamma."""
    gamma_eff = 1.0 + pressure / (density * internal_energy)
    return np.sqrt(gamma_eff * pressure / density)


def pw_orbit_derivatives(_, state, mass, schwarzschild_radius):
    """Paczynski-Wiita orbit equation used by the RICH TDE setup."""
    x, y, vx, vy = state
    radius = np.hypot(x, y)
    factor = -mass / (radius * (radius - schwarzschild_radius) ** 2)
    return vx, vy, factor * x, factor * y


def returning_orbit_grid(
    config: RunConfig,
    time_tfb: float,
    box_scale: float,
    resolution: int,
    radius_scale: float = 1.0,
    phase_offset_deg: float = 0.0,
    center_shift_rsun: float = 0.0,
    return_s_rsun: bool = False,
):
    """Return a uniform-phi grid along the returning PW orbit."""
    reference_pericenter = config.r_p
    pericenter = radius_scale * reference_pericenter
    schwarzschild_radius = PW_SCHWARZSCHILD_RSUN_1E6 * config.m_bh / 1e6
    energy_spread = (
        config.m_star * (config.m_bh / config.m_star) ** (1 / 3) / config.r_star
    )
    binding_energy = -energy_spread * time_tfb ** (-2 / 3)
    pericenter_speed = np.sqrt(
        2 * (config.m_bh / (pericenter - schwarzschild_radius) + binding_energy)
    )
    newtonian_period = 2 * np.pi * config.m_bh / (-2 * binding_energy) ** 1.5

    solution = solve_ivp(
        pw_orbit_derivatives,
        (0.0, 1.6 * newtonian_period),
        (pericenter, 0.0, 0.0, -pericenter_speed),
        args=(config.m_bh, schwarzschild_radius),
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=newtonian_period / 5000,
    )
    sample_time = np.linspace(0.0, solution.t[-1], 50000)
    x, y = solution.sol(sample_time)[:2]
    radius = np.hypot(x, y)
    minima = np.where((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:]))[0] + 1
    pericenter_index = minima[sample_time[minima] > 0.5 * newtonian_period][0]

    x_rp = x / reference_pericenter
    y_rp = y / reference_pericenter
    phi = np.unwrap(np.arctan2(y, x))
    phi -= phi[pericenter_index]
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_rp), np.diff(y_rp)))))
    half_length = box_scale * (YZ_WINDOW_RP[1] - YZ_WINDOW_RP[0]) / 2
    arc_limits = arc[pericenter_index] + np.array([-half_length, half_length])
    phi_limits = np.interp(arc_limits, arc, phi)
    phase_offset = np.deg2rad(phase_offset_deg)
    phi_grid = np.linspace(
        phi_limits[1] + phase_offset,
        phi_limits[0] + phase_offset,
        resolution,
        endpoint=False,
    )
    base_arc = np.interp(phi_grid, phi[::-1], arc[::-1])
    orbit_s_rsun = (
        -(base_arc - arc[pericenter_index]) * reference_pericenter + center_shift_rsun
    )
    shifted_arc = base_arc - center_shift_rsun / reference_pericenter
    phi_grid = np.interp(shifted_arc, arc, phi)
    orbit_x_rp = np.interp(phi_grid, phi[::-1], x_rp[::-1])
    orbit_y_rp = np.interp(phi_grid, phi[::-1], y_rp[::-1])
    if return_s_rsun:
        return phi_grid, orbit_x_rp, orbit_y_rp, orbit_s_rsun
    return phi_grid, orbit_x_rp, orbit_y_rp


def orbit_cache_complete(path: Path, resolution: int) -> bool:
    fields = [name for name, *_ in BROAD_FIELDS]
    axes = ["phi", "z_rsun", "orbit_x_rp", "orbit_y_rp"]
    flow = ["dphi_dt", "dz_rsun_dt"]
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return (
                all(name in data for name in fields + axes + flow + ["time_tfb"])
                and all(
                    data[name].shape == (resolution, resolution)
                    for name in fields + flow
                )
                and all(data[name].shape == (resolution,) for name in axes)
            )
    except (OSError, ValueError):
        return False


def cache_orbit_snapshot(
    path: Path,
    output: Path,
    config: RunConfig,
    resolution: int,
    workers: int,
    box_scale: float = 1.0,
    radius_scale: float = 1.0,
    phase_offset_deg: float = 0.0,
) -> None:
    """Cache a vertical slice whose midline follows the returning PW orbit."""
    snap = richio.load(str(path))
    time = scalar_time(snap)
    time_tfb = float(time / config.t_fb)
    x, y, z = snap.X, snap.Y, snap.Z
    if needs_reference_frame(config.run, path):
        offset = dev.reference_frame_offset(
            t=time,
            Mbh=config.m_bh * richio.units.mscale,
            Mstar=config.m_star * richio.units.mscale,
            Rstar=config.r_star * richio.units.lscale,
            beta=1,
        )
        x = x + offset[0]
        y = y + offset[1]

    r_sun = richio.units.lscale
    r_p = config.r_p * r_sun
    phi, orbit_x_rp, orbit_y_rp = returning_orbit_grid(
        config,
        time_tfb,
        box_scale,
        resolution,
        radius_scale=radius_scale,
        phase_offset_deg=phase_offset_deg,
    )
    zmin, zmax = YZ_WINDOW_RP[2:]
    z_rsun = np.linspace(
        box_scale * zmin * config.r_p,
        box_scale * zmax * config.r_p,
        resolution,
        endpoint=False,
    )
    orbit_x = orbit_x_rp * r_p
    orbit_y = orbit_y_rp * r_p
    grid_x, grid_z = np.meshgrid(orbit_x, z_rsun * r_sun, indexing="ij")
    grid_y = np.broadcast_to(orbit_y.to_value(r_sun)[:, None], grid_x.shape)
    query_points = u.unyt_array(
        np.stack([grid_x.to_value(r_sun), grid_y, grid_z.to_value(r_sun)], axis=-1),
        r_sun,
    )

    padding = 0.1 * r_p
    selection = (
        (x > orbit_x.min() - padding)
        & (x < orbit_x.max() + padding)
        & (y > orbit_y.min() - padding)
        & (y < orbit_y.max() + padding)
        & (z > z_rsun[0] * r_sun - padding)
        & (z < z_rsun[-1] * r_sun + padding)
    )
    indices = snap.nearest_indices(
        query_points,
        X=x,
        Y=y,
        Z=z,
        selection=selection,
        workers=workers,
    )

    arrays = {
        "time_tfb": np.asarray(time_tfb),
        "box_scale": np.asarray(float(box_scale)),
        "radius_scale": np.asarray(float(radius_scale)),
        "phase_offset_deg": np.asarray(float(phase_offset_deg)),
        "phi": np.asarray(phi, dtype="float64"),
        "z_rsun": np.asarray(z_rsun, dtype="float64"),
        "orbit_x_rp": np.asarray(orbit_x_rp, dtype="float64"),
        "orbit_y_rp": np.asarray(orbit_y_rp, dtype="float64"),
    }
    for name, *_ in BROAD_FIELDS:
        arrays[name] = positive_log10(getattr(snap, name)[indices].in_cgs())

    dx_dphi = np.gradient(orbit_x, phi)
    dy_dphi = np.gradient(orbit_y, phi)
    ds_dphi = np.hypot(dx_dphi, dy_dphi)
    tangent_x = dx_dphi / ds_dphi
    tangent_y = dy_dphi / ds_dphi
    tangent_speed = (
        snap.vx[indices] * tangent_x[:, None] + snap.vy[indices] * tangent_y[:, None]
    )
    arrays["dphi_dt"] = (tangent_speed / ds_dphi[:, None]).to_value("1/s")
    arrays["dz_rsun_dt"] = (snap.vz[indices] / r_sun).to_value("1/s")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)


def grid_indices(
    snapshot, x, y, z, r_p, resolution, workers, y_center=None, box_scale=1.0
):
    ymin, ymax, zmin, zmax = YZ_WINDOW_RP
    if y_center is None:
        y_center = 0.0 * r_p
    return snapshot.to_2dgrid(
        res=(resolution, resolution),
        X=x,
        Y=y,
        Z=z,
        plane="yz",
        slice_coord=SLICE_X_RP * r_p,
        box_size=(
            box_scale * ymin * r_p + y_center,
            box_scale * zmin * r_p,
            box_scale * ymax * r_p + y_center,
            box_scale * zmax * r_p,
        ),
        volume_selection=True,
        workers=workers,
    )


def cache_complete(path: Path, resolution: int) -> bool:
    grids = [name for name, *_ in BROAD_FIELDS]
    grids += ["vy_kms", "vz_stream_kms"]
    grids += [name for name, *_ in DIAGNOSTIC_FIELDS]
    axes = ["y_rsun", "z_rsun"]
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return (
                all(name in data for name in grids + axes + ["time_tfb"])
                and all(data[name].shape == (resolution, resolution) for name in grids)
                and all(data[name].shape == (resolution,) for name in axes)
            )
    except (OSError, ValueError):
        return False


def cache_snapshot(
    path: Path,
    output: Path,
    config: RunConfig,
    resolution: int,
    workers: int,
    y_center_rsun: float = 0.0,
    box_scale: float = 1.0,
) -> None:
    snap = richio.load(str(path))
    time = scalar_time(snap)
    x, y, z = snap.X, snap.Y, snap.Z
    if needs_reference_frame(config.run, path):
        offset = dev.reference_frame_offset(
            t=time,
            Mbh=config.m_bh * richio.units.mscale,
            Mstar=config.m_star * richio.units.mscale,
            Rstar=config.r_star * richio.units.lscale,
            beta=1,
        )
        x = x + offset[0]
        y = y + offset[1]

    r_sun = richio.units.lscale
    r_p = config.r_p * r_sun
    idx, grid_y, grid_z = grid_indices(
        snap,
        x,
        y,
        z,
        r_p,
        resolution,
        workers,
        y_center=y_center_rsun * r_sun,
        box_scale=box_scale,
    )

    arrays = {
        "time_tfb": np.asarray(float(time / config.t_fb)),
        "y_center_rsun": np.asarray(float(y_center_rsun)),
        "box_scale": np.asarray(float(box_scale)),
        "y_rsun": np.asarray(grid_y / r_sun, dtype="float64"),
        "z_rsun": np.asarray(grid_z / r_sun, dtype="float64"),
    }
    for name, *_ in BROAD_FIELDS:
        arrays[name] = positive_log10(getattr(snap, name)[idx].in_cgs())
    arrays["vy_kms"] = snap.vy[idx].to_value("km/s")
    arrays["vz_stream_kms"] = snap.vz[idx].to_value("km/s")

    rho = snap.density[idx].in_cgs()
    pressure = snap.pressure[idx].in_cgs()
    internal_energy = snap.internal_energy[idx].in_cgs()
    vz = snap.vz[idx].in_cgs()
    cs = sound_speed(rho, pressure, internal_energy)
    cell_entropy = (snap.entropy[idx] * snap.density[idx] * snap.volume[idx]).in_cgs()

    arrays["abs_vz_mach_proxy"] = np.asarray(np.abs(vz) / cs, dtype="float64")
    arrays["cell_entropy"] = positive_log10(cell_entropy)
    arrays["abs_vz_kms"] = positive_log10(np.abs(vz.to("km/s")))
    arrays["sound_speed_kms"] = positive_log10(cs.to("km/s"))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)


def logarithmic_limits(
    cache_paths: list[Path], fields
) -> dict[str, tuple[float, float]]:
    limits = {}
    for name, *_ in fields:
        finite_parts = []
        for path in cache_paths:
            with np.load(path) as data:
                values = data[name]
                finite_parts.append(values[np.isfinite(values)])
        finite = np.concatenate(finite_parts)
        if finite.size == 0:
            raise ValueError(f"all cached {name} values are non-finite")
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        if data_max - data_min < 3.0:
            limits[name] = (data_min, data_max)
        else:
            vmax = math.ceil(2.0 * data_max) / 2.0
            limits[name] = (max(math.floor(2.0 * data_min) / 2.0, vmax - 6.0), vmax)
    return limits


def diagnostic_limits(cache_paths: list[Path]) -> dict[str, tuple[float, float]]:
    limits = logarithmic_limits(
        cache_paths, [field for field in DIAGNOSTIC_FIELDS if field[-1]]
    )
    limits["abs_vz_mach_proxy"] = MACH_PROXY_LIMITS
    return limits


def render_broad(cache_path: Path, destination: Path, limits, dpi: int) -> None:
    with np.load(cache_path) as data:
        y_rsun = data["y_rsun"]
        z_rsun = data["z_rsun"]
        grids = {name: np.array(data[name]) for name, *_ in BROAD_FIELDS}
        vy = np.array(data["vy_kms"])
        vz = np.array(data["vz_stream_kms"])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 6.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, (name, title, label, cmap) in zip(axes.flat, BROAD_FIELDS):
        ax.set_box_aspect(0.5)
        scalar_map(
            grids[name],
            y_rsun,
            z_rsun,
            ax=ax,
            cmap=cmap,
            colorbar_label=label,
            log_scale=False,
            vmin=limits[name][0],
            vmax=limits[name][1],
            aspect_equal=False,
            colorbar_pad=0.02,
            shading="auto",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(float(y_rsun[0]), float(y_rsun[-1]))
        ax.set_ylim(float(z_rsun[0]), float(z_rsun[-1]))
        if name == "density":
            ax.streamplot(
                y_rsun,
                z_rsun,
                vy.T,
                vz.T,
                color="white",
                density=1.5,
                linewidth=0.6,
                arrowsize=0.8,
            )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$y/R_\odot$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z/R_\odot$")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def render_diagnostics(cache_path: Path, destination: Path, limits, dpi: int) -> None:
    with np.load(cache_path) as data:
        y_rsun = data["y_rsun"]
        z_rsun = data["z_rsun"]
        grids = {name: np.array(data[name]) for name, *_ in DIAGNOSTIC_FIELDS}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 6.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, (name, title, label, cmap, _) in zip(axes.flat, DIAGNOSTIC_FIELDS):
        ax.set_box_aspect(0.5)
        scalar_map(
            grids[name],
            y_rsun,
            z_rsun,
            ax=ax,
            cmap=cmap,
            colorbar_label=label,
            log_scale=False,
            vmin=limits[name][0],
            vmax=limits[name][1],
            aspect_equal=False,
            colorbar_pad=0.02,
            shading="auto",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(float(y_rsun[0]), float(y_rsun[-1]))
        ax.set_ylim(float(z_rsun[0]), float(z_rsun[-1]))
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$y/R_\odot$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z/R_\odot$")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def main(
    mode: int = typer.Option(..., help="1: 1e4, 2: 1e5, 3: 1e6 solar-mass BH"),
    resolution: int = typer.Option(768, min=16, help="Pixels along each slice axis"),
    workers: int = typer.Option(8, min=1, help="KD-tree query threads"),
    dpi: int = typer.Option(240, min=50, help="Output PNG resolution"),
    output_root: Path = typer.Option(OUTPUT_ROOT, help="Study output directory"),
    overwrite: bool = typer.Option(False, help="Recompute cached grids and figures"),
    rerender: bool = typer.Option(
        False, help="Redraw figures from existing cached grids"
    ),
    list_only: bool = typer.Option(
        False, help="Print selected snapshots without loading them"
    ),
) -> None:
    config = mode_settings(mode)
    selected = selected_snapshots(config.run)
    output_dir = output_root / config.run
    cache_dir = output_dir / "grids"

    for snapnum, path, is_last in selected:
        print(f"[{config.run}] snap {snapnum}: {path}{' (last)' if is_last else ''}")
    if list_only:
        return

    cache_paths = []
    for snapnum, path, _ in selected:
        cache_path = cache_dir / f"nozzle_yz_snap_{snapnum:04d}_{resolution}.npz"
        cache_paths.append(cache_path)
        if overwrite or not cache_complete(cache_path, resolution):
            print(f"[{config.run}] gridding snap {snapnum}", flush=True)
            cache_snapshot(path, cache_path, config, resolution, workers)
        else:
            print(f"[{config.run}] cached snap {snapnum}", flush=True)

    broad_limits = logarithmic_limits(cache_paths, BROAD_FIELDS)
    diagnostic_color_limits = diagnostic_limits(cache_paths)
    for (snapnum, _, _), cache_path in zip(selected, cache_paths):
        broad = output_dir / f"nozzle_yz_snap_{snapnum:04d}.png"
        diagnostics = output_dir / f"nozzle_yz_diagnostics_snap_{snapnum:04d}.png"
        for destination, renderer, limits in (
            (broad, render_broad, broad_limits),
            (diagnostics, render_diagnostics, diagnostic_color_limits),
        ):
            exists = destination.is_file() and destination.stat().st_size > 0
            if exists and not overwrite and not rerender:
                print(f"[{config.run}] exists {destination.name}", flush=True)
                continue
            print(f"[{config.run}] rendering {destination.name}", flush=True)
            renderer(cache_path, destination, limits, dpi)


if __name__ == "__main__":
    typer.run(main)
