#!/usr/bin/env python3
"""Cache shock geometry, physical zoom slices, and gridded pressure Mach slices."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import dev
import numpy as np
import typer
import unyt as u

import richio
from dev.datapaths import TDE_PARAMETERS


RESULT_ROOT = Path("/home/hey4/rich_tde/data/processed/ShockFinderEdissSelection")
ANALYSIS_ROOT = RESULT_ROOT / "analysis" / "zoom-shocks"
PER_CELL_ROOT = RESULT_ROOT / "analysis" / "per-cell"
SELF_ROOT = ANALYSIS_ROOT / "self-intersection"
GEOMETRY_ROOT = ANALYSIS_ROOT / "cells"
MACH_ROOT = ANALYSIS_ROOT / "mach-slices"
NOZZLE_ZOOM_ROOT = ANALYSIS_ROOT / "nozzle-orbit-zoom50"
SELF_MACH_MAX_ROOT = ANALYSIS_ROOT / "self-mach-max"
SELF_RESOLUTION = (1024, 512)
SELF_MACH_MAX_RESOLUTION = (256, 128)
SELF_MACH_MIN = 1.5
NOZZLE_RESOLUTION = (768, 768)
NOZZLE_ORBIT_VIEW_FRACTION = 0.5

NOZZLE_CASES = {("1e4", 77), ("1e5", 161), ("1e6", 850)}
NOZZLE_XY_WINDOW_RP = (-1.0, 2.0, -1.5, 1.5)
REFERENCE_RP_RSUN = 0.47 * (1e4 / 0.5) ** (1 / 3)
NOZZLE_YZ_SLICE_X_RP = 13.0 / REFERENCE_RP_RSUN
NOZZLE_YZ_WINDOW_RP = tuple(
    value / REFERENCE_RP_RSUN for value in (-10.0, 10.0, -5.0, 5.0)
)
NOZZLE_YZ_CENTER_RSUN = {"1e4": 0.0, "1e5": -20.0, "1e6": -80.0}
NOZZLE_YZ_BOX_SCALE = {"1e4": 1.0, "1e5": 1.2, "1e6": 1.2}
NOZZLE_ORBIT_RADIUS_SCALE = {"1e4": 0.9, "1e5": 1.0, "1e6": 1.0}
NOZZLE_ORBIT_PHASE_DEG = {"1e4": 0.0, "1e5": -13.0, "1e6": -13.0}
NOZZLE_ORBIT_CENTER_SHIFT_RSUN = {"1e4": -4.0, "1e5": -3.0, "1e6": -40.0}

CASES = (
    ("1e4", 77),
    ("1e5", 142),
    ("1e5", 161),
    ("1e6", 850),
)

# Centers and half widths are the executed values from
# 0.2-self-intersection-shock-zoom.ipynb, in units of r_p.
SELF_WINDOWS = {
    ("1e4", 77): {
        "center": (-10.583425085843034, 4.01533458408444, -0.0026021542615478465),
        "half_width": (7.0, 3.5, 2.0),
        "self_intersection_present": True,
    },
    ("1e5", 142): {
        "center": (-70.10152164964266, -3.1450587104153924, -0.00035850041190064064),
        "half_width": (15.0, 7.5, 3.0),
        "self_intersection_present": False,
    },
    ("1e6", 850): {
        "center": (-81.04461158864497, 12.495227152492864, -0.014973195701669205),
        "half_width": (27.0, 13.5, 5.4),
        "self_intersection_present": True,
    },
}


def load_study_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def needs_reference_frame(run: str, path: Path) -> bool:
    if run == "1e6":
        return path.parent.name == "TEMPTDE"
    return re.fullmatch(r"snap_\d+\.h5", path.name) is not None


def plotting_coordinates(snap, run: str, path: Path):
    x, y, z = snap.X, snap.Y, snap.Z
    if needs_reference_frame(run, path):
        m_bh, m_star, r_star = TDE_PARAMETERS[run]
        offset = dev.reference_frame_offset(
            t=snap.time,
            Mbh=m_bh * richio.units.mscale,
            Mstar=m_star * richio.units.mscale,
            Rstar=r_star * richio.units.lscale,
            beta=1,
        )
        x = x + offset[0]
        y = y + offset[1]
    return x, y, z


def atomic_save(destination: Path, **arrays) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def geometry_path(run: str, snapnum: int) -> Path:
    return GEOMETRY_ROOT / f"{run}_shock_cells_snap_{snapnum:04d}.npz"


def self_path(run: str, snapnum: int) -> Path:
    return SELF_ROOT / run / f"self_intersection_snap_{snapnum:04d}.npz"


def mach_path(run: str, snapnum: int) -> Path:
    return MACH_ROOT / run / f"mach_P_snap_{snapnum:04d}.npz"


def self_mach_max_path(run: str, snapnum: int) -> Path:
    return SELF_MACH_MAX_ROOT / run / f"self_mach_max_snap_{snapnum:04d}.npz"


def nozzle_zoom_path(run: str, snapnum: int) -> Path:
    return NOZZLE_ZOOM_ROOT / run / f"nozzle_orbit_zoom50_snap_{snapnum:04d}.npz"


def build_geometry(snap, run: str, snapnum: int, result, x, y, z) -> None:
    destination = geometry_path(run, snapnum)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Exists {destination}", flush=True)
        return

    surf_idx = np.asarray(result["surf_idx"], dtype=np.intp)
    mach = np.asarray(result["mach_T"], dtype="float64")
    per_cell_path = PER_CELL_ROOT / f"{run}_shock_dissipation_snap_{snapnum:04d}.npz"
    with np.load(per_cell_path) as cells:
        if not np.array_equal(surf_idx, cells["surf_idx"]):
            raise ValueError(f"Surface-cell ordering differs in {per_cell_path}")
        power = np.asarray(cells["shock_power_erg_s"], dtype="float64")

    volume = snap.volume[surf_idx].to("Rsun**3")
    effective_radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p_rsun = r_star * (m_bh / m_star) ** (1 / 3)
    atomic_save(
        destination,
        run=np.asarray(run),
        snapnum=np.asarray(snapnum),
        time_tfb=np.asarray(float(result["time_tfb"])),
        r_p_rsun=np.asarray(r_p_rsun),
        x_rsun=np.asarray(x[surf_idx].to_value("Rsun"), dtype="float64"),
        y_rsun=np.asarray(y[surf_idx].to_value("Rsun"), dtype="float64"),
        z_rsun=np.asarray(z[surf_idx].to_value("Rsun"), dtype="float64"),
        effective_radius_rsun=np.asarray(
            effective_radius.to_value("Rsun"), dtype="float64"
        ),
        mach_T=mach,
        shock_power_erg_s=power,
    )
    print(f"Saved {destination}", flush=True)


def positive_log10(values) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def sound_speed(density, pressure, specific_internal_energy):
    gamma_effective = 1 + pressure / (density * specific_internal_energy)
    return np.sqrt(gamma_effective * pressure / density)


def build_self_slice(snap, run: str, snapnum: int, x, y, z, workers: int) -> None:
    window = SELF_WINDOWS.get((run, snapnum))
    if window is None:
        return
    destination = self_path(run, snapnum)
    if destination.is_file() and destination.stat().st_size > 0:
        with np.load(destination) as cached:
            if {"vx_kms", "vy_kms"}.issubset(cached.files):
                print(f"Exists {destination}", flush=True)
                return

    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p = r_star * (m_bh / m_star) ** (1 / 3) * richio.units.lscale
    center = np.asarray(window["center"])
    half_width = np.asarray(window["half_width"])
    bounds = tuple(np.r_[center - half_width, center + half_width] * r_p)
    indices, xspace, yspace = snap.to_2dgrid(
        res=SELF_RESOLUTION,
        X=x,
        Y=y,
        Z=z,
        plane="xy",
        slice_coord=center[2] * r_p,
        box_size=bounds,
        volume_selection=True,
        workers=workers,
    )
    atomic_save(
        destination,
        run=np.asarray(run),
        snapnum=np.asarray(snapnum),
        center_rp=center,
        half_width_rp=half_width,
        self_intersection_present=np.asarray(window["self_intersection_present"]),
        x_rp=np.asarray(xspace / r_p, dtype="float64"),
        y_rp=np.asarray(yspace / r_p, dtype="float64"),
        density=positive_log10(snap.density[indices].in_cgs()),
        dissipation=positive_log10(snap.dissipation[indices].in_cgs()),
        vx_kms=snap.vx[indices].to_value("km/s"),
        vy_kms=snap.vy[indices].to_value("km/s"),
    )
    print(f"Saved {destination}", flush=True)


def build_nozzle_zoom(
    snap, run: str, snapnum: int, result, x, y, z, workers: int
) -> None:
    if (run, snapnum) not in NOZZLE_CASES:
        return
    destination = nozzle_zoom_path(run, snapnum)
    required = {
        "nozzle_xy_x_rp",
        "nozzle_xy_y_rp",
        "nozzle_xy_specific_vertical_kinetic",
        "nozzle_xy_specific_internal_energy",
        "nozzle_xy_abs_vz_over_cs",
        "nozzle_xy_vx_kms",
        "nozzle_xy_vy_kms",
        "nozzle_orbit_phi",
        "nozzle_orbit_s_rsun",
        "nozzle_orbit_z_rsun",
        "nozzle_orbit_x_rp",
        "nozzle_orbit_y_rp",
        "nozzle_orbit_density",
        "nozzle_orbit_pressure",
        "nozzle_orbit_dissipation",
        "nozzle_orbit_specific_vertical_kinetic",
        "nozzle_orbit_specific_internal_energy",
        "nozzle_orbit_abs_vz_over_cs",
        "nozzle_orbit_dphi_dt",
        "nozzle_orbit_dz_rsun_dt",
        "nozzle_orbit_mach_P",
    }
    axes = {
        "nozzle_xy_x_rp",
        "nozzle_xy_y_rp",
        "nozzle_orbit_phi",
        "nozzle_orbit_s_rsun",
        "nozzle_orbit_z_rsun",
        "nozzle_orbit_x_rp",
        "nozzle_orbit_y_rp",
    }
    if destination.is_file() and destination.stat().st_size > 0:
        with np.load(destination) as cached:
            complete = (
                required.issubset(cached.files)
                and all(
                    cached[name].shape == NOZZLE_RESOLUTION for name in required - axes
                )
                and all(cached[name].shape == (NOZZLE_RESOLUTION[0],) for name in axes)
                and "nozzle_orbit_center_shift_rsun" in cached
                and np.isclose(
                    float(cached["nozzle_orbit_center_shift_rsun"]),
                    NOZZLE_ORBIT_CENTER_SHIFT_RSUN[run],
                )
            )
        if complete:
            print(f"Exists {destination}", flush=True)
            return

    surf_idx = np.asarray(result["surf_idx"], dtype=np.intp)
    mach_p = np.zeros(len(snap), dtype="float64")
    mach_p[surf_idx] = np.asarray(result["mach_P"], dtype="float64")

    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p = r_star * (m_bh / m_star) ** (1 / 3) * richio.units.lscale
    xmin, xmax, ymin, ymax = NOZZLE_XY_WINDOW_RP
    xy_indices, xy_x, xy_y = snap.to_2dgrid(
        res=NOZZLE_RESOLUTION,
        X=x,
        Y=y,
        Z=z,
        plane="xy",
        slice_coord=0 * richio.units.lscale,
        box_size=(xmin * r_p, ymin * r_p, xmax * r_p, ymax * r_p),
        volume_selection=True,
        workers=workers,
    )
    xy_density = snap.density[xy_indices].in_cgs()
    xy_pressure = snap.pressure[xy_indices].in_cgs()
    xy_sie = snap.sie[xy_indices].to("erg/g")
    xy_vz = snap.vz[xy_indices].in_cgs()
    xy_cs = sound_speed(xy_density, xy_pressure, xy_sie)

    orbit_module = load_study_script(
        "nozzle_yz_slices_for_zoom50",
        Path(__file__).with_name("nozzle-yz-slices.py"),
    )
    mode = ("1e4", "1e5", "1e6").index(run) + 1
    config = orbit_module.mode_settings(mode)
    box_scale = NOZZLE_YZ_BOX_SCALE[run] * NOZZLE_ORBIT_VIEW_FRACTION
    radius_scale = NOZZLE_ORBIT_RADIUS_SCALE[run]
    phase_deg = NOZZLE_ORBIT_PHASE_DEG[run]
    center_shift_rsun = NOZZLE_ORBIT_CENTER_SHIFT_RSUN[run]
    time_tfb = float(result["time_tfb"])
    phi, orbit_x_rp, orbit_y_rp, orbit_s_rsun = orbit_module.returning_orbit_grid(
        config,
        time_tfb,
        box_scale,
        NOZZLE_RESOLUTION[0],
        radius_scale=radius_scale,
        phase_offset_deg=phase_deg,
        center_shift_rsun=center_shift_rsun,
        return_s_rsun=True,
    )
    r_sun = richio.units.lscale
    zmin, zmax = orbit_module.YZ_WINDOW_RP[2:]
    z_rsun = np.linspace(
        box_scale * zmin * config.r_p,
        box_scale * zmax * config.r_p,
        NOZZLE_RESOLUTION[1],
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
    orbit_indices = snap.nearest_indices(
        query_points,
        X=x,
        Y=y,
        Z=z,
        selection=selection,
        workers=workers,
    )
    orbit_density = snap.density[orbit_indices].in_cgs()
    orbit_pressure = snap.pressure[orbit_indices].in_cgs()
    orbit_sie = snap.sie[orbit_indices].to("erg/g")
    orbit_vz = snap.vz[orbit_indices].in_cgs()
    orbit_cs = sound_speed(orbit_density, orbit_pressure, orbit_sie)

    dx_dphi = np.gradient(orbit_x, phi)
    dy_dphi = np.gradient(orbit_y, phi)
    ds_dphi = np.hypot(dx_dphi, dy_dphi)
    tangent_x = dx_dphi / ds_dphi
    tangent_y = dy_dphi / ds_dphi
    tangent_speed = (
        snap.vx[orbit_indices] * tangent_x[:, None]
        + snap.vy[orbit_indices] * tangent_y[:, None]
    )
    atomic_save(
        destination,
        run=np.asarray(run),
        snapnum=np.asarray(snapnum),
        time_tfb=np.asarray(time_tfb),
        resolution=np.asarray(NOZZLE_RESOLUTION),
        nozzle_orbit_view_fraction=np.asarray(NOZZLE_ORBIT_VIEW_FRACTION),
        nozzle_orbit_box_scale=np.asarray(box_scale),
        nozzle_orbit_radius_scale=np.asarray(radius_scale),
        nozzle_orbit_phase_deg=np.asarray(phase_deg),
        nozzle_orbit_center_shift_rsun=np.asarray(center_shift_rsun),
        nozzle_xy_x_rp=np.asarray(xy_x / r_p, dtype="float64"),
        nozzle_xy_y_rp=np.asarray(xy_y / r_p, dtype="float64"),
        nozzle_xy_specific_vertical_kinetic=positive_log10(
            (0.5 * xy_vz**2).to("erg/g")
        ),
        nozzle_xy_specific_internal_energy=positive_log10(xy_sie),
        nozzle_xy_abs_vz_over_cs=np.asarray(np.abs(xy_vz) / xy_cs, dtype="float64"),
        nozzle_xy_vx_kms=snap.vx[xy_indices].to_value("km/s"),
        nozzle_xy_vy_kms=snap.vy[xy_indices].to_value("km/s"),
        nozzle_orbit_phi=np.asarray(phi, dtype="float64"),
        nozzle_orbit_s_rsun=np.asarray(orbit_s_rsun, dtype="float64"),
        nozzle_orbit_z_rsun=np.asarray(z_rsun, dtype="float64"),
        nozzle_orbit_x_rp=np.asarray(orbit_x_rp, dtype="float64"),
        nozzle_orbit_y_rp=np.asarray(orbit_y_rp, dtype="float64"),
        nozzle_orbit_density=positive_log10(orbit_density),
        nozzle_orbit_pressure=positive_log10(orbit_pressure),
        nozzle_orbit_dissipation=positive_log10(
            snap.dissipation[orbit_indices].in_cgs()
        ),
        nozzle_orbit_specific_vertical_kinetic=positive_log10(
            (0.5 * orbit_vz**2).to("erg/g")
        ),
        nozzle_orbit_specific_internal_energy=positive_log10(orbit_sie),
        nozzle_orbit_abs_vz_over_cs=np.asarray(
            np.abs(orbit_vz) / orbit_cs, dtype="float64"
        ),
        nozzle_orbit_dphi_dt=(tangent_speed / ds_dphi[:, None]).to_value("1/s"),
        nozzle_orbit_dz_rsun_dt=(orbit_vz / r_sun).to_value("1/s"),
        nozzle_orbit_mach_P=mach_p[orbit_indices],
    )
    print(f"Saved {destination}", flush=True)


def build_self_mach_max(snap, run: str, snapnum: int, result, x, y, z) -> None:
    window = SELF_WINDOWS.get((run, snapnum))
    if window is None:
        return
    destination = self_mach_max_path(run, snapnum)
    if destination.is_file() and destination.stat().st_size > 0:
        with np.load(destination) as cached:
            complete = (
                "mach_P_max" in cached
                and cached["mach_P_max"].shape == SELF_MACH_MAX_RESOLUTION
                and "mach_min" in cached
                and np.isclose(float(cached["mach_min"]), SELF_MACH_MIN)
            )
        if complete:
            print(f"Exists {destination}", flush=True)
            return

    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p = r_star * (m_bh / m_star) ** (1 / 3) * richio.units.lscale
    center = np.asarray(window["center"])
    half_width = np.asarray(window["half_width"])
    nx, ny = SELF_MACH_MAX_RESOLUTION
    grid_x = np.linspace(
        center[0] - half_width[0],
        center[0] + half_width[0],
        nx,
        endpoint=False,
    )
    grid_y = np.linspace(
        center[1] - half_width[1],
        center[1] + half_width[1],
        ny,
        endpoint=False,
    )
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    x_edges = np.linspace(grid_x[0] - 0.5 * dx, grid_x[-1] + 0.5 * dx, nx + 1)
    y_edges = np.linspace(grid_y[0] - 0.5 * dy, grid_y[-1] + 0.5 * dy, ny + 1)

    surf_idx = np.asarray(result["surf_idx"], dtype=np.intp)
    mach_p = np.asarray(result["mach_P"], dtype="float64")
    surface_x = np.asarray(x[surf_idx] / r_p, dtype="float64")
    surface_y = np.asarray(y[surf_idx] / r_p, dtype="float64")
    surface_z = z[surf_idx]
    surface_length = snap.volume[surf_idx] ** (1 / 3)
    slab = np.abs(surface_z - center[2] * r_p) < surface_length
    selected = (
        slab
        & np.isfinite(surface_x)
        & np.isfinite(surface_y)
        & np.isfinite(mach_p)
        & (mach_p >= SELF_MACH_MIN)
        & (surface_x >= x_edges[0])
        & (surface_x < x_edges[-1])
        & (surface_y >= y_edges[0])
        & (surface_y < y_edges[-1])
    )

    x_bin = np.searchsorted(x_edges, surface_x[selected], side="right") - 1
    y_bin = np.searchsorted(y_edges, surface_y[selected], side="right") - 1
    mach_max = np.zeros(SELF_MACH_MAX_RESOLUTION, dtype="float64")
    np.maximum.at(mach_max, (x_bin, y_bin), mach_p[selected])

    atomic_save(
        destination,
        run=np.asarray(run),
        snapnum=np.asarray(snapnum),
        center_rp=center,
        half_width_rp=half_width,
        resolution=np.asarray(SELF_MACH_MAX_RESOLUTION),
        mach_min=np.asarray(SELF_MACH_MIN),
        x_rp=grid_x,
        y_rp=grid_y,
        mach_P_max=mach_max,
        slab_cell_count=np.asarray(np.count_nonzero(selected)),
        nonzero_pixel_count=np.asarray(np.count_nonzero(mach_max)),
    )
    print(
        f"Saved {destination}: {np.count_nonzero(selected):,} slab shock cells with "
        f"M_P >= {SELF_MACH_MIN:g}, "
        f"{np.count_nonzero(mach_max):,} nonzero pixels",
        flush=True,
    )


def build_mach_slices(
    snap, run: str, snapnum: int, result, x, y, z, workers: int
) -> None:
    destination = mach_path(run, snapnum)
    required = set()
    if (run, snapnum) in NOZZLE_CASES:
        required.update({"nozzle_orbit_mach_P", "nozzle_orbit_pressure"})
    if (run, snapnum) in SELF_WINDOWS:
        required.add("self_mach_P")
    if destination.is_file() and destination.stat().st_size > 0:
        with np.load(destination) as cached:
            complete = required.issubset(cached.files)
            if (run, snapnum) in NOZZLE_CASES and complete:
                complete = np.isclose(
                    float(cached["nozzle_orbit_radius_scale"]),
                    NOZZLE_ORBIT_RADIUS_SCALE[run],
                )
            if complete:
                print(f"Exists {destination}", flush=True)
                return

    surf_idx = np.asarray(result["surf_idx"], dtype=np.intp)
    mach_p = np.zeros(len(snap), dtype="float64")
    mach_p[surf_idx] = np.asarray(result["mach_P"], dtype="float64")

    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    r_p = r_star * (m_bh / m_star) ** (1 / 3) * richio.units.lscale
    arrays = {}

    if (run, snapnum) in NOZZLE_CASES:
        xmin, xmax, ymin, ymax = NOZZLE_XY_WINDOW_RP
        indices, grid_x, grid_y = snap.to_2dgrid(
            res=NOZZLE_RESOLUTION,
            X=x,
            Y=y,
            Z=z,
            plane="xy",
            slice_coord=0 * richio.units.lscale,
            box_size=(xmin * r_p, ymin * r_p, xmax * r_p, ymax * r_p),
            volume_selection=True,
            workers=workers,
        )
        arrays.update(
            nozzle_xy_x_rp=np.asarray(grid_x / r_p, dtype="float64"),
            nozzle_xy_y_rp=np.asarray(grid_y / r_p, dtype="float64"),
            nozzle_xy_mach_P=mach_p[indices],
        )

        ymin, ymax, zmin, zmax = NOZZLE_YZ_WINDOW_RP
        scale = NOZZLE_YZ_BOX_SCALE[run]
        y_center = NOZZLE_YZ_CENTER_RSUN[run] * richio.units.lscale
        indices, grid_y, grid_z = snap.to_2dgrid(
            res=NOZZLE_RESOLUTION,
            X=x,
            Y=y,
            Z=z,
            plane="yz",
            slice_coord=NOZZLE_YZ_SLICE_X_RP * r_p,
            box_size=(
                scale * ymin * r_p + y_center,
                scale * zmin * r_p,
                scale * ymax * r_p + y_center,
                scale * zmax * r_p,
            ),
            volume_selection=True,
            workers=workers,
        )
        arrays.update(
            nozzle_yz_y_rsun=np.asarray(grid_y / richio.units.lscale, dtype="float64"),
            nozzle_yz_z_rsun=np.asarray(grid_z / richio.units.lscale, dtype="float64"),
            nozzle_yz_mach_P=mach_p[indices],
        )

        orbit_module = load_study_script(
            "nozzle_yz_slices_for_shocks",
            Path(__file__).with_name("nozzle-yz-slices.py"),
        )
        mode = ("1e4", "1e5", "1e6").index(run) + 1
        config = orbit_module.mode_settings(mode)
        box_scale = NOZZLE_YZ_BOX_SCALE[run]
        radius_scale = NOZZLE_ORBIT_RADIUS_SCALE[run]
        phase_deg = NOZZLE_ORBIT_PHASE_DEG[run]
        time_tfb = float(result["time_tfb"])
        phi, orbit_x_rp, orbit_y_rp = orbit_module.returning_orbit_grid(
            config,
            time_tfb,
            box_scale,
            NOZZLE_RESOLUTION[0],
            radius_scale=radius_scale,
            phase_offset_deg=phase_deg,
        )
        r_sun = richio.units.lscale
        zmin, zmax = orbit_module.YZ_WINDOW_RP[2:]
        z_rsun = np.linspace(
            box_scale * zmin * config.r_p,
            box_scale * zmax * config.r_p,
            NOZZLE_RESOLUTION[1],
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
        dx_dphi = np.gradient(orbit_x, phi)
        dy_dphi = np.gradient(orbit_y, phi)
        ds_dphi = np.hypot(dx_dphi, dy_dphi)
        tangent_x = dx_dphi / ds_dphi
        tangent_y = dy_dphi / ds_dphi
        tangent_speed = (
            snap.vx[indices] * tangent_x[:, None]
            + snap.vy[indices] * tangent_y[:, None]
        )
        arrays.update(
            nozzle_orbit_phi=np.asarray(phi, dtype="float64"),
            nozzle_orbit_z_rsun=np.asarray(z_rsun, dtype="float64"),
            nozzle_orbit_x_rp=np.asarray(orbit_x_rp, dtype="float64"),
            nozzle_orbit_y_rp=np.asarray(orbit_y_rp, dtype="float64"),
            nozzle_orbit_density=positive_log10(snap.density[indices].in_cgs()),
            nozzle_orbit_pressure=positive_log10(snap.pressure[indices].in_cgs()),
            nozzle_orbit_dissipation=positive_log10(snap.dissipation[indices].in_cgs()),
            nozzle_orbit_dphi_dt=(tangent_speed / ds_dphi[:, None]).to_value("1/s"),
            nozzle_orbit_dz_rsun_dt=(snap.vz[indices] / r_sun).to_value("1/s"),
            nozzle_orbit_mach_P=mach_p[indices],
            nozzle_orbit_box_scale=np.asarray(box_scale),
            nozzle_orbit_radius_scale=np.asarray(radius_scale),
            nozzle_orbit_phase_deg=np.asarray(phase_deg),
        )

    window = SELF_WINDOWS.get((run, snapnum))
    if window is not None:
        center = np.asarray(window["center"])
        half_width = np.asarray(window["half_width"])
        bounds = tuple(np.r_[center - half_width, center + half_width] * r_p)
        indices, grid_x, grid_y = snap.to_2dgrid(
            res=SELF_RESOLUTION,
            X=x,
            Y=y,
            Z=z,
            plane="xy",
            slice_coord=center[2] * r_p,
            box_size=bounds,
            volume_selection=True,
            workers=workers,
        )
        arrays.update(
            self_x_rp=np.asarray(grid_x / r_p, dtype="float64"),
            self_y_rp=np.asarray(grid_y / r_p, dtype="float64"),
            self_mach_P=mach_p[indices],
        )

    atomic_save(destination, **arrays)
    print(f"Saved {destination}", flush=True)


def main(
    task_index: int = typer.Option(..., min=0, max=len(CASES) - 1),
    workers: int = typer.Option(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), min=1),
) -> None:
    run, snapnum = CASES[task_index]
    result_path = RESULT_ROOT / run / f"shockfinder_snap_{snapnum:04d}.npz"
    with np.load(result_path) as result:
        snap_path = Path(str(result["snap_path"].item()))
        snap = richio.load(str(snap_path))
        x, y, z = plotting_coordinates(snap, run, snap_path)
        print(f"[{task_index}] {run} snap {snapnum}: {snap_path}", flush=True)
        build_geometry(snap, run, snapnum, result, x, y, z)
        build_self_slice(snap, run, snapnum, x, y, z, workers)
        build_nozzle_zoom(snap, run, snapnum, result, x, y, z, workers)
        build_self_mach_max(snap, run, snapnum, result, x, y, z)
        build_mach_slices(snap, run, snapnum, result, x, y, z, workers)


if __name__ == "__main__":
    typer.run(main)
