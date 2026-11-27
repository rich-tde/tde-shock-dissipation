#!/usr/bin/env python3
r"""Build compact geometry and field caches for inspecting detected TDE shocks.

Read an existing shock detection and its simulation snapshot, move positions to
the black-hole (BH) frame when needed, and sample nozzle or self-intersection
views. This command writes numerical caches, not figures. The boxes and shifted
ballistic orbits are descriptive choices, not an automatic classifier.

Input files
-----------
Run from ``/home/hey4/rich_tde``. ``ROOT`` is
``data/processed/ShockFinderResults`` (``--result-root``), ``RUN`` is
``1e4``, ``1e5`` or ``ss24`` and ``NNNN`` a four-digit snapshot number.

``ROOT/RUN/shockfinder_snap_NNNN.npz``
    From ``shock-finder-ediss-selection.py``: ``snap_path``, ``time_tfb``,
    ``surf_idx`` (N,) and ``mach_T`` / ``mach_P`` (N,) in the same order.
``ROOT/analysis/per-cell/RUN_shock_dissipation_snap_NNNN.npz``
    Only for ``--product geometry``; from ``0.5-shock-finder-ediss-analysis.ipynb``.
    ``surf_idx`` must equal the detector's; ``shock_power_erg_s`` (N,).

Output files
------------
Compressed ``.npz`` under ``OUT = ROOT/analysis/zoom-shocks`` (``--output-root``).
Grids are indexed ``field[i, j]`` along the first then the second named axis, so
plot with ``pcolormesh(a, b, field.T)``. Axes exclude the upper box edge.
Positions are BH-frame; velocities keep the snapshot's stored frame. Logged
fields are base-10 logs in cgs with non-positive values set to NaN. Mach grids
are zero where the nearest cell is not a detected surface cell. ``*_rsun`` grid
axes are in RICH code lengths (``richio.units.lscale``, ≈ R_sun); only the
geometry file uses physical ``Rsun``.

``OUT/cells/RUN_shock_cells_snap_NNNN.npz`` (geometry)
    ``time_tfb``, ``r_p_rsun`` (code lengths), ``x_rsun``/``y_rsun``/``z_rsun`` and
    ``effective_radius_rsun = (3V/4π)^(1/3)`` (N,) in Rsun, ``mach_T``,
    ``shock_power_erg_s``, in the detector's ``surf_idx`` order.
``OUT/self-intersection/RUN/self_intersection_snap_NNNN.npz`` (self)
    1024 × 512 xy slice through the box centre: ``center_rp``, ``half_width_rp``
    (3,), ``self_intersection_present`` (manual label; NaN for a custom box),
    ``x_rp``, ``y_rp``, logged ``density`` and ``dissipation``, ``vx_kms``, ``vy_kms``.
``OUT/nozzle-orbit-zoom50/RUN/nozzle_orbit_zoom50_snap_NNNN.npz`` (nozzle)
    768 × 768. ``nozzle_xy_*``: z = 0 slice with ``x_rp``, ``y_rp``,
    ``specific_vertical_kinetic`` (log ½v_z²), ``specific_internal_energy`` (log),
    ``abs_vz_over_cs``, ``vx_kms``, ``vy_kms``. ``nozzle_orbit_*``: vertical
    section along the returning orbit at half the window, with ``phi`` (rad, zero
    at pericentre), ``s_rsun`` (signed arc, positive before pericentre), ``z_rsun``,
    ``x_rp``, ``y_rp``, logged ``density``, ``pressure``, ``dissipation``,
    ``specific_vertical_kinetic``, ``specific_internal_energy``, linear
    ``abs_vz_over_cs``, ``dphi_dt`` and ``dz_rsun_dt`` (1/s), ``mach_P``, and the
    window settings ``view_fraction``, ``box_scale``, ``radius_scale``,
    ``phase_deg``, ``center_shift_rsun``. ``c_s = sqrt((1 + P/(ρu)) P/ρ)``.
``OUT/self-mach-max/RUN/self_mach_max_snap_NNNN.npz`` (self-mach-max)
    Maximum ``mach_P >= mach_min`` (1.5) of surface cells within one cell length
    ``V^(1/3)`` of the box's z plane, in 256 × 128 xy bins: ``x_rp``, ``y_rp``
    (bin centres), ``mach_P_max`` (zero = empty), ``slab_cell_count``,
    ``nonzero_pixel_count``.
``OUT/mach-slices/RUN/mach_P_snap_NNNN.npz`` (mach)
    Pressure-jump Mach for the nozzle cases: ``nozzle_xy_*`` (as above),
    ``nozzle_yz_*`` (x = 13 R_sun slice, axes ``y_rsun``, ``z_rsun``) and the
    full-window, unshifted orbit section ``nozzle_orbit_*`` (no ``s_rsun``); and
    ``self_x_rp``, ``self_y_rp``, ``self_mach_P`` for self-intersection boxes.

Usage
-----
::

    python works/shock-tde/shock-zoom-caches.py --list-only
    python works/shock-tde/shock-zoom-caches.py --task-index 0 --workers 8
    python works/shock-tde/shock-zoom-caches.py --run 1e4 --snapnum 77 --product nozzle
    python works/shock-tde/shock-zoom-caches.py --run 1e4 --snapnum 80 \
        --product self --self-center -10 4 0 --self-half-width 7 3.5 2 \
        --output-root data/processed/MyShockZoom

Repeat ``--product`` to choose products; by default all applicable ones are
built. Caches that already match the current settings are reused unless
``--overwrite``.

Loading example
---------------
::

    from dev.io import load_npz
    root = "data/processed/ShockFinderResults/analysis/zoom-shocks"
    orbit = load_npz(f"{root}/nozzle-orbit-zoom50/1e4/nozzle_orbit_zoom50_snap_0077.npz",
                     prefix="nozzle_orbit_")
    plt.pcolormesh(orbit["s_rsun"], orbit["z_rsun"], orbit["density"].T)
"""

import os
from dataclasses import dataclass

import numpy as np
import typer
import unyt as u
from dev.datapaths import TDE_PARAMETERS
from dev.io import cache_matches, load_npz, save_npz_atomic
from dev.plotting import positive_log10
from dev.tde import bh_frame_positions, returning_orbit_window, tde_basics

import richio

CODE_LENGTH = richio.units.lscale
RESULT_ROOT = "/home/hey4/rich_tde/data/processed/ShockFinderResults"
PRODUCTS = ("geometry", "self", "nozzle", "self-mach-max", "mach")
CACHE_NAMES = {
    "geometry": "cells/{run}_shock_cells_snap_{snapnum:04d}.npz",
    "self": "self-intersection/{run}/self_intersection_snap_{snapnum:04d}.npz",
    "nozzle": "nozzle-orbit-zoom50/{run}/nozzle_orbit_zoom50_snap_{snapnum:04d}.npz",
    "self-mach-max": "self-mach-max/{run}/self_mach_max_snap_{snapnum:04d}.npz",
    "mach": "mach-slices/{run}/mach_P_snap_{snapnum:04d}.npz",
}

CASES = (("1e4", 77), ("1e5", 142), ("1e5", 161), ("ss24", 850))
NOZZLE_CASES = {("1e4", 77), ("1e5", 161), ("ss24", 850)}
NOZZLE_RESOLUTION = (768, 768)
NOZZLE_XY_WINDOW_RP = (-1.0, 2.0, -1.5, 1.5)
REFERENCE_RP = tde_basics("1e4").r_p.to_value(CODE_LENGTH)
NOZZLE_YZ_SLICE_X = 13.0
NOZZLE_YZ_HALF_WIDTH_RP = 10.0 / REFERENCE_RP
NOZZLE_YZ_HALF_HEIGHT_RP = 5.0 / REFERENCE_RP
NOZZLE_YZ_CENTER = {"1e4": 0.0, "1e5": -20.0, "ss24": -80.0}
NOZZLE_YZ_BOX_SCALE = {"1e4": 1.0, "1e5": 1.2, "ss24": 1.2}
NOZZLE_ORBIT_VIEW_FRACTION = 0.5
NOZZLE_ORBIT_RADIUS_SCALE = {"1e4": 0.9, "1e5": 1.0, "ss24": 1.0}
NOZZLE_ORBIT_PHASE_DEG = {"1e4": 0.0, "1e5": -13.0, "ss24": -13.0}
NOZZLE_ORBIT_CENTER_SHIFT = {"1e4": -4.0, "1e5": -3.0, "ss24": -40.0}

SELF_RESOLUTION = (1024, 512)
SELF_MACH_MAX_RESOLUTION = (256, 128)
SELF_MACH_MIN = 1.5
# Executed values from 0.2-self-intersection-shock-zoom.ipynb, in units of r_p.
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
    ("ss24", 850): {
        "center": (-81.04461158864497, 12.495227152492864, -0.014973195701669205),
        "half_width": (27.0, 13.5, 5.4),
        "self_intersection_present": True,
    },
}

NOZZLE_KEYS = [
    f"nozzle_xy_{name}"
    for name in (
        "x_rp",
        "y_rp",
        "specific_vertical_kinetic",
        "specific_internal_energy",
        "abs_vz_over_cs",
        "vx_kms",
        "vy_kms",
    )
] + [
    f"nozzle_orbit_{name}"
    for name in (
        "phi",
        "s_rsun",
        "z_rsun",
        "x_rp",
        "y_rp",
        "density",
        "pressure",
        "dissipation",
        "specific_vertical_kinetic",
        "specific_internal_energy",
        "abs_vz_over_cs",
        "dphi_dt",
        "dz_rsun_dt",
        "mach_P",
    )
]


@dataclass
class Case:
    run: str
    snapnum: int
    snap: object
    result: dict
    x: u.unyt_array
    y: u.unyt_array
    z: u.unyt_array
    per_cell_path: str
    workers: int

    @property
    def r_p(self):
        return tde_basics(self.run).r_p

    @property
    def mach_p(self):
        """Pressure-jump Mach on every cell, zero off the shock surface."""
        mach = np.zeros(len(self.snap))
        mach[self.result["surf_idx"]] = self.result["mach_P"]
        return mach

    def grid(self, plane, box, slice_coord, resolution):
        return self.snap.to_2dgrid(
            res=resolution,
            X=self.x,
            Y=self.y,
            Z=self.z,
            plane=plane,
            slice_coord=slice_coord,
            box_size=box,
            volume_selection=True,
            workers=self.workers,
        )


def sample_fields(snap, indices):
    density, pressure, internal_energy, vz = (
        snap.density[indices],
        snap.pressure[indices],
        snap.sie[indices],
        snap.vz[indices],
    )
    sound_speed = np.sqrt(
        (1 + pressure / (density * internal_energy)) * pressure / density
    )
    return {
        "density": positive_log10(density.in_cgs()),
        "pressure": positive_log10(pressure.in_cgs()),
        "dissipation": positive_log10(snap.dissipation[indices].in_cgs()),
        "specific_vertical_kinetic": positive_log10((0.5 * vz**2).to("erg/g")),
        "specific_internal_energy": positive_log10(internal_energy.to("erg/g")),
        "abs_vz_over_cs": (np.abs(vz) / sound_speed).to_value("dimensionless"),
        "vx_kms": snap.vx[indices].to_value("km/s"),
        "vy_kms": snap.vy[indices].to_value("km/s"),
    }


def nozzle_xy_slice(case):
    xmin, xmax, ymin, ymax = np.multiply(NOZZLE_XY_WINDOW_RP, case.r_p)
    indices, x, y = case.grid(
        "xy", (xmin, ymin, xmax, ymax), 0 * CODE_LENGTH, NOZZLE_RESOLUTION
    )
    return indices, {
        "x_rp": (x / case.r_p).to_value("dimensionless"),
        "y_rp": (y / case.r_p).to_value("dimensionless"),
    }


def self_slice(case):
    window = SELF_WINDOWS[(case.run, case.snapnum)]
    center, half_width = np.asarray(window["center"]), np.asarray(window["half_width"])
    box = tuple(np.r_[center - half_width, center + half_width] * case.r_p)
    indices, x, y = case.grid("xy", box, center[2] * case.r_p, SELF_RESOLUTION)
    return indices, {
        "x_rp": (x / case.r_p).to_value("dimensionless"),
        "y_rp": (y / case.r_p).to_value("dimensionless"),
    }


def orbit_section(case, box_scale, center_shift):
    """Nearest cells on a vertical section following the returning orbit."""
    run = case.run
    phi, x_rp, y_rp, s = returning_orbit_window(
        run,
        case.result["time_tfb"],
        box_scale * NOZZLE_YZ_HALF_WIDTH_RP,
        NOZZLE_RESOLUTION[0],
        NOZZLE_ORBIT_RADIUS_SCALE[run],
        NOZZLE_ORBIT_PHASE_DEG[run],
        center_shift * CODE_LENGTH,
    )
    height = box_scale * NOZZLE_YZ_HALF_HEIGHT_RP * case.r_p.to_value(CODE_LENGTH)
    z_axis = np.linspace(-height, height, NOZZLE_RESOLUTION[1], endpoint=False)
    orbit_x, orbit_y = x_rp * case.r_p, y_rp * case.r_p

    points = np.stack(
        np.broadcast_arrays(
            orbit_x.value[:, None], orbit_y.value[:, None], z_axis[None, :]
        ),
        axis=-1,
    )
    padding = 0.1 * case.r_p
    nearby = (
        (case.x > orbit_x.min() - padding)
        & (case.x < orbit_x.max() + padding)
        & (case.y > orbit_y.min() - padding)
        & (case.y < orbit_y.max() + padding)
        & (case.z > (z_axis[0] * CODE_LENGTH - padding))
        & (case.z < (z_axis[-1] * CODE_LENGTH + padding))
    )
    indices = case.snap.nearest_indices(
        points * CODE_LENGTH,
        X=case.x,
        Y=case.y,
        Z=case.z,
        selection=nearby,
        workers=case.workers,
    )

    dx_dphi, dy_dphi = np.gradient(orbit_x, phi), np.gradient(orbit_y, phi)
    ds_dphi = np.hypot(dx_dphi, dy_dphi)
    along_orbit = (
        case.snap.vx[indices] * dx_dphi[:, None]
        + case.snap.vy[indices] * dy_dphi[:, None]
    ) / ds_dphi[:, None]
    geometry = {
        "phi": phi,
        "s_rsun": s.to_value(CODE_LENGTH),
        "z_rsun": z_axis,
        "x_rp": x_rp,
        "y_rp": y_rp,
        "dphi_dt": (along_orbit / ds_dphi[:, None]).to_value("1/s"),
        "dz_rsun_dt": (case.snap.vz[indices] / CODE_LENGTH).to_value("1/s"),
        "box_scale": box_scale,
        "radius_scale": NOZZLE_ORBIT_RADIUS_SCALE[run],
        "phase_deg": NOZZLE_ORBIT_PHASE_DEG[run],
    }
    return indices, geometry


def prefixed(prefix, arrays):
    return {f"{prefix}{name}": value for name, value in arrays.items()}


def build_geometry(case):
    surface = case.result["surf_idx"]
    with np.load(case.per_cell_path) as cells:
        if not np.array_equal(surface, cells["surf_idx"]):
            raise ValueError(f"Surface-cell ordering differs in {case.per_cell_path}")
        power = cells["shock_power_erg_s"]
    effective_radius = (3 * case.snap.volume[surface] / (4 * np.pi)) ** (1 / 3)
    return {
        "run": case.run,
        "snapnum": case.snapnum,
        "time_tfb": case.result["time_tfb"],
        "r_p_rsun": case.r_p.to_value(CODE_LENGTH),
        "x_rsun": case.x[surface].to_value("Rsun"),
        "y_rsun": case.y[surface].to_value("Rsun"),
        "z_rsun": case.z[surface].to_value("Rsun"),
        "effective_radius_rsun": effective_radius.to_value("Rsun"),
        "mach_T": case.result["mach_T"],
        "shock_power_erg_s": power,
    }


def build_self(case):
    window = SELF_WINDOWS[(case.run, case.snapnum)]
    indices, axes = self_slice(case)
    fields = sample_fields(case.snap, indices)
    return {
        "run": case.run,
        "snapnum": case.snapnum,
        "center_rp": np.asarray(window["center"]),
        "half_width_rp": np.asarray(window["half_width"]),
        "self_intersection_present": window["self_intersection_present"],
        **axes,
        **{
            name: fields[name]
            for name in ("density", "dissipation", "vx_kms", "vy_kms")
        },
    }


def build_nozzle(case):
    xy_indices, xy_axes = nozzle_xy_slice(case)
    xy_fields = sample_fields(case.snap, xy_indices)
    box_scale = NOZZLE_YZ_BOX_SCALE[case.run] * NOZZLE_ORBIT_VIEW_FRACTION
    center_shift = NOZZLE_ORBIT_CENTER_SHIFT[case.run]
    orbit_indices, orbit = orbit_section(case, box_scale, center_shift)
    orbit_fields = sample_fields(case.snap, orbit_indices)
    del orbit_fields["vx_kms"], orbit_fields["vy_kms"]
    xy_names = (
        "specific_vertical_kinetic",
        "specific_internal_energy",
        "abs_vz_over_cs",
        "vx_kms",
        "vy_kms",
    )
    return {
        "run": case.run,
        "snapnum": case.snapnum,
        "time_tfb": case.result["time_tfb"],
        "resolution": np.asarray(NOZZLE_RESOLUTION),
        **prefixed(
            "nozzle_xy_", xy_axes | {name: xy_fields[name] for name in xy_names}
        ),
        **prefixed(
            "nozzle_orbit_",
            orbit
            | orbit_fields
            | {
                "view_fraction": NOZZLE_ORBIT_VIEW_FRACTION,
                "center_shift_rsun": center_shift,
                "mach_P": case.mach_p[orbit_indices],
            },
        ),
    }


def build_self_mach_max(case):
    window = SELF_WINDOWS[(case.run, case.snapnum)]
    center, half_width = np.asarray(window["center"]), np.asarray(window["half_width"])
    centres, edges = zip(
        *(
            bin_centres_and_edges(
                center[axis], half_width[axis], SELF_MACH_MAX_RESOLUTION[axis]
            )
            for axis in (0, 1)
        )
    )

    surface = case.result["surf_idx"]
    mach = case.result["mach_P"]
    surface_x = (case.x[surface] / case.r_p).to_value("dimensionless")
    surface_y = (case.y[surface] / case.r_p).to_value("dimensionless")
    in_slab = np.abs(case.z[surface] - center[2] * case.r_p) < case.snap.volume[
        surface
    ] ** (1 / 3)
    selected = (
        in_slab
        & np.isfinite(surface_x)
        & np.isfinite(surface_y)
        & np.isfinite(mach)
        & (mach >= SELF_MACH_MIN)
        & (surface_x >= edges[0][0])
        & (surface_x < edges[0][-1])
        & (surface_y >= edges[1][0])
        & (surface_y < edges[1][-1])
    )
    bins = tuple(
        np.searchsorted(axis_edges, values[selected], side="right") - 1
        for axis_edges, values in zip(edges, (surface_x, surface_y))
    )
    mach_max = np.zeros(SELF_MACH_MAX_RESOLUTION)
    np.maximum.at(mach_max, bins, mach[selected])
    print(
        f"{np.count_nonzero(selected):,} slab shock cells with M_P >= {SELF_MACH_MIN:g}, "
        f"{np.count_nonzero(mach_max):,} nonzero pixels"
    )
    return {
        "run": case.run,
        "snapnum": case.snapnum,
        "center_rp": center,
        "half_width_rp": half_width,
        "resolution": np.asarray(SELF_MACH_MAX_RESOLUTION),
        "mach_min": SELF_MACH_MIN,
        "x_rp": centres[0],
        "y_rp": centres[1],
        "mach_P_max": mach_max,
        "slab_cell_count": np.count_nonzero(selected),
        "nonzero_pixel_count": np.count_nonzero(mach_max),
    }


def bin_centres_and_edges(center, half_width, count):
    centres = np.linspace(
        center - half_width, center + half_width, count, endpoint=False
    )
    step = centres[1] - centres[0]
    return centres, np.linspace(
        centres[0] - step / 2, centres[-1] + step / 2, count + 1
    )


def build_mach(case):
    mach = case.mach_p
    arrays = {}
    if (case.run, case.snapnum) in NOZZLE_CASES:
        indices, axes = nozzle_xy_slice(case)
        arrays |= prefixed("nozzle_xy_", axes | {"mach_P": mach[indices]})

        scale = NOZZLE_YZ_BOX_SCALE[case.run]
        y_center = NOZZLE_YZ_CENTER[case.run] * CODE_LENGTH
        half_width, half_height = (
            scale * NOZZLE_YZ_HALF_WIDTH_RP * case.r_p,
            scale * NOZZLE_YZ_HALF_HEIGHT_RP * case.r_p,
        )
        box = (y_center - half_width, -half_height, y_center + half_width, half_height)
        indices, y, z = case.grid(
            "yz", box, NOZZLE_YZ_SLICE_X / REFERENCE_RP * case.r_p, NOZZLE_RESOLUTION
        )
        arrays |= prefixed(
            "nozzle_yz_",
            {
                "y_rsun": y.to_value(CODE_LENGTH),
                "z_rsun": z.to_value(CODE_LENGTH),
                "mach_P": mach[indices],
            },
        )

        indices, orbit = orbit_section(case, scale, center_shift=0.0)
        del orbit["s_rsun"]
        fields = sample_fields(case.snap, indices)
        arrays |= prefixed(
            "nozzle_orbit_",
            orbit
            | {name: fields[name] for name in ("density", "pressure", "dissipation")}
            | {"mach_P": mach[indices]},
        )

    if (case.run, case.snapnum) in SELF_WINDOWS:
        indices, axes = self_slice(case)
        arrays |= prefixed("self_", axes | {"mach_P": mach[indices]})
    return arrays


BUILDERS = {
    "geometry": build_geometry,
    "self": build_self,
    "nozzle": build_nozzle,
    "self-mach-max": build_self_mach_max,
    "mach": build_mach,
}


def is_current(product, path, run, snapnum):
    nozzle = (run, snapnum) in NOZZLE_CASES
    if product == "self":
        return cache_matches(path, keys=("vx_kms", "vy_kms"))
    if product == "nozzle":
        return cache_matches(
            path,
            keys=NOZZLE_KEYS,
            resolution=NOZZLE_RESOLUTION,
            nozzle_orbit_center_shift_rsun=NOZZLE_ORBIT_CENTER_SHIFT[run],
        )
    if product == "self-mach-max":
        return cache_matches(
            path,
            keys=("mach_P_max",),
            resolution=SELF_MACH_MAX_RESOLUTION,
            mach_min=SELF_MACH_MIN,
        )
    if product == "mach":
        keys = ["nozzle_orbit_mach_P", "nozzle_orbit_pressure"] * nozzle
        keys += ["self_mach_P"] * ((run, snapnum) in SELF_WINDOWS)
        expected = (
            {"nozzle_orbit_radius_scale": NOZZLE_ORBIT_RADIUS_SCALE[run]}
            if nozzle
            else {}
        )
        return cache_matches(path, keys=keys, **expected)
    return cache_matches(path)


def applicable(product, run, snapnum):
    has_self = (run, snapnum) in SELF_WINDOWS
    has_nozzle = (run, snapnum) in NOZZLE_CASES
    return {
        "self": has_self,
        "self-mach-max": has_self,
        "nozzle": has_nozzle,
        "mach": has_self or has_nozzle,
    }.get(product, True)


def choose_case(task_index, run, snapnum):
    if task_index is not None:
        if run is not None or snapnum is not None:
            raise typer.BadParameter("Choose --task-index or --run/--snapnum.")
        return CASES[task_index]
    if run not in TDE_PARAMETERS or snapnum is None:
        raise typer.BadParameter(
            "Choose --task-index, or --run {1e4,1e5,ss24} --snapnum N."
        )
    return run, snapnum


def main(
    task_index: int | None = typer.Option(
        None,
        min=0,
        max=len(CASES) - 1,
        help="Established case: 0=1e4/77, 1=1e5/142, 2=1e5/161, 3=ss24/850.",
    ),
    run: str | None = typer.Option(
        None, help="Explicit run: 1e4, 1e5 or ss24; use with --snapnum."
    ),
    snapnum: int | None = typer.Option(
        None, min=0, help="Explicit snapshot number; use with --run."
    ),
    product: list[str] | None = typer.Option(
        None, help=f"Repeat: {', '.join(PRODUCTS)}; default all."
    ),
    result_root: str = typer.Option(
        RESULT_ROOT, help="Shock-finder root with RUN/shockfinder_snap_NNNN.npz."
    ),
    output_root: str | None = typer.Option(
        None, help="Cache root; default RESULT_ROOT/analysis/zoom-shocks."
    ),
    per_cell_root: str | None = typer.Option(
        None, help="Geometry input root; default RESULT_ROOT/analysis/per-cell."
    ),
    self_center: tuple[float, float, float] | None = typer.Option(
        None,
        help="Self-intersection box centre X Y Z in r_p; pair with --self-half-width.",
    ),
    self_half_width: tuple[float, float, float] | None = typer.Option(
        None, help="Box half widths X Y Z in r_p."
    ),
    workers: int = typer.Option(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        min=1,
        help="KD-tree query workers.",
    ),
    overwrite: bool = typer.Option(
        False, help="Rebuild caches even if they match the current settings."
    ),
    list_only: bool = typer.Option(
        False, help="Print cases or input/output paths without loading snapshots."
    ),
) -> None:
    """Build small shock-geometry and physical-field caches from raw snapshots."""
    if list_only and task_index is None and run is None and snapnum is None:
        for index, (case_run, number) in enumerate(CASES):
            print(f"[{index}] {case_run} snap {number}")
        return
    run, snapnum = choose_case(task_index, run, snapnum)
    output_root = output_root or os.path.join(result_root, "analysis/zoom-shocks")
    per_cell_root = per_cell_root or os.path.join(result_root, "analysis/per-cell")

    if (self_center is None) != (self_half_width is None):
        raise typer.BadParameter("Supply both --self-center and --self-half-width.")
    if self_center is not None:
        present = SELF_WINDOWS.get((run, snapnum), {}).get(
            "self_intersection_present", np.nan
        )
        SELF_WINDOWS[(run, snapnum)] = {
            "center": self_center,
            "half_width": self_half_width,
            "self_intersection_present": present,
        }
    if unknown := set(product or ()) - set(PRODUCTS):
        raise typer.BadParameter(f"Unknown products: {', '.join(sorted(unknown))}")
    if product and "nozzle" in product:
        NOZZLE_CASES.add((run, snapnum))
    if missing := [
        name for name in product or () if not applicable(name, run, snapnum)
    ]:
        raise typer.BadParameter(
            f"No configured geometry for {', '.join(missing)}; supply a box or --product nozzle."
        )

    products = [name for name in product or PRODUCTS if applicable(name, run, snapnum)]
    paths = {
        name: os.path.join(
            output_root, CACHE_NAMES[name].format(run=run, snapnum=snapnum)
        )
        for name in products
    }
    result_path = os.path.join(result_root, run, f"shockfinder_snap_{snapnum:04d}.npz")
    per_cell_path = os.path.join(
        per_cell_root, f"{run}_shock_dissipation_snap_{snapnum:04d}.npz"
    )
    print(f"Shock-finder input: {result_path}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    if list_only:
        return

    result = load_npz(result_path)
    snap_path = str(result["snap_path"])
    print(f"Snapshot input: {snap_path}", flush=True)
    snap = richio.load(snap_path)
    case = Case(
        run,
        snapnum,
        snap,
        result,
        *bh_frame_positions(snap, run, snap_path),
        per_cell_path,
        workers,
    )
    for name, path in paths.items():
        if not overwrite and is_current(name, path, run, snapnum):
            print(f"Exists {path}", flush=True)
            continue
        save_npz_atomic(path, **BUILDERS[name](case))
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    typer.run(main)
