"""Shared calculation and cache reader for nozzle cooling maps.

Integrate a nearest-cell Cartesian grid along z inside spherical radius
``r < 3 r_p``, after correcting moving-frame coordinates to the BH frame.
Let ``Sigma = integral(rho dz)``, ``H = integral(rho |z| dz)/Sigma`` and
``vzbar = integral(rho |vz| dz)/Sigma``. The times are emission
``tc = integral(rho sie dz)/integral(alpha_P a_rad T**4 c dz)``, vertical flow
``tv = H/vzbar``, diffusion ``tdiff = H tau_R/c``, and photon escape
``tesc = H (1 + tau_R)/c``, with ``tau_R = integral(alpha_R dz)``.
Here emission is gross thermal emission, not net radiative energy exchange.
The ratio ``max(tc, tesc)/tv`` is a diagnostic, not evidence of a causal cooling
change. The accepted wedge has projected radius ``0.6 <= R/r_p <= 1.75`` and
angular half-width 4.5 degrees about the supplied/native-peak direction.

Input files
-----------
This is a helper module with no CLI and no files produced on import. The two
entry scripts are ``nozzle-timescale-validation.py`` and
``nozzle-timescale-series.py`` in this directory. ``snapshot_path(run, snapnum)``
resolves an existing ``snap_full_<n>.h5`` or ``snap_<n>.h5`` through
``dev.datapaths.DATAPATHS``. Required ``richio`` snapshot quantities are time,
X/Y/Z, rho, T, vz, sie and dissipation; opacity comes from ``richio.opacity``.
``config_for(run)`` returns a dictionary with ``run`` (str), and ``m_bh``,
``m_star``, ``r_star``, ``r_p``, ``t_fb`` (numbers in code mass, length and time
units), using ``dev.datapaths.TDE_PARAMETERS``.

``load_direction(run, snapnum, root)`` reads
``<root>/<run>/directions/direction_snap_<NNNN>.npz`` from
``nozzle-wedge-validation.py`` and returns an ``(x, y, z)`` tuple of floats in
pericentre units. It selects ``direction_peak_{x,y,z}_rp_dirmin_0p6`` for
``1e4``/``1e5`` or ``..._0p8`` for ``1e6``. The default stage-1 direction root
used by the validation CLI is
``data/processed/CoolingChecks/nozzle-timescale-series/stage1-wedge-selection``.

Output files
------------
``calculate_snapshot(path, destination, config, shape, workers, direction=None,
z_spacing="linear", sinh_scale_rp=0.1)`` returns None and atomically replaces the
explicit ``destination`` NPZ. ``shape=(Nx, Ny, Nz)`` counts grid points;
``direction`` is the tuple above. If omitted, select the native maximum of
volumetric dissipation in ``0.6 <= r/r_p <= 1.75`` (inner radius 0.8 for ``1e6``).

Each map is a compressed NumPy ``.npz`` archive loaded with ``np.load``;
there is no single rectangular table. ``Nx``, ``Ny``, ``Nz`` are grid-point
counts. Maps have axis order ``(x, y)``, with z integrated out; plot ``map.T``
against the one-dimensional x/y coordinates. Values are linear, not logarithms.
Empty columns can yield NaN or infinity in divisions; mask non-finite values
for statistics/plots. Zero integrated density/dissipation outside the aperture
is retained. The keys written by the current calculation are:

``run``, ``snapshot_path`` : Unicode arrays, shape ``()``
    Mass label (``1e4``, ``1e5`` or ``1e6``) and source HDF5 path. Use ``.item()``
    to retrieve a Python string.
``snapnum`` : int64 array, shape ``()``
    Source snapshot number.
``resolution_x``, ``resolution_y``, ``resolution_z`` : int64 arrays, shape ``()``
    ``Nx``, ``Ny``, ``Nz``; these describe the sampling grid, not native cells.
``z_spacing`` : Unicode array, shape ``()``
    ``linear`` or ``sinh``. The z integration uses ``Nz - 1`` left samples and
    the differences between consecutive z coordinates.
``sinh_scale_rp`` : float64 array, shape ``()``
    Sinh scale divided by pericentre radius; NaN for linear spacing.
``time_tfb``, ``time_days`` : float64 arrays, shape ``()``
    Snapshot time divided by fallback time, and snapshot time in days.
``x_rp``, ``y_rp`` : float64 arrays, shapes ``(Nx,)``, ``(Ny,)``
    BH-frame x/y sampling coordinates divided by pericentre radius.
``wedge_mask`` : bool array, shape ``(Nx, Ny)``
    True for wedge pixels with positive integrated dissipation. The other maps
    cover the full aperture; apply this mask when selecting nozzle statistics.
``dissipation_column_erg_s_cm2`` : float64 array, shape ``(Nx, Ny)``
    ``integral(dissipation dz)`` in erg/s/cm**2; this is power per projected
    area, not pixel power. Multiply by pixel area in cm**2 before summing power.
``sigma_g_cm2`` : float64 array, shape ``(Nx, Ny)``
    Surface density ``Sigma`` in g/cm**2.
``H_Rstar`` : float64 array, shape ``(Nx, Ny)``
    Density-weighted absolute height ``H`` divided by stellar radius.
``vzbar_cm_s`` : float64 array, shape ``(Nx, Ny)``
    Density-weighted absolute vertical velocity in cm/s.
``tau_R`` : float64 array, shape ``(Nx, Ny)``
    Dimensionless Rosseland optical depth integrated through the aperture.
``tc_tdyn``, ``tv_tdyn``, ``tdiff_tdyn``, ``tesc_tdyn`` : float64 arrays, shape ``(Nx, Ny)``
    The four times defined above divided by stellar dynamical time
    ``sqrt(R_star**3/(G M_star))``; all dimensionless.
``tc_over_tv``, ``tdiff_over_tv``, ``tesc_over_tv`` : float64 arrays, shape ``(Nx, Ny)``
    Emission, diffusion and escape time divided by vertical-flow time.
``effective_over_tv`` : float64 array, shape ``(Nx, Ny)``
    ``max(tc, tesc)/tv``, dimensionless.

Previously produced caches may additionally contain ``resolution``, physical
scales, direction metadata or dimensional time/height maps. These are optional
legacy keys and are not written by the current calculation; the keys above are
the current contract. Old cubic caches can use ``resolution`` instead of
``resolution_x``, ``resolution_y``, ``resolution_z``, and omit z-spacing metadata.
The cache reader treats those as linear cubic grids.

``summarize_cache(path)`` returns a list of three dictionaries, one for each
``statistic``: ``median``, ``dissipation_weighted_mean`` and
``max_dissipation_pixel``. The median ignores NaNs; the weighted mean does not
filter NaNs or infinities. Weights are positive column dissipation in the wedge;
the peak pixel is the largest column dissipation there. Each dictionary has:

``run``, ``z_spacing``, ``statistic`` : str
    Mass label, vertical grid spacing and aggregation method.
``snapnum``, ``resolution``, ``resolution_x``, ``resolution_y``, ``resolution_z`` : int
    Snapshot and sampling-grid sizes; ``resolution`` repeats ``resolution_x``.
``sinh_scale_rp``, ``time_tfb``, ``time_days`` : float
    Same units/meaning as the NPZ scalar metadata; linear-grid sinh scale is NaN.
``selected_pixels`` : int
    Number of True wedge pixels.
``captured_total_dissipation_fraction`` : float
    Wedge dissipation sum divided by the sum over all positive map pixels.
``max_dissipation_x_rp``, ``max_dissipation_y_rp`` : float
    Coordinates of the maximum-dissipation wedge pixel in pericentre units.
``sigma_g_cm2``, ``H_Rstar``, ``vzbar_cm_s``, ``tau_R`` : float
    Statistic of each map, in g/cm**2, stellar radii, cm/s and dimensionless
    optical depth, respectively.
``tc_tdyn``, ``tv_tdyn``, ``tdiff_tdyn``, ``tesc_tdyn`` : float
    Statistic of each dimensionless time map.
``tc_over_tv``, ``tdiff_over_tv``, ``tesc_over_tv``, ``effective_over_tv`` : float
    Statistic of each dimensionless ratio map. These are statistics of ratios,
    not ratios computed from separately aggregated times.

Usage
-----
Import from the repository root (``/home/hey4/rich_tde``) in the richanalysis
environment. Use a persistent study directory for generated maps::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("dev").resolve()))
    sys.path.insert(0, str(Path("works/cooling-checks").resolve()))
    import nozzle_timescales as nt

    source = nt.snapshot_path("1e4", 108)
    destination = Path("data/processed/CoolingChecks/custom/1e4/snap_0108.npz")
    # This performs a full snapshot calculation; choose a cluster job as needed.
    nt.calculate_snapshot(source, destination, nt.config_for("1e4"),
                          (256, 256, 512), workers=8, z_spacing="sinh")

The helper overwrites its explicit destination. ``cache_complete(path, shape)``
returns bool and checks required field names and grid-shape metadata, not source
contents, direction or sinh scale. CLI callers own skip/overwrite decisions;
use a separate output location when those inputs change. ``atomic_npz`` stages
files next to their final destination.

Loading examples
----------------
Read the persisted map, select a nozzle diagnostic and plot with x horizontal::

    from pathlib import Path
    import numpy as np
    import dev
    import matplotlib.pyplot as plt

    path = Path("data/processed/CoolingChecks/custom/1e4/snap_0108.npz")
    with np.load(path) as data:
        run = data["run"].item()
        x, y = data["x_rp"], data["y_rp"]
        ratio = data["effective_over_tv"]
        mask = data["wedge_mask"] & np.isfinite(ratio)
    print(run, np.median(ratio[mask]))
    fig, ax = plt.subplots()
    image = ax.pcolormesh(x, y, np.where(mask, ratio, np.nan).T,
                         shading="nearest")
    ax.set(xlabel="x/r_p", ylabel="y/r_p", aspect="equal")
    fig.colorbar(image, ax=ax, label="max(tc, tesc)/tv")
    plt.show()
"""

import math
import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

import numpy as np
import unyt as u
from dev.datapaths import TDE_PARAMETERS

import dev
import richio
from dev import DATAPATHS

DIRECTION_RADIUS_MIN_RP = {"1e4": 0.6, "1e5": 0.6, "1e6": 0.8}
APERTURE_RP = 3.0
WEDGE_RADIUS_RP = (0.6, 1.75)
WEDGE_HALF_WIDTH_DEG = 4.5
A_RAD = (4 * u.stefan_boltzmann_constant / u.c).to_value("erg/cm**3/K**4")
C = u.c.to_value("cm/s")
G = u.G.to_value("cm**3/g/s**2")
FIELDS = (
    "dissipation_column_erg_s_cm2",
    "sigma_g_cm2",
    "H_Rstar",
    "vzbar_cm_s",
    "tau_R",
    "tc_tdyn",
    "tv_tdyn",
    "tdiff_tdyn",
    "tesc_tdyn",
    "tc_over_tv",
    "tdiff_over_tv",
    "tesc_over_tv",
    "effective_over_tv",
)
STATISTICS = ("median", "dissipation_weighted_mean", "max_dissipation_pixel")


def config_for(run):
    """Return stellar/BH parameters, pericentre and fallback time in code units."""
    m_bh, m_star, r_star = TDE_PARAMETERS[run]
    return {
        "run": run,
        "m_bh": m_bh,
        "m_star": m_star,
        "r_star": r_star,
        "r_p": r_star * (m_bh / m_star) ** (1 / 3),
        "t_fb": math.pi
        / math.sqrt(2)
        * math.sqrt(r_star**3 / m_star)
        * math.sqrt(m_bh / m_star),
    }


def scalar_time(snapshot):
    value = float(np.asarray(snapshot.time.to_value("code_time")).squeeze())
    return u.unyt_quantity(value, "code_time", registry=snapshot.time.units.registry)


def snapshot_path(run, snapnum):
    snapnums, paths = DATAPATHS(run)
    return Path(paths[snapnums.index(snapnum)])


def coordinates(snapshot, path, config):
    x, y, z = snapshot.X, snapshot.Y, snapshot.Z
    plain = re.fullmatch(r"snap_\d+\.h5", path.name)
    if path.parent.name == "TEMPTDE" or (config["run"] != "1e6" and plain):
        offset = dev.reference_frame_offset(
            t=scalar_time(snapshot),
            Mbh=config["m_bh"] * richio.units.mscale,
            Mstar=config["m_star"] * richio.units.mscale,
            Rstar=config["r_star"] * richio.units.lscale,
            beta=1,
        )
        x, y = x + offset[0], y + offset[1]
    return x, y, z


def load_direction(run, snapnum, root):
    radius = str(DIRECTION_RADIUS_MIN_RP[run]).replace(".", "p")
    path = root / run / "directions" / f"direction_snap_{snapnum:04d}.npz"
    with np.load(path) as data:
        return tuple(
            float(data[f"direction_peak_{a}_rp_dirmin_{radius}"]) for a in "xyz"
        )


def atomic_npz(path, arrays):
    """Replace an NPZ using a sibling staging file; never stage in system scratch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def cache_complete(path, shape):
    """Check required fields and grid shape; source content is not fingerprinted."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            if "resolution_x" in data:
                saved_shape = tuple(int(data[f"resolution_{a}"]) for a in "xyz")
            else:
                saved_shape = (int(data["resolution"]),) * 3
            return saved_shape == shape and all(field in data for field in FIELDS)
    except (KeyError, OSError, ValueError):
        return False


def calculate_snapshot(
    path,
    destination,
    config,
    shape,
    workers,
    direction=None,
    z_spacing="linear",
    sinh_scale_rp=0.1,
):
    """Integrate one spherical aperture and overwrite destination with physical maps.

    All lengths in ``direction`` are in r_p units. A missing direction selects
    the native-cell dissipation maximum. See the module docstring for formulas,
    stored fields and the distinction between emission and net cooling.
    """
    snapshot = richio.load(str(path))
    time = scalar_time(snapshot)
    x, y, z = coordinates(snapshot, path, config)
    r_p = config["r_p"] * richio.units.lscale
    r_star = config["r_star"] * richio.units.lscale
    m_star = config["m_star"] * richio.units.mscale
    r_p_cm = r_p.to_value("cm")
    r_star_cm = r_star.to_value("cm")
    t_dyn = math.sqrt(r_star_cm**3 / (G * m_star.to_value("g")))

    radius = np.sqrt(x**2 + y**2 + z**2)
    selection = np.asarray(radius < APERTURE_RP * r_p)
    if direction is None:
        shell = (radius >= DIRECTION_RADIUS_MIN_RP[config["run"]] * r_p) & (
            radius <= WEDGE_RADIUS_RP[1] * r_p
        )
        indices = np.flatnonzero(shell)
        peak = indices[np.argmax(snapshot.dissipation[indices])]
        direction = tuple(float(array[peak] / r_p) for array in (x, y, z))

    bound = APERTURE_RP * r_p
    spacing = ("linear", "linear", z_spacing)
    scale = (None, None, sinh_scale_rp * r_p) if z_spacing == "sinh" else None
    grid, xspace, yspace, zspace = snapshot.to_3dgrid(
        res=shape,
        X=x,
        Y=y,
        Z=z,
        box_size=(-bound, -bound, -bound, bound, bound, bound),
        selection=selection,
        workers=workers,
        spacing=spacing,
        sinh_scale=scale,
    )

    nx, ny, _ = shape
    dz = (zspace[1:] - zspace[:-1]).to_value("cm")
    abs_z = np.abs(zspace[:-1].to_value("cm"))
    sigma, rho_z, rho_vz, tau, energy, emission, dissipation = (
        np.zeros((nx, ny)) for _ in range(7)
    )
    for start in range(0, nx, 32):
        stop = min(start + 32, nx)
        idx = grid[start:stop, :, :-1]
        xx = xspace[start:stop].to_value("cm")[:, None, None]
        yy = yspace.to_value("cm")[None, :, None]
        zz = zspace[:-1].to_value("cm")[None, None, :]
        inside = xx**2 + yy**2 + zz**2 < (APERTURE_RP * r_p_cm) ** 2
        rho = snapshot.rho[idx].to("g/cm**3")
        temperature = snapshot.T[idx].to("K")
        slab_shape = temperature.shape
        alpha_R = (
            richio.opacity.rosseland_alpha(temperature.reshape(-1), rho.reshape(-1))
            .to_value("cm**-1")
            .reshape(slab_shape)
        )
        alpha_P = (
            richio.opacity.planck_alpha(temperature.reshape(-1), rho.reshape(-1))
            .to_value("cm**-1")
            .reshape(slab_shape)
        )
        rho = np.where(inside, rho.to_value("g/cm**3"), 0)
        temperature = temperature.to_value("K")
        vz = np.abs(snapshot.vz[idx].to_value("cm/s"))
        sie = snapshot.sie[idx].to_value("erg/g")
        diss = snapshot.dissipation[idx].to_value("erg/s/cm**3")
        widths = dz[None, None, :]
        target = np.s_[start:stop]
        sigma[target] = np.sum(rho * widths, axis=-1)
        rho_z[target] = np.sum(rho * abs_z[None, None, :] * widths, axis=-1)
        rho_vz[target] = np.sum(rho * vz * widths, axis=-1)
        tau[target] = np.sum(np.where(inside, alpha_R, 0) * widths, axis=-1)
        energy[target] = np.sum(rho * sie * widths, axis=-1)
        emission[target] = np.sum(
            np.where(inside, alpha_P * A_RAD * temperature**4 * C, 0) * widths,
            axis=-1,
        )
        dissipation[target] = np.sum(np.where(inside, diss, 0) * widths, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        H = rho_z / sigma
        vz = rho_vz / sigma
        tc = energy / emission
        tv = H / vz
        tdiff = H * tau / C
        tesc = H * (1 + tau) / C

    x_rp, y_rp = np.asarray(xspace / r_p), np.asarray(yspace / r_p)
    xgrid, ygrid = np.meshgrid(x_rp, y_rp, indexing="ij")
    angle0 = math.atan2(direction[1], direction[0])
    angle = (np.arctan2(ygrid, xgrid) - angle0 + np.pi) % (2 * np.pi) - np.pi
    radius_xy = np.hypot(xgrid, ygrid)
    wedge = (
        (radius_xy >= WEDGE_RADIUS_RP[0])
        & (radius_xy <= WEDGE_RADIUS_RP[1])
        & (np.abs(angle) <= math.radians(WEDGE_HALF_WIDTH_DEG))
        & (dissipation > 0)
    )
    atomic_npz(
        destination,
        {
            "run": config["run"],
            "snapshot_path": str(path),
            "snapnum": int(re.search(r"(\d+)\.h5$", path.name).group(1)),
            "resolution_x": shape[0],
            "resolution_y": shape[1],
            "resolution_z": shape[2],
            "z_spacing": z_spacing,
            "sinh_scale_rp": sinh_scale_rp if z_spacing == "sinh" else np.nan,
            "time_tfb": time.to_value("code_time") / config["t_fb"],
            "time_days": time.to_value("day"),
            "x_rp": x_rp,
            "y_rp": y_rp,
            "wedge_mask": wedge,
            "dissipation_column_erg_s_cm2": dissipation,
            "sigma_g_cm2": sigma,
            "H_Rstar": H / r_star_cm,
            "vzbar_cm_s": vz,
            "tau_R": tau,
            "tc_tdyn": tc / t_dyn,
            "tv_tdyn": tv / t_dyn,
            "tdiff_tdyn": tdiff / t_dyn,
            "tesc_tdyn": tesc / t_dyn,
            "tc_over_tv": tc / tv,
            "tdiff_over_tv": tdiff / tv,
            "tesc_over_tv": tesc / tv,
            "effective_over_tv": np.maximum(tc, tesc) / tv,
        },
    )


def summarize_cache(path):
    """Return median, dissipation-weighted and peak-pixel statistics in the wedge."""
    with np.load(path) as data:
        wedge = data["wedge_mask"].astype(bool)
        weights = data["dissipation_column_erg_s_cm2"]
        if "resolution_x" in data:
            resolution = tuple(int(data[f"resolution_{a}"]) for a in "xyz")
        else:
            resolution = (int(data["resolution"]),) * 3
        peak = np.unravel_index(
            np.nanargmax(np.where(wedge, weights, np.nan)), wedge.shape
        )
        common = {
            "run": str(data["run"]),
            "snapnum": int(data["snapnum"]),
            "resolution": resolution[0],
            "resolution_x": resolution[0],
            "resolution_y": resolution[1],
            "resolution_z": resolution[2],
            "z_spacing": str(data["z_spacing"]) if "z_spacing" in data else "linear",
            "sinh_scale_rp": float(data["sinh_scale_rp"])
            if "sinh_scale_rp" in data
            else float("nan"),
            "time_tfb": float(data["time_tfb"]),
            "time_days": float(data["time_days"]),
            "selected_pixels": int(wedge.sum()),
            "captured_total_dissipation_fraction": float(
                weights[wedge].sum() / weights[weights > 0].sum()
            ),
            "max_dissipation_x_rp": float(data["x_rp"][peak[0]]),
            "max_dissipation_y_rp": float(data["y_rp"][peak[1]]),
        }
        rows = []
        for statistic in STATISTICS:
            row = {**common, "statistic": statistic}
            for field in FIELDS[1:]:
                values = data[field]
                if statistic == "median":
                    value = np.nanmedian(values[wedge])
                elif statistic == "dissipation_weighted_mean":
                    value = np.average(values[wedge], weights=weights[wedge])
                else:
                    value = values[peak]
                row[field] = float(value)
            rows.append(row)
    return rows
