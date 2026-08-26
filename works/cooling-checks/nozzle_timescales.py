"""Nozzle cooling-timescale calculation used by validation and time series."""

import math
import re
from pathlib import Path

import dev
import numpy as np
import richio
import unyt as u

from dev import DATAPATHS
from dev.datapaths import TDE_PARAMETERS

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def cache_complete(path, shape):
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
