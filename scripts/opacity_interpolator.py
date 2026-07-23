#!/usr/bin/env python3
"""Rosseland-mean opacity table interpolation. Thanks to Paola Martire.

Pure-numpy port of the (2-D log-log, bilinear + edge-extrapolated) table
interpolator used by RICH's grey opacity module
(``/home/hey4/RICH/source/Radiation/STAgreyOpacity.cpp``).

The table stores ``ln(sigma)`` on a log(T[K]) x log(rho[g/cm3]) grid, where
``sigma`` is used directly as a 1/length extinction coefficient (cm^-1) in
RICH's radiative transfer -- *not* a per-gram opacity (cm^2/g).  That makes
``rosseland_alpha`` below literally "alpha_ross" in the optical-depth integral
tau = int alpha_ross dr.
"""

import functools

import numpy as np
import unyt as u

TABLE_DIR = "/home/hey4/RICH/data/STA"


def bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr, y_arr):
    """Vectorized bilinear interpolation; data[i, j] <-> (x_vec[i], y_vec[j])."""
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    data = np.asarray(data)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    # Clamp to grid
    x_arr = np.clip(x_arr, x_vec[0], x_vec[-1])
    y_arr = np.clip(y_arr, y_vec[0], y_vec[-1])

    # Find indices for all points
    i = np.searchsorted(x_vec, x_arr, side="right") - 1
    j = np.searchsorted(y_vec, y_arr, side="right") - 1

    i = np.clip(i, 0, len(x_vec) - 2).astype(np.intp)
    j = np.clip(j, 0, len(y_vec) - 2).astype(np.intp)

    x0 = x_vec[i]
    x1 = x_vec[i + 1]
    y0 = y_vec[j]
    y1 = y_vec[j + 1]

    tx = (x_arr - x0) / (x1 - x0)
    ty = (y_arr - y0) / (y1 - y0)

    # Use advanced indexing to get the four corners
    d00 = data[i, j]
    d10 = data[i + 1, j]
    d01 = data[i, j + 1]
    d11 = data[i + 1, j + 1]

    result = (
        d00 * (1 - tx) * (1 - ty)
        + d10 * tx * (1 - ty)
        + d01 * (1 - tx) * ty
        + d11 * tx * ty
    )

    return result


def interpolate_2d_table_vectorized(
    x_vec, y_vec, data, x_arr, y_arr, x_vec_high_slope=0.0, slope_length=7
):
    """Vectorized 2D extrapolation-aware interpolation.

    Returns ``(interp_val, x_slope, y_slope)`` arrays.
    """
    x_vec = np.asarray(x_vec, dtype=np.float64)
    y_vec = np.asarray(y_vec, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    N = len(x_arr)
    interp_val = np.empty(N, dtype=np.float64)
    x_slope = np.empty(N, dtype=np.float64)
    y_slope = np.empty(N, dtype=np.float64)

    # Masks
    mask_x_low = x_arr < x_vec[0]
    mask_x_high = x_arr > x_vec[-1]
    mask_x_mid = ~mask_x_low & ~mask_x_high

    mask_y_low = y_arr < y_vec[0]

    # --- Region: x < x_vec[0] and y < y_vec[0] ---
    mask = mask_x_low & mask_y_low
    if np.any(mask):
        x_slope[mask] = (data[slope_length - 1, 0] - data[0, 0]) / (x_vec[slope_length - 1] - x_vec[0])
        y_slope[mask] = (data[0, slope_length - 1] - data[0, 0]) / (y_vec[slope_length - 1] - y_vec[0])
        base = data[0, 0] + y_slope[mask] * (y_arr[mask] - y_vec[0]) + x_slope[mask] * (x_arr[mask] - x_vec[0])
        interp_val[mask] = np.exp(base)

    # --- Region: x < x_vec[0] and y >= y_vec[0] ---
    mask = mask_x_low & ~mask_y_low
    if np.any(mask):
        x0 = x_vec[0] * 1.00001
        data_x0 = bilinear_interpolation_vectorized(x_vec, y_vec, data, np.full_like(y_arr[mask], x0), y_arr[mask])
        x_high = x_vec[slope_length - 1]
        data_xhigh = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, np.full_like(y_arr[mask], x_high), y_arr[mask]
        )
        x_slope[mask] = (data_xhigh - data_x0) / (x_vec[slope_length - 1] - x_vec[0])
        interp_val[mask] = np.exp(data_x0 + x_slope[mask] * (x_arr[mask] - x_vec[0]))
        y_slope[mask] = 0.0

    # --- Region: x > x_vec[-1] and y < y_vec[0] ---
    mask = mask_x_high & mask_y_low
    if np.any(mask):
        y_slope[mask] = (data[-1, slope_length - 1] - data[-1, 0]) / (y_vec[slope_length - 1] - y_vec[0])
        base = data[-1, 0] + y_slope[mask] * (y_arr[mask] - y_vec[0]) + x_vec_high_slope * (x_arr[mask] - x_vec[-1])
        interp_val[mask] = np.exp(base)
        x_slope[mask] = x_vec_high_slope

    # --- Region: x > x_vec[-1] and y >= y_vec[0] ---
    mask = mask_x_high & ~mask_y_low
    if np.any(mask):
        x_near = x_vec[-1] * 0.99999
        base = bilinear_interpolation_vectorized(x_vec, y_vec, data, np.full_like(y_arr[mask], x_near), y_arr[mask])
        interp_val[mask] = np.exp(base + x_vec_high_slope * (x_arr[mask] - x_vec[-1]))
        x_slope[mask] = x_vec_high_slope
        y_slope[mask] = 0.0

    # --- Region: x mid, y < y_vec[0] ---
    mask = mask_x_mid & mask_y_low
    if np.any(mask):
        y0 = y_vec[0] * 0.9999
        data_y0 = bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y0))
        y_high = y_vec[slope_length - 1]
        data_yhigh = bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y_high))
        y_slope[mask] = (data_yhigh - data_y0) / (y_vec[slope_length - 1] - y_vec[0])
        interp_val[mask] = np.exp(data_y0 + y_slope[mask] * (y_arr[mask] - y_vec[0]))
        x_slope[mask] = 0.0

    # --- Region: fully inside grid ---
    mask = mask_x_mid & ~mask_y_low
    if np.any(mask):
        interp_val[mask] = np.exp(
            bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], y_arr[mask])
        )
        x_slope[mask] = 0.0
        y_slope[mask] = 0.0

    return interp_val, x_slope, y_slope


def calc_scattering_opacity_vectorized(T_, rho_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False):
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)
    scatter_ = np.asarray(scatter_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]

    rho_min = rho_[0]
    rho_max = rho_[-1]
    d_ratio[mask_low] = np.exp(rhocell_arr[mask_low]) / np.exp(rho_min)
    d_log[mask_low] = rho_min
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_max)
    d_log[mask_high] = rho_max

    interp_val, T_slope, d_slope = interpolate_2d_table_vectorized(T_, rho_, scatter_, Tcell_arr, d_log)

    scatter = interp_val * d_ratio

    if return_coeff:
        # slope from interpolate_2d_table_vectorized is not used because d was shifted inside the table
        d_slope[mask_low] = 1
        d_slope[mask_high] = 1
        return scatter, T_slope, d_slope

    return scatter


def calc_ross_opacity_vectorized(T_, rho_, rossland_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False):
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]

    rho_max = rho_[-1]
    d_log[mask_high] = rho_max
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_max)

    # Interpolate Rosseland table everywhere
    interp_val, T_slope, d_slope = interpolate_2d_table_vectorized(T_, rho_, rossland_, Tcell_arr, d_log)

    rossland = interp_val * d_ratio

    # Handle rho < rho_min using scattering opacity
    if np.any(mask_low):
        scattering, Tscatt_slope, dscatt_slope = calc_scattering_opacity_vectorized(
            T_, rho_, scatter_, Tcell_arr[mask_low], rhocell_arr[mask_low], return_coeff=True
        )

        use_ross = rossland[mask_low] > scattering
        use_scatt = ~use_ross

        rossland[mask_low] = np.where(use_scatt, scattering, rossland[mask_low])
        T_slope[mask_low] = np.where(use_scatt, Tscatt_slope, T_slope[mask_low])
        d_slope[mask_low] = np.where(use_scatt, dscatt_slope, d_slope[mask_low])

        if return_coeff:
            return rossland, T_slope, d_slope

    return rossland


@functools.lru_cache(maxsize=4)
def load_opacity_table(table_dir=TABLE_DIR):
    """Load the (ln T, ln rho, ln rossland, ln scatter) STA opacity table."""
    ln_T = np.loadtxt(f"{table_dir}/T.txt")
    ln_rho = np.loadtxt(f"{table_dir}/rho.txt")
    ln_ross = np.loadtxt(f"{table_dir}/ross.txt")
    ln_scatt = np.loadtxt(f"{table_dir}/scatter.txt")
    return ln_T, ln_rho, ln_ross, ln_scatt


def rosseland_alpha(T_cgs, rho_cgs, table=None):
    """Per-cell Rosseland extinction coefficient alpha_ross.

    :param T_cgs: Temperature [K], flat array (plain float or unyt quantity).
    :param rho_cgs: Density [g/cm^3], flat array (plain float or unyt quantity,
                    same length as ``T_cgs``).
    :param table: ``(ln_T, ln_rho, ln_ross, ln_scatt)``, as returned by
                  :func:`load_opacity_table`. Loaded from :data:`TABLE_DIR` if
                  ``None``.
    :returns: ``unyt_array`` in ``cm**-1`` -- the extinction coefficient used
              directly (no division by density) as "alpha_ross" in
              tau = int alpha_ross dr. Tagged with unyt (not a bare ndarray) so
              downstream code keeps richio's dimensional-analysis unit checking
              instead of trusting a hand-written unit string.
    """
    if table is None:
        table = load_opacity_table()
    ln_T, ln_rho, ln_ross, ln_scatt = table
    T_val = np.asarray(T_cgs, dtype=np.float64)
    rho_val = np.asarray(rho_cgs, dtype=np.float64)
    sigma = calc_ross_opacity_vectorized(ln_T, ln_rho, ln_ross, ln_scatt, np.log(T_val), np.log(rho_val))
    return u.unyt_array(sigma, "cm**-1")


if __name__ == "__main__":
    # Quick sanity check against the table's own grid corners.
    ln_T, ln_rho, ln_ross, ln_scatt = load_opacity_table()
    T_cgs = np.exp([ln_T[0], ln_T[len(ln_T) // 2], ln_T[-1]])
    rho_cgs = np.exp([ln_rho[0], ln_rho[len(ln_rho) // 2], ln_rho[-1]])
    alpha = rosseland_alpha(T_cgs, rho_cgs)
    print("T [K]:", T_cgs)
    print("rho [g/cm^3]:", rho_cgs)
    print("alpha_ross [cm^-1]:", alpha)
