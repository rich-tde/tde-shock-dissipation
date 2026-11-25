"""Plotting pieces shared by the presentation notebooks and scripts."""

from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.signal import savgol_filter

MACH_THRESHOLDS = (2.0, 7.0, 15.0, 25.0)
MACH_LABELS = ("<2", "2–7", "7–15", "15–25", "≥25")
MACH_CMAP = ListedColormap(("#0072B2", "#009E73", "#F0E442", "#E69F00", "#D55E00")).with_extremes(
    bad="0.65"
)
MACH_NORM = BoundaryNorm(np.arange(-0.5, len(MACH_LABELS)), MACH_CMAP.N)


def positive_log10(values):
    """log10 of the positive finite values as a plain float array, NaN elsewhere."""
    values = np.asarray(values, dtype="float64")
    return np.log10(np.where(np.isfinite(values) & (values > 0), values, np.nan))


def mach_classes(mach, shocks_only=True):
    """Index of the Mach class of each value, masked where there is no shock."""
    empty = ~np.isfinite(mach)
    if shocks_only:
        empty |= mach <= 0
    return np.ma.array(np.digitize(np.nan_to_num(mach), MACH_THRESHOLDS), mask=empty)


def log_limits(logged_values, decades):
    """Half-dex rounded limits of log10 data, at most ``decades`` wide."""
    finite = logged_values[np.isfinite(logged_values)]
    low, high = float(finite.min()), float(finite.max())
    if high - low < 3:
        return low, high
    top = np.ceil(2 * high) / 2
    return max(np.floor(2 * low) / 2, top - decades), top


def colorbar_above(fig, ax, image, label):
    """Thin horizontal colorbar sitting just above ``ax``."""
    cax = inset_axes(
        ax,
        width="100%",
        height="6%",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.085, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label(label, fontsize=10, labelpad=4)
    cax.tick_params(which="both", labelsize=9, length=2, pad=1)
    cax.xaxis.set_label_position("top")
    if image.norm is MACH_NORM:
        bar.set_ticks(np.arange(len(MACH_LABELS)), labels=MACH_LABELS)
    return bar


def savgol_smooth(values, window_length, polyorder=2, log=False):
    """Savitzky-Golay filter over the finite points (positive ones in log10 if log); the rest become NaN."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & ((values > 0) if log else True)
    window_length = min(window_length, (valid.sum() - 1) // 2 * 2 + 1)
    smoothed = np.full_like(values, np.nan)
    if window_length <= polyorder:
        smoothed[valid] = values[valid]
        return smoothed
    fitted = savgol_filter(
        np.log10(values[valid]) if log else values[valid], window_length, polyorder
    )
    smoothed[valid] = 10**fitted if log else fitted
    return smoothed
