"""Calculations and plotting helpers for the elapsed-time cooling profiles.

The notebook keeps snapshot selection and the calculation/plot sequence. Edit
this module's uppercase settings to change tracing, sampling, or plot defaults;
notebook overrides should use the module name, e.g. ``cooling.TIME_NSTEP = 480``.

Input files
-----------
Pass a loaded ``richio`` snapshot to ``load_fields`` and use the returned
unitful field dictionary in the calculation helpers.

Output files
------------
Calculations return in-memory arrays/dictionaries. Plot helpers display
figures and return figure/axes objects.

Usage
-----
Import ``dev`` and apply the repository plotting style in the notebook before
calling the plots. Run the snapshot loop in the notebook.

Loading examples
----------------
With ``snap`` already loaded::

    import cooling_profile_helpers as cooling

    fields = cooling.load_fields(snap)
    # The notebook supplies r_p, the KD tree, and the visible diagnostic sequence.
"""

from collections.abc import Generator

import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from matplotlib.axes import Axes
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError, cKDTree

import richio

# ---------------------------------------------------------------------------- #
#                                    Configs                                   #
# ---------------------------------------------------------------------------- #

A_RAD = 4 * u.stefan_boltzmann_constant / u.c
SMOOTHING_LENGTH_RP = 0.6  # Simulation harmonic-core radius.

WORKERS = 8
NXY, NZ = 256, 512

Z_HALF_WIDTH_RP = 10
DZ_MIDPLANE = 0.1  # RICH code-length units; target spacing next to z=0.
STEAMLINE_INT_RANGE = 600 * u.s  # Stream line integral from - this to + this
SEED_MIN_RADIUS_RP = 0.7  # Exclude cells closer than this * r_p

# Common elapsed-time grid: 961 points, -600 to +600 s, spacing 1.25 s.
TIME_NSTEP = 480
MAX_LSTEP = (0.05 * u.Rsun).in_units(richio.units.lscale)

SLICE_RADIUS_RP = 3
SLICE_DENSITY_FRACTION = 1e-2  # Require rho > this fraction of the seed density.
TAU_CELL_FRACTION = 0.95  # Fraction of the actual upward tau used for N_cells.

PROFILE_FIGSIZE = (9, 4.6)
MASS_PROFILE_FIGSIZE = (8, 3.4)
TIMESCALE_PROFILE_FIGSIZE = (5.5, 4.2)
HEIGHT_PROFILE_FIGSIZE = (5.5, 3.0)
OPTICAL_DEPTH_FIGSIZE = (9, 3.6)
HEIGHT_SCALING_FIGSIZE = (9, 7.6)


# ---------------------------------------------------------------------------- #
#                                  Calculation                                 #
# ---------------------------------------------------------------------------- #


def trace_streamlines(
    pos0: u.unyt_array,
    times: u.unyt_array,
    tree: cKDTree,
    velocities: u.unyt_array,
    max_step: u.unyt_quantity,
) -> Generator[tuple[int, u.unyt_array, u.unyt_array], None, None]:
    """
    Find a group of streamlines grouped by the time.

    Sample the nearest particle's velocity using the tree of code-length
    positions. ``velocities`` is a unitful (N, 3) array in the tree's cell order.
    Yield ``(index, pos, s)`` at zero, then at positive times, then at
    negative times, restarting from the initial positions for each branch.
    ``s`` is each tracer's arc length from zero.

    The tracing method uses a midpoint sampling for the velocity. This should
    have second order accuracy in a smooth flow.

    x(n+1) = x(n) + v(x(n+1/2)) * dt
    x(n+1/2) = x(n) + v(x_n) * dt / 2

    Do a couple of substeps if the velocity is too large and causing each jump
    to be too big.
    """

    def velocity_at(x: u.unyt_array) -> NDArray[np.intp]:
        return tree.query(x.to_value("code_length"), workers=WORKERS)[1]

    i0 = np.flatnonzero(times == 0).item()
    s0 = u.unyt_array(np.zeros(len(pos0)), pos0.units)
    yield i0, pos0.copy(), s0.copy()
    for branch in (
        range(i0 + 1, len(times)),
        range(i0 - 1, -1, -1),
    ):
        pos = pos0.copy()
        s = s0.copy()
        t = 0 * times.units
        for i_t in branch:
            t_target = times[i_t]
            sign = np.sign(t_target - t)
            while sign * (t_target - t) > 0:
                dt_remaining = abs(t_target - t)
                dt = dt_remaining
                # Sample native particle velocities at the current positions.
                cell_indices = velocity_at(pos)
                v_pos = velocities[cell_indices]
                v_mag = np.max(np.linalg.norm(v_pos, axis=1))
                dt = min(dt, max_step / v_mag) if v_mag else dt
                while True:
                    pos_mid = pos + 0.5 * sign * dt * v_pos
                    # Resample velocities at the predicted halfway positions.
                    cell_indices = velocity_at(pos_mid)
                    v_mid = velocities[cell_indices]
                    v_mag_mid = np.linalg.norm(v_mid, axis=1)
                    dx = dt * v_mag_mid.max()
                    if dx <= max_step:
                        break
                    dt = dt * 0.95 * max_step / dx
                pos = pos + sign * dt * v_mid
                s = s + sign * dt * v_mag_mid
                if dt == dt_remaining:
                    t = t_target
                else:
                    t += sign * dt
            yield i_t, pos, s


def get_nozzle_slice_cells(
    fields: dict[str, u.unyt_array],
    seed_cell: int | np.intp,
    r_p: u.unyt_quantity,
) -> NDArray[np.intp]:
    """Select unique fields centers with the existing plane-proximity rule."""
    # Quantities for seed cells
    r0 = fields["position"][seed_cell]
    v = fields["velocity"][seed_cell]
    n = v / np.sqrt(np.sum(v**2))  # velocity direction vector

    # Pre-selecting a cube at pericenter to speed up compute
    Rcell = fields["volume"] ** (1 / 3) * (3 / 4 / np.pi) ** (1 / 3)
    selected = fields["density"] > SLICE_DENSITY_FRACTION * fields["density"][seed_cell]
    for axis, coord in zip(fields["position"].T, r0):
        selected &= np.abs(axis - coord) <= SLICE_RADIUS_RP * r_p + Rcell
    indices = np.flatnonzero(selected)

    # Final selection
    # \vec{r} \cdot \vec{r}_center > 0
    r_r0 = fields["position"][indices] - r0  # r - r0
    d_normal = np.sum(r_r0 * n, axis=1)  # (r - r0) dot n
    d_inplane = np.sum(r_r0**2, axis=1) - d_normal**2  # (r - r0)^2 - ((r - r0) dot n)^2
    indices = indices[
        (np.abs(d_normal) < Rcell[indices])  # in-plane
        & (
            d_inplane <= (SLICE_RADIUS_RP * r_p) ** 2
        )  # in-plane distance < SLICE_RADIUS_RP * r_p
        & (np.sum(fields["position"][indices] * r0, axis=1) > 0)  # same side of BH
    ]
    return indices


def get_seed_coords(
    fields: dict[str, u.unyt_array],
    i_seed_cells: NDArray[np.intp],
    N: int | None = None,
) -> u.unyt_array:
    """Get seed coordinates for streamline traces."""
    pos0 = fields["position"][i_seed_cells]
    print("Number of seed cells", len(pos0))
    if N is not None:
        rng = np.random.default_rng(seed=42)
        if N < len(i_seed_cells):
            pos0 = pos0[rng.choice(len(pos0), size=N, replace=False)]
        elif N > len(i_seed_cells):
            pos0 = fields["position"][i_seed_cells]
            xmin = np.min(pos0[:, 0])
            xmax = np.max(pos0[:, 0])
            ymin = np.min(pos0[:, 1])
            ymax = np.max(pos0[:, 1])
            zmin = np.min(pos0[:, 2])
            zmax = np.max(pos0[:, 2])

            pos_append = []
            Nappend = N - len(i_seed_cells)
            while len(pos_append) < Nappend:
                x = rng.random() * (xmax - xmin) + xmin
                y = rng.random() * (ymax - ymin) + ymin
                z = rng.random() * (zmax - zmin) + zmin
                coord = u.unyt_array([x, y, z], pos0.units)
                if in_hull(
                    pos0.to_value("code_length"),
                    coord.to_value("code_length"),
                ):
                    pos_append.append(coord)
                    print(f"{len(pos_append)} / {Nappend}")
                else:
                    continue

            pos0 = np.concatenate((pos0, u.unyt_array(pos_append, pos0.units)))
    return pos0


def get_seed_coords_plane(
    fields: dict[str, u.unyt_array],
    i_seed_cells: NDArray[np.intp],
    i_center_cell: int | np.intp,
    N: int | None = None,
) -> u.unyt_array:
    """Project seeds onto the center cell's velocity-normal plane.

    Keep all projected originals for N=None or their original count; smaller N
    selects without replacement. Larger N appends uniform samples from their
    2-D convex hull. Use only hull vertices in the LP and batch proposals while
    preserving their RNG order. Coordinates retain the input length units.
    """
    # Project to plane
    r0 = fields["position"][i_center_cell]
    v = fields["velocity"][i_center_cell]
    n = v / np.linalg.norm(v)
    pos0 = fields["position"][i_seed_cells]
    pos0 = pos0 - np.sum((pos0 - r0) * n, axis=1)[:, None] * n

    print("Number of seed cells", len(pos0))
    if N is None or N == len(pos0):
        return pos0

    rng = np.random.default_rng(seed=42)
    if N < len(pos0):
        return pos0[rng.choice(len(pos0), size=N, replace=False)]

    # Cross with the least-aligned coordinate axis to form an in-plane basis.
    e1 = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    e1 /= np.linalg.norm(e1)
    basis = np.stack((e1, np.cross(n, e1)))
    coords = (pos0 - r0) @ basis.T
    lower = coords.min(axis=0)
    width = np.ptp(coords, axis=0)
    hull_points = coords.to_value(pos0.units)
    try:
        # Interior points are redundant variables in the same hull LP.
        hull_points = hull_points[ConvexHull(hull_points).vertices]
    except QhullError:
        pass  # Keep the original LP for degenerate point/line hulls.

    pos_append = []
    remaining = N - len(pos0)
    while remaining:
        proposals = lower + rng.random((min(remaining, 1024), 2)) * width
        accepted = [
            in_hull(hull_points, coord) for coord in proposals.to_value(pos0.units)
        ]
        pos_append.append(r0 + proposals[accepted] @ basis)
        remaining -= sum(accepted)
    return np.concatenate([pos0, *pos_append])


# Source - https://stackoverflow.com/a/43564754
# Posted by Nils, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-27, License - CC BY-SA 3.0
def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def compute_streamline_profiles(
    fields: dict[str, u.unyt_array],
    energy_fields: dict[str, u.unyt_array],
    pos0: u.unyt_array,
    times: u.unyt_array,
    tree: cKDTree,
    max_step: u.unyt_quantity,
) -> dict[str, u.unyt_array | dict[str, u.unyt_array]]:
    """Average energies over distinct nearest native cells at each output time.

    Sample velocity and energy from the nearest native cell in the frozen
    snapshot. Each sampled cell contributes once with its mass rho V.
    Store the averaged profiles, total sampled mass, and mass flux.
    """
    # Allocate the specific-energy and energy-density profiles.
    energy_specific_profile = {
        name: u.unyt_array(np.empty(len(times)), field.units / fields["density"].units)
        for name, field in energy_fields.items()
    }
    energy_density_profile = {
        name: u.unyt_array(np.empty(len(times)), field.units)
        for name, field in energy_fields.items()
    }
    mass_profile = u.unyt_array(
        np.empty(len(times)), fields["density"].units * fields["volume"].units
    )
    mass_flux_profile = u.unyt_array(
        np.empty(len(times)), mass_profile.units / times.units
    )

    # Sample native cells and form the mass-weighted means at each time.
    for i_time, pos, s in trace_streamlines(
        pos0, times, tree, fields["velocity"], max_step
    ):
        cell_indices = tree.query(pos.to_value("code_length"), workers=WORKERS)[1]
        cell_indices = np.unique(cell_indices)
        rho = fields["density"][cell_indices]
        mass = fields["density"][cell_indices] * fields["volume"][cell_indices]
        mass_slice = np.sum(mass)
        mass_profile[i_time] = mass_slice
        mass_flux_profile[i_time] = np.sum(
            fields["density"][cell_indices]
            * np.linalg.norm(fields["velocity"][cell_indices], axis=1)
            * fields["volume"][cell_indices] ** (2 / 3)
            * (3 / 4) ** (2 / 3)
            * np.pi ** (1 / 3)
        )
        for name, field in energy_fields.items():
            energy_density_cell = field[cell_indices]
            energy_specific_profile[name][i_time] = (
                np.sum(mass * energy_density_cell / rho) / mass_slice
            )
            energy_density_profile[name][i_time] = (
                np.sum(mass * energy_density_cell) / mass_slice
            )

    # Integrate cumulative dissipated energy
    Ediss_specific_profile = u.unyt_array(
        np.zeros(len(times)), energy_specific_profile["dissipation"].units * times.units
    )
    Ediss_density_profile = u.unyt_array(
        np.zeros(len(times)), energy_density_profile["dissipation"].units * times.units
    )
    Ediss_specific_profile[1:] = np.cumsum(  # first entry is zero
        0.5
        * (
            energy_specific_profile["dissipation"][1:]
            + energy_specific_profile["dissipation"][:-1]
        )
        * np.diff(times)
    )
    Ediss_density_profile[1:] = np.cumsum(
        0.5
        * (
            energy_density_profile["dissipation"][1:]
            + energy_density_profile["dissipation"][:-1]
        )
        * np.diff(times)
    )
    return {
        "specific"         : energy_specific_profile,
        "density"          : energy_density_profile,
        "Ediss_specific"   : Ediss_specific_profile.in_units("erg/g"),
        "Ediss_density"    : Ediss_density_profile.in_units("erg/cm**3"),
        "mass"             : mass_profile,
        "mass_flux"        : mass_flux_profile
    }  # fmt:skip


def load_fields(snap: richio.Snapshot) -> dict[str, u.unyt_array]:
    """Read native fields; position and velocity are (N, 3) unyt arrays in code units."""
    return {
        "position"    : np.column_stack((snap.X, snap.Y, snap.Z)),
        "velocity"    : np.column_stack((snap.vx, snap.vy, snap.vz)),
        "density"     : snap.rho,
        "volume"      : snap.volume,
        "temperature" : snap.T,
        "sie"         : snap.sie,
        "erad"        : snap.Erad,
        "dissipation" : snap.dissipation,
        "gas_pressure": snap.pressure,
    }  # fmt:skip


def get_max_density_cell(
    fields: dict[str, u.unyt_array], r_p: u.unyt_quantity
) -> np.intp:
    """Return the densest native cell in the ROI outside the seed radius cut."""
    x, y, z = fields["position"].T
    r = np.sqrt(x**2 + y**2 + z**2)
    # Search 0 <= x <= 2 rp, |y| and |z| <= rp, with r >= SEED_MIN_RADIUS_RP * rp.
    r_box = r_p
    candidate_cells = np.flatnonzero(
        (x >= 0 * r_box)
        & (x <= 2 * r_box)
        & (np.abs(y) <= r_box)
        & (np.abs(z) <= r_box)
        & (r >= SEED_MIN_RADIUS_RP * r_box)
    )
    # Choose the highest-density cell among the candidates.
    seed_cell = candidate_cells[np.argmax(fields["density"][candidate_cells])]
    return seed_cell


def get_streamline(
    fields: dict[str, u.unyt_array], seed_cell: int | np.intp, tree: cKDTree
) -> dict[str, u.unyt_array | NDArray[np.intp]]:
    """Trace the central path in time and accumulate its signed arc length.

    Follow nearest-cell velocities with midpoint time steps on the shared
    time grid. Collapse consecutive samples of each native cell, retaining its
    center and the time of closest approach. Later returns are separate visits.
    Halfway times between changed cell IDs estimate the residence intervals.
    """
    # Trace both directions from the seed on a common time grid.
    t_grid = np.linspace(-STEAMLINE_INT_RANGE, STEAMLINE_INT_RANGE, 2 * TIME_NSTEP + 1)

    pos0 = fields["position"][seed_cell : seed_cell + 1]

    pos_grid = u.unyt_array(np.empty((len(t_grid), 3)), pos0.units)
    s_grid = u.unyt_array(np.empty(len(t_grid)), pos0.units)
    for i_time, pos, s in trace_streamlines(
        pos0, t_grid, tree, fields["velocity"], MAX_LSTEP
    ):
        pos_grid[i_time] = pos[0]
        s_grid[i_time] = s[0]
    # Group consecutive path samples that visit the same native cell.
    cell_indices = tree.query(pos_grid.to_value("code_length"), workers=WORKERS)[1]
    cell_changes = np.flatnonzero(cell_indices[1:] != cell_indices[:-1]) + 1
    visit_starts = np.r_[0, cell_changes]
    visit_stops = np.r_[cell_changes, len(cell_indices)]
    visited_cells = cell_indices[visit_starts]
    cell_centers = fields["position"][visited_cells]
    # Assign each visited cell the time of closest approach to its center.
    closest_samples = []
    for start, stop, center in zip(visit_starts, visit_stops, cell_centers):
        dr = pos_grid[start:stop] - center
        closest_samples.append(start + np.argmin(np.sum(dr**2, axis=1)))

    # Place cell-entry and cell-exit times halfway between adjacent samples.
    t = t_grid
    visit_edges = np.concatenate(
        [
            t[:1],
            0.5 * (t[cell_changes - 1] + t[cell_changes]),
            t[-1:],
        ]
    )
    return {
        "time"          : t_grid[closest_samples],
        "dt"            : np.diff(visit_edges),
        "t"             : t_grid,
        "s"             : s_grid,
        "positions"     : cell_centers,
        "indices"       : visited_cells,
        "grid_positions": pos_grid,
        "grid_indices"  : cell_indices,
    }  # fmt:skip


def compute_energy_fields(
    fields: dict[str, u.unyt_array], MBH: u.unyt_quantity, r_p: u.unyt_quantity
) -> dict[str, u.unyt_array]:
    """Compute native-cell energy densities with the simulation's harmonic core."""
    # Read the native velocity and distance from the BH.
    v2 = np.sum(fields["velocity"] ** 2, axis=1)
    r = np.sqrt(np.sum(fields["position"] ** 2, axis=1))

    # PW potential with the simulation's harmonic acceleration inside r0.
    # r_s = 2 r_g; potential and force match continuously at r0.
    r_s = 2 * u.G * MBH / u.c**2
    r0 = SMOOTHING_LENGTH_RP * r_p
    PhiPW = -u.G * MBH / (np.maximum(r, r0) - r_s)
    inner = r < r0
    PhiPW[inner] += u.G * MBH * (r[inner] ** 2 - r0**2) / (2 * r0 * (r0 - r_s) ** 2)
    # Assemble the orbital, thermal, radiation, pressure, and heating fields.
    volume_fields = {
        "orbital"           : fields["density"] * (0.5 * v2 + PhiPW),
        "radiation"         : fields["density"] * fields["erad"],
        "internal"          : fields["density"] * fields["sie"],
        "gas_pressure"      : fields["gas_pressure"],
        "radiation_pressure": fields["density"] * fields["erad"] / 3,
        "dissipation"       : fields["dissipation"],
    }  # fmt:skip
    return volume_fields


def compute_column_profiles(
    snap: richio.Snapshot,
    fields: dict[str, u.unyt_array],
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    r_p: u.unyt_quantity,
    XY: float | u.unyt_quantity,
    tree: cKDTree,
) -> dict[str, u.unyt_array | NDArray[np.intp] | dict[str, u.unyt_array]]:
    """Project on the 256 x 256 x 512 grid, then sample nearest XY columns.

    The z grid uses sinh spacing over +/- 10 rp, central spacing 0.1 code
    length, and left-endpoint sums. Optical depths average both sides. N95
    counts distinct positive-opacity cells through the upward 95% crossing.
    Photospheres use separate full-box adaptive rays.
    """
    z_half_width = Z_HALF_WIDTH_RP * r_p
    stream_positions, stream_indices = streamline["positions"], streamline["indices"]

    # Calculate radiation energy density and extinction on native cells.
    u_rad_cell = fields["density"] * fields["erad"]
    alpha_ross_cell = richio.opacity.rosseland_alpha(
        fields["temperature"], fields["density"]
    )
    alpha_planck_cell = richio.opacity.planck_alpha(
        fields["temperature"], fields["density"]
    )

    # Resample onto a grid uniform in x/y and sinh-spaced in z.
    box = [-XY, -XY, -z_half_width, XY, XY, z_half_width]
    cell_grid, x_grid, y_grid, z_grid = snap.to_3dgrid(
        res=(NXY, NXY, NZ),
        box_size=box,
        workers=WORKERS,
        endpoint=False,
        spacing=("linear", "linear", "sinh"),
        central_spacing=(None, None, DZ_MIDPLANE),
    )

    # Evaluate each integration interval at its left endpoint.
    volume_bins = np.s_[:-1, :-1, :-1]
    rho = fields["density"][cell_grid][volume_bins]
    T = fields["temperature"][cell_grid][volume_bins]
    vz = fields["velocity"][:, 2][cell_grid][volume_bins]
    eps_int = fields["sie"][cell_grid][volume_bins]
    alpha_ross = alpha_ross_cell[cell_grid][volume_bins]
    alpha_planck = alpha_planck_cell[cell_grid][volume_bins]

    dz = z_grid[1:] - z_grid[:-1]
    z = np.abs(z_grid[:-1])

    # Column density and density-weighted height and vertical speed.
    Sigma = np.sum(rho * dz, axis=-1)
    H = np.sum(rho * z * dz, axis=-1) / Sigma
    vz_mean = np.sum(rho * np.abs(vz) * dz, axis=-1) / Sigma
    t_vertical = H / vz_mean

    # Divide column internal energy by its integrated emissivity.
    t_cool = np.sum(rho * eps_int * dz, axis=-1) / np.sum(
        alpha_planck * A_RAD * T**4 * u.c * dz, axis=-1
    )

    # Average the two one-sided optical depths and estimate H tau / c.
    tau_full = np.sum(alpha_ross * dz, axis=-1)
    tau = 0.5 * tau_full  # Mean of the upper and lower optical depths.
    t_escape = H * tau / u.c
    stream_xy = stream_positions[:, :2]
    # Match each streamline cell to the nearest column in x_grid/y_grid[:-1].
    pixel_x = np.abs(stream_xy[:, 0, None] - x_grid[None, :-1]).argmin(axis=1)
    pixel_y = np.abs(stream_xy[:, 1, None] - y_grid[None, :-1]).argmin(axis=1)
    column_indices = cell_grid[pixel_x, pixel_y, :-1]
    column_tau = tau[pixel_x, pixel_y].in_units("dimensionless")
    # Pair the column density proxy Sigma/H with the native streamline density.
    H_column = H[pixel_x, pixel_y]
    Sigma_column = Sigma[pixel_x, pixel_y].in_units("g/cm**2")
    rho_column = (Sigma_column / H_column).in_units("g/cm**3")
    rho_stream = fields["density"][stream_indices].in_units("g/cm**3")
    # Integrate upward from z=0 to locate the 95% optical-depth crossing.
    dz_upper = np.diff(np.maximum(z_grid, 0 * z_grid.units))
    upper = dz_upper > 0 * dz_upper.units
    upper_indices = column_indices[:, upper]
    dtau_upper = (alpha_ross_cell[upper_indices] * dz_upper[upper]).in_units(
        "dimensionless"
    )
    tau_from_midplane = np.cumsum(dtau_upper, axis=1)
    tau_upper = tau_from_midplane[:, -1]
    crossing_bin = np.argmax(
        tau_from_midplane >= TAU_CELL_FRACTION * tau_upper[:, None], axis=1
    )
    N95 = []
    for ids, contributions, stop in zip(upper_indices, dtau_upper, crossing_bin):
        # Count distinct positive-opacity cells up to and including the crossing.
        contributing_ids = ids[: stop + 1][contributions[: stop + 1] > 0]
        N95.append(np.unique(contributing_ids).size)
    N95 = np.array(N95)
    eps_rad_stream = fields["erad"][stream_indices].in_units("erg/g")
    # Density-weighted specific radiation energy: integral(rho eps_rad dz) / Sigma.
    eps_rad_column = (
        np.sum(u_rad_cell[column_indices] * dz, axis=1) / Sigma_column
    ).in_units("erg/g")
    columns = {
        "cooling_time"   : t_cool[pixel_x, pixel_y],
        "vertical_time"  : t_vertical[pixel_x, pixel_y],
        "escape_time"    : t_escape[pixel_x, pixel_y],
        "tau"            : column_tau,
        "height"         : H_column,
        "surface_density": Sigma_column,
        "density_proxy"  : rho_column,
        "cell_density"   : rho_stream,
        "cell_erad"      : eps_rad_stream,
        "column_erad"    : eps_rad_column,
        "cell_count95"   : N95,
    }  # fmt:skip
    column_xy = np.column_stack([x_grid[pixel_x], y_grid[pixel_y]])
    columns.update(
        compute_photosphere_profiles(fields, alpha_ross_cell, column_xy, snap.box, tree)
    )
    return columns


def compute_photosphere_profiles(
    fields: dict[str, u.unyt_array],
    alpha: u.unyt_array,
    column_xy: u.unyt_array,
    box: u.unyt_array,
    tree: cKDTree,
) -> dict[str, u.unyt_array | dict[str, u.unyt_array]]:
    """Trace full-box observer rays at the same XY positions as the H columns.

    The photosphere search samples nearest-cell alpha and integrates with
    adaptive trapezoids from the snapshot boundary. The interior optical-depth
    integral uses quarter-cell steps from the midplane to each photosphere.
    """
    # Trace each distinct XY column once and retain its unitful coordinates.
    _, unique_indices, column_indices = np.unique(
        column_xy, axis=0, return_index=True, return_inverse=True
    )
    xy = column_xy[unique_indices]
    midplane = u.unyt_array(np.zeros((len(xy), 3)), column_xy.units)
    midplane[:, :2] = xy
    midplane_points = np.concatenate([midplane, midplane])
    # Directions are inward ray travel: observer +z -> -z, observer -z -> +z.
    directions = np.repeat([[0, 0, -1], [0, 0, 1]], len(xy), axis=0)
    cell_positions = fields["position"]  # Same code_length coordinates as the tree.
    photospheres = richio.analysis.find_photosphere(
        cell_positions,
        alpha,
        fields["volume"],
        box,
        midplane_points,
        directions,
        tree=tree,
    )
    # Integrate from z=0 to each photosphere when they bracket the midplane.
    z_plus, z_minus = np.split(photospheres["position"][:, 2], 2)
    encloses_midplane = (z_plus >= 0 * cell_positions.units) & (
        z_minus <= 0 * cell_positions.units
    )
    ray_mask = np.tile(encloses_midplane, 2)
    tau_to_ph = u.unyt_array(np.full(len(midplane_points), np.nan), "dimensionless")
    if np.any(ray_mask):
        column_integral = richio.analysis.integrate_optical_depth(
            cell_positions,
            alpha,
            fields["volume"],
            midplane_points[ray_mask],
            photospheres["position"][ray_mask],
            max_delta_tau=None,
            tree=tree,
        )
        tau_to_ph[ray_mask] = column_integral["tau"]
    # Average both observer directions and restore the original column order.
    tau_plus, tau_minus = np.split(tau_to_ph, 2)
    tau_ph = 0.5 * (tau_plus + tau_minus)
    H_ph = 0.5 * (np.abs(z_plus) + np.abs(z_minus))
    return {
        "photosphere_z": {"+z": z_plus[column_indices], "-z": z_minus[column_indices]},
        "photosphere_height": H_ph[column_indices],
        "photosphere_tau_midplane": tau_ph[column_indices],
        "escape_time_photosphere": (H_ph * tau_ph / u.c)[column_indices].in_units("s"),
    }


def compute_single_cell_profiles(
    fields: dict[str, u.unyt_array],
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    energy_fields: dict[str, u.unyt_array],
) -> dict[str, u.unyt_array | dict[str, u.unyt_array]]:
    """Sample native energies and accumulate constant-per-visit heating at sample times."""
    stream_indices, dt_cell = streamline["indices"], streamline["dt"]

    # Pair volumetric energy with specific energy at each visited cell.
    energy_density = {
        name: field[stream_indices] for name, field in energy_fields.items()
    }
    specific_energy = {
        name: field[stream_indices] / fields["density"][stream_indices]
        for name, field in energy_fields.items()
    }
    # Locate visit entries and the elapsed part of each visit at the plotted time.
    t_entry = streamline["t"][0] + np.concatenate(
        [0 * dt_cell[:1], np.cumsum(dt_cell[:-1])]
    )
    dt_to_sample = streamline["time"] - t_entry

    # Sum completed visits, then add the current visit up to its sample time.
    dEdiss_density = energy_density["dissipation"] * dt_cell
    Ediss_density = (
        np.concatenate([0 * dEdiss_density[:1], np.cumsum(dEdiss_density[:-1])])
        + energy_density["dissipation"] * dt_to_sample
    )
    dEdiss_specific = specific_energy["dissipation"] * dt_cell
    Ediss_specific = (
        np.concatenate([0 * dEdiss_specific[:1], np.cumsum(dEdiss_specific[:-1])])
        + specific_energy["dissipation"] * dt_to_sample
    )

    # Set the cumulative energy to zero at the first plotted sample.
    Ediss_density = Ediss_density - Ediss_density[0]
    Ediss_specific = Ediss_specific - Ediss_specific[0]
    return {
        "density"          : energy_density,
        "specific"         : specific_energy,
        "Ediss_density"    : Ediss_density.in_units("erg/cm**3"),
        "Ediss_specific"   : Ediss_specific.in_units("erg/g"),
    }  # fmt:skip


def compute_slice_profiles(
    fields: dict[str, u.unyt_array],
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    energy_fields: dict[str, u.unyt_array],
    r_p: u.unyt_quantity,
    i_center_cell: int | np.intp,
    tree: cKDTree,
    Nseed: int | None = None,
) -> dict[str, u.unyt_array | dict[str, u.unyt_array]]:
    """Follow trajectories launched from one perpendicular layer.

    Apply aperture, hemisphere and density cuts only at launch. At common
    elapsed times, average energies over distinct nearest cells with mass rho V.
    Heating uses trapezoids of mean(q/rho) or mean(q) in time.
    """
    # Launch trajectories from the selected cells in the central plane.
    i_seed_cells = get_nozzle_slice_cells(fields, i_center_cell, r_p)
    pos0 = get_seed_coords_plane(fields, i_seed_cells, i_center_cell, Nseed)
    return compute_streamline_profiles(
        fields,
        energy_fields,
        pos0,
        streamline["t"],
        tree,
        MAX_LSTEP,
    )


def add_distance_axis(
    ax: Axes, streamline: dict[str, u.unyt_array | NDArray[np.intp]]
) -> SecondaryAxis:
    """Label time by signed central arc length, with a piecewise-linear mapping."""
    # Build the t <-> s mapping using numerical coordinates for Matplotlib.
    time = streamline["t"].to_value("s")
    distance = streamline["s"].to_value("Rsun")
    time_to_distance = interp1d(
        time, distance, bounds_error=False, fill_value="extrapolate", assume_sorted=True
    )
    distance_to_time = interp1d(
        distance, time, bounds_error=False, fill_value="extrapolate", assume_sorted=True
    )
    # Use the top axis for distance and the bottom axis for time.
    ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    top = ax.secondary_xaxis("top", functions=(time_to_distance, distance_to_time))
    top.set_xlabel(r"$s\ [R_\odot]$")
    # Place the y multiplier above the top ticks, clear of both axis labels.
    offset = ax.yaxis.get_offset_text()
    offset.set_x(0)
    offset.set_horizontalalignment("left")
    offset.set_transform(
        offset.get_transform()
        + ScaledTranslation(0, 16 / 72, ax.figure.dpi_scale_trans)
    )
    return top


# ---------------------------------------------------------------------------- #
#                                   Plotting                                   #
# ---------------------------------------------------------------------------- #


def plot_timescales(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    columns: dict[str, u.unyt_array | NDArray[np.intp] | dict[str, u.unyt_array]],
    t_dyn: u.unyt_quantity,
) -> tuple[Figure, Axes]:
    """Plot three column timescales and the photosphere escape time in stellar dynamical-time units."""
    stream_time = streamline["time"]
    stream_marker = "o" if len(stream_time) == 1 else None
    fig, timescale_ax = plt.subplots(
        figsize=TIMESCALE_PROFILE_FIGSIZE, constrained_layout=True
    )
    # Compare each column timescale with the stellar dynamical time.
    for t_over_tdyn, label in [
        (columns["cooling_time"] / t_dyn, r"$t_c$"),
        (columns["vertical_time"] / t_dyn, r"$t_v$"),
        (columns["escape_time"] / t_dyn, r"$t_\mathrm{es}$"),
        (columns["escape_time_photosphere"] / t_dyn, r"$t_{\mathrm{es,ph}}$"),
    ]:
        # Keep repeated pixel matches so the profile retains the particle ordering.
        timescale_ax.semilogy(
            stream_time,
            t_over_tdyn,
            nonpositive="mask",
            label=label,
            marker=stream_marker,
            markersize=3,
        )
    timescale_ax.axhline(
        1,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][3],
        linestyle="--",
    )
    timescale_ax.set_xlim(streamline["t"][0], streamline["t"][-1])
    timescale_ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    timescale_ax.set_ylabel(r"$t/t_\mathrm{dyn}$")
    timescale_ax.legend(ncols=2)
    plt.show()
    plt.close(fig)
    return fig, timescale_ax


def plot_single_cell_energy_density(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    single: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot cell energy densities, their changes, Bernoulli density, and heating."""
    # Read the native-cell energy and pressure profiles along the streamline.
    stream_time = streamline["time"]
    stream_marker = "o" if len(stream_time) == 1 else None
    energy_fields = single["density"]
    orbital_energy = energy_fields["orbital"]
    radiation_energy = energy_fields["radiation"]
    internal_energy = energy_fields["internal"]
    gas_pressure = energy_fields["gas_pressure"]
    radiation_pressure = energy_fields["radiation_pressure"]
    Ediss = single["Ediss_density"]
    fig, (ax, delta_ax) = plt.subplots(
        1, 2, figsize=PROFILE_FIGSIZE, sharex=True, constrained_layout=True
    )
    profile_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Plot energy magnitudes; distinguish positive and negative orbital energy.
    for field, label, linestyle, color_index in [
        (orbital_energy, r"$|u_\mathrm{orb}|$", "-", 0),
        (radiation_energy, r"$u_\mathrm{rad}$", "-", 1),
        (internal_energy, r"$u_\mathrm{int}$", "-", 3),
        (gas_pressure, r"$P_\mathrm{gas}$", "-", 4),
        (radiation_pressure, r"$P_\mathrm{rad}$", "-", 6),
    ]:
        if field is None:
            continue
        energy_profile = field.in_units("erg/cm**3")
        for sign in (-1, 1) if color_index == 0 else (0,):
            plotted = energy_profile.copy()
            sign_label = ""
            if sign:
                plotted[(energy_profile > 0) != (sign > 0)] = np.nan
                plotted = np.abs(plotted)
                linestyle = "--" if sign > 0 else "-"
                sign_label = r" ($+$)" if sign > 0 else r" ($-$)"
            ax.semilogy(
                stream_time,
                plotted,
                nonpositive="mask",
                linestyle=linestyle,
                color=profile_colors[color_index],
                label=label + sign_label,
                marker=stream_marker,
                markersize=3,
            )
    ax.set_xlim(streamline["t"][0], streamline["t"][-1])
    ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    ax.set_ylabel(r"Energy density $[\mathrm{erg\,cm^{-3}}]$")
    # Changes from the first retained upstream particle; orbital energy is signed.
    for field, label, color_index in [
        (orbital_energy, r"$\Delta u_\mathrm{orb}$", 0),
        (radiation_energy, r"$\Delta u_\mathrm{rad}$", 1),
        (internal_energy, r"$\Delta u_\mathrm{int}$", 3),
        (gas_pressure, r"$\Delta P_\mathrm{gas}$", 4),
        (radiation_pressure, r"$\Delta P_\mathrm{rad}$", 6),
    ]:
        if field is None:
            continue
        energy_profile = field.in_units("erg/cm**3")
        for sign in (-1, 1) if color_index == 0 else (0,):
            change = energy_profile - energy_profile[0]
            linestyle, sign_label = "-", ""
            if sign:
                change[(energy_profile > 0) != (sign > 0)] = np.nan
                linestyle = "--" if sign > 0 else "-"
                sign_label = r" ($+$)" if sign > 0 else r" ($-$)"
            delta_ax.plot(
                stream_time,
                change,
                color=profile_colors[color_index],
                linestyle=linestyle,
                label=label + sign_label,
                marker=stream_marker,
                markersize=3,
            )
    # Bernoulli density: rho B = u_orb + u_int + u_rad + P_gas + P_rad.
    bernoulli_density = (
        orbital_energy
        + internal_energy
        + radiation_energy
        + gas_pressure
        + radiation_pressure
    ).in_units("erg/cm**3")
    for sign in (-1, 1):
        opposite_sign = (bernoulli_density > 0) != (sign > 0)
        magnitude = np.abs(bernoulli_density)
        change = bernoulli_density - bernoulli_density[0]
        magnitude[opposite_sign] = np.nan
        change[opposite_sign] = np.nan
        linestyle = "--" if sign > 0 else "-"
        sign_label = r" ($+$)" if sign > 0 else r" ($-$)"
        ax.semilogy(
            stream_time,
            magnitude,
            color=profile_colors[7],
            linestyle=linestyle,
            nonpositive="mask",
            label=r"$|\rho B|$" + sign_label,
        )
        delta_ax.plot(
            stream_time,
            change,
            color=profile_colors[7],
            linestyle=linestyle,
            label=r"$\Delta(\rho B)$" + sign_label,
        )
    # Overlay the energy accumulated from the first plotted time.
    for profile_ax in (ax, delta_ax):
        profile_ax.plot(
            stream_time,
            Ediss,
            color=profile_colors[5],
            linestyle=":",
            label=r"$E_\mathrm{diss}$",
        )
        add_distance_axis(profile_ax, streamline)
        profile_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    # Show energy gains and losses on a symmetric scale.
    delta_limit = max(abs(limit) for limit in delta_ax.get_ylim())
    delta_ax.set_ylim(-delta_limit, delta_limit)
    delta_ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    delta_ax.set_ylabel(r"Energy density change $[\mathrm{erg\,cm^{-3}}]$")
    plt.show()
    plt.close(fig)
    return fig, (ax, delta_ax)


def plot_energy_profile(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    time_plot: u.unyt_array,
    profiles: dict[str, u.unyt_array],
    Ediss: u.unyt_array,
    energy_unit: str,
    mass_flux: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot energies, changes from the first point, and dissipation estimates."""
    # Express all energy components and cumulative heating in the plot units.
    time_limits = streamline["t"][[0, -1]]
    profile_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    energy = {
        name: profiles[name].in_units(energy_unit)
        for name in (
            "orbital",
            "radiation",
            "internal",
            "gas_pressure",
            "radiation_pressure",
        )
    }
    Ediss = Ediss.in_units(energy_unit)
    if energy_unit == "erg/g":
        symbol = r"\varepsilon"
        ylabel = "Specific energy"
        unit_label = r"\mathrm{erg\,g^{-1}}"
    else:
        symbol = "u"
        ylabel = "Energy density"
        unit_label = r"\mathrm{erg\,cm^{-3}}"
    if mass_flux:
        ylabel = "Mass-flux mean\nspecific energy"
    marker = "o" if len(time_plot) == 1 else None
    fig, (energy_ax, change_ax) = plt.subplots(
        1, 2, figsize=PROFILE_FIGSIZE, sharex=True, constrained_layout=True
    )
    # Pair each energy profile with its change from the first upstream sample.
    for name, subscript, color_index in [
        ("orbital", "orb", 0),
        ("radiation", "rad", 1),
        ("internal", "int", 3),
        ("gas_pressure", "gas", 4),
        ("radiation_pressure", "rad", 6),
    ]:
        energy_profile = energy[name]
        label = rf"{symbol}_\mathrm{{{subscript}}}"
        if name.endswith("_pressure"):
            label = rf"P_\mathrm{{{subscript}}}"
            if energy_unit == "erg/g":
                label = rf"({label}/\rho)"
        # Style by the signed orbital energy, including in the change panel.
        for sign in (-1, 1) if name == "orbital" else (0,):
            plotted = (
                np.abs(energy_profile) if name == "orbital" else energy_profile.copy()
            )
            change = energy_profile - energy_profile[0]
            linestyle, sign_label = "-", ""
            if sign:
                opposite_sign = (energy_profile > 0) != (sign > 0)
                plotted[opposite_sign] = np.nan
                change[opposite_sign] = np.nan
                linestyle = "--" if sign > 0 else "-"
                sign_label = r" ($+$)" if sign > 0 else r" ($-$)"
            energy_ax.semilogy(
                time_plot,
                plotted,
                nonpositive="mask",
                color=profile_colors[color_index],
                linestyle=linestyle,
                label=(rf"$|{label}|$" if name == "orbital" else rf"${label}$")
                + sign_label,
                marker=marker,
                markersize=3,
            )
            change_ax.plot(
                time_plot,
                change,
                color=profile_colors[color_index],
                linestyle=linestyle,
                label=rf"$\Delta {label}$" + sign_label,
                marker=marker,
                markersize=3,
            )
    # Sum the signed energy and pressure-work terms to form B or rho B.
    bernoulli = (
        energy["orbital"]
        + energy["internal"]
        + energy["radiation"]
        + energy["gas_pressure"]
        + energy["radiation_pressure"]
    )
    bernoulli_label = "B" if energy_unit == "erg/g" else r"\rho B"
    for sign in (-1, 1):
        opposite_sign = (bernoulli > 0) != (sign > 0)
        magnitude = np.abs(bernoulli)
        change = bernoulli - bernoulli[0]
        magnitude[opposite_sign] = np.nan
        change[opposite_sign] = np.nan
        linestyle = "--" if sign > 0 else "-"
        sign_label = r" ($+$)" if sign > 0 else r" ($-$)"
        energy_ax.semilogy(
            time_plot,
            magnitude,
            color=profile_colors[7],
            linestyle=linestyle,
            nonpositive="mask",
            label=rf"$|{bernoulli_label}|$" + sign_label,
        )
        change_ax.plot(
            time_plot,
            change,
            color=profile_colors[7],
            linestyle=linestyle,
            label=rf"$\Delta({bernoulli_label})$" + sign_label,
        )
    for profile_ax in (energy_ax, change_ax):
        profile_ax.plot(
            time_plot,
            Ediss,
            color=profile_colors[5],
            linestyle=":",
            label=r"$E_\mathrm{diss}$",
        )
        profile_ax.set_xlim(*time_limits)
        profile_ax.set_xlabel(r"$t\ [\mathrm{s}]$")
        add_distance_axis(profile_ax, streamline)
        profile_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    energy_ax.set_ylabel(rf"{ylabel} $[{unit_label}]$")
    change_ax.set_ylabel(rf"{ylabel} change $[{unit_label}]$")
    # Show energy gains and losses on a symmetric scale.
    change_limit = max(abs(limit) for limit in change_ax.get_ylim())
    change_ax.set_ylim(-change_limit, change_limit)
    plt.show()
    plt.close(fig)
    return fig, (energy_ax, change_ax)


def plot_single_cell_specific_energy(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    single: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot native-cell energies per gram along the central streamline."""
    return plot_energy_profile(
        streamline,
        streamline["time"],
        single["specific"],
        single["Ediss_specific"],
        "erg/g",
    )


def plot_slice_specific_energy(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    slices: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Fixed initial-mass mean of specific energies along the advected cohort."""
    return plot_energy_profile(
        streamline,
        streamline["t"],
        slices["specific"],
        slices["Ediss_specific"],
        "erg/g",
    )


def plot_slice_energy_density(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    slices: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Fixed initial-mass mean of energy densities along the advected cohort."""
    return plot_energy_profile(
        streamline,
        streamline["t"],
        slices["density"],
        slices["Ediss_density"],
        "erg/cm**3",
    )


def plot_slice_mass_and_flux(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    slices: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot sampled-cell mass and mass-flux estimate against time and arc length.

    The profiles sum unique nearest native cells at each tracer output time:
    M = sum(rho V) and Mdot = sum(rho |v| pi R_cell**2), where
    R_cell = (3 V / (4 pi))**(1/3). The latter is a positive cell-cross-section
    estimate, not a signed flux through one oriented surface. The upper axes
    use the central trajectory's piecewise-linear time-to-distance mapping.
    """
    time = streamline["t"]
    mass = slices["mass"].in_units("Msun")
    mass_flux = slices["mass_flux"].in_units("Msun/yr")
    fig, (mass_ax, flux_ax) = plt.subplots(
        1, 2, figsize=MASS_PROFILE_FIGSIZE, sharex=True, constrained_layout=True
    )
    mass_ax.plot(time, mass)
    flux_ax.plot(time, mass_flux)
    mass_ax.set_ylabel(r"$M\ [M_\odot]$")
    flux_ax.set_ylabel(r"$\dot M\ [M_\odot\,\mathrm{yr}^{-1}]$")
    for ax in (mass_ax, flux_ax):
        ax.set_xlim(time[0], time[-1])
        ax.set_xlabel(r"$t\ [\mathrm{s}]$")
        add_distance_axis(ax, streamline)
    plt.show()
    plt.close(fig)
    return fig, (mass_ax, flux_ax)


def plot_flux_specific_energy(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    flux: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Advected-surface sum(U v dot dA) / net Mdot, retaining signed orientation."""
    return plot_energy_profile(
        streamline,
        streamline["t"],
        flux["specific"],
        flux["Ediss_specific"],
        "erg/g",
        mass_flux=True,
    )


def plot_mass_flux(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    flux: dict[str, u.unyt_array | dict[str, u.unyt_array]],
) -> tuple[Figure, Axes]:
    """Compare signed lab-frame flux through the advected surface with its forward part."""
    grid_time = streamline["t"]
    mdot_net, mdot_forward = flux["net"], flux["forward"]
    profile_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, mdot_ax = plt.subplots(
        figsize=TIMESCALE_PROFILE_FIGSIZE, constrained_layout=True
    )
    mdot_ax.plot(grid_time, mdot_net, color=profile_colors[7], label="Net")
    mdot_ax.plot(grid_time, mdot_forward, linestyle="--", label="Forward")
    mdot_ax.set_xlim(streamline["t"][0], streamline["t"][-1])
    mdot_ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    mdot_ax.set_ylabel(r"$\dot M\ [M_\odot\,\mathrm{yr}^{-1}]$")
    mdot_ax.legend()
    plt.show()
    plt.close(fig)
    return fig, mdot_ax


def plot_streamline_height(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
) -> tuple[Figure, Axes]:
    """Compare heights of the traced path and sampled native-cell centers."""
    grid_time, stream_time = streamline["t"], streamline["time"]
    grid_positions, stream_positions = (
        streamline["grid_positions"],
        streamline["positions"],
    )
    profile_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Express traced and native-cell heights above the midplane in solar radii.
    path_z = grid_positions[:, 2].in_units("Rsun")
    cell_z = stream_positions[:, 2].in_units("Rsun")
    fig, height_ax = plt.subplots(
        figsize=HEIGHT_PROFILE_FIGSIZE, constrained_layout=True
    )
    height_ax.plot(grid_time, path_z, label="Streamline")
    height_ax.plot(stream_time, cell_z, linestyle="--", label="Cell centers")
    height_ax.axhline(0, color=profile_colors[3], linestyle=":")
    height_ax.set_xlim(streamline["t"][0], streamline["t"][-1])
    height_ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    height_ax.set_ylabel(r"$z\ [R_\odot]$")
    height_ax.legend(ncols=2)
    plt.show()
    plt.close(fig)
    return fig, height_ax


def plot_optical_depth_diagnostics(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    columns: dict[str, u.unyt_array | NDArray[np.intp] | dict[str, u.unyt_array]],
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot tau versus time with H/Hph, and tau versus the upward N95 count."""
    stream_time = streamline["time"]
    column_tau = columns["tau"]
    column_height = columns["height"].in_units("Rsun")
    photosphere_height = columns["photosphere_height"].in_units("Rsun")
    column_cell_count95 = columns["cell_count95"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        1, 2, figsize=OPTICAL_DEPTH_FIGSIZE, constrained_layout=True
    )
    # Follow optical depth, scale height, and photosphere height along the flow.
    height_ax = axes[0].twinx()
    axes[0].semilogy(stream_time, column_tau, color=colors[0])
    height_ax.plot(
        stream_time, column_height, color=colors[1], linestyle="--", label=r"$H$"
    )
    height_ax.plot(
        stream_time, photosphere_height, color=colors[2], label=r"$H_{\rm ph}$"
    )
    height_ax.set_yscale("log")
    height_ax.legend(loc="upper right", ncols=2)
    axes[0].set_xlim(streamline["t"][0], streamline["t"][-1])
    axes[0].set_xlabel(r"$t\ [\mathrm{s}]$")
    axes[0].set_ylabel(r"$\tau$")
    height_ax.set_ylabel(r"$H,\ H_{\rm ph}\ [R_\odot]$")

    # Compare tau with the number of cells supplying 95% of the upward tau.
    axes[1].scatter(column_cell_count95, column_tau, s=12, color=colors[0])
    axes[1].set_yscale("log")
    axes[1].set_xlabel(rf"$N_{{{100 * TAU_CELL_FRACTION:g}\%,+}}$ (midplane upward)")
    axes[1].set_ylabel(r"$\tau$")

    plt.show()
    plt.close(fig)
    return fig, axes


def plot_height_scalings(
    streamline: dict[str, u.unyt_array | NDArray[np.intp]],
    columns: dict[str, u.unyt_array | NDArray[np.intp] | dict[str, u.unyt_array]],
    run: str,
    snapshot_number: int,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot radiation-H, density-H and radiation-density fits with equal visit weights."""
    stream_time = streamline["time"]
    column_height = columns["height"].in_units("Rsun")
    column_density_proxy = columns["density_proxy"]
    cell_density = columns["cell_density"]
    cell_erad, column_erad = columns["cell_erad"], columns["column_erad"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        2, 2, figsize=HEIGHT_SCALING_FIGSIZE, constrained_layout=True
    )
    axes = axes.ravel()
    # Fit epsilon_rad versus H separately upstream and downstream of t=0.
    # Each native-cell visit has equal weight in the log-log fit.
    log_height = np.log10(column_height.to_value("Rsun"))
    fit_height = column_height[[column_height.argmin(), column_height.argmax()]]
    seed_mask = stream_time == 0 * u.s
    for radiation_profile, sample_label, linestyle, marker in [
        (cell_erad, "cell", "-", "o"),
        (column_erad, "column", "--", "x"),
    ]:
        log_erad = np.log10(radiation_profile.to_value("erg/g"))
        for branch_label, branch_mask, color in [
            ("t<0", stream_time < 0 * u.s, colors[0]),
            ("t>0", stream_time > 0 * u.s, colors[1]),
        ]:
            branch_height = column_height[branch_mask]
            branch_log_height = log_height[branch_mask]
            branch_log_erad = log_erad[branch_mask]
            branch_power, branch_log_norm = np.polyfit(
                branch_log_height, branch_log_erad, 1
            )
            branch_normalization = u.unyt_quantity(10**branch_log_norm, "erg/g")
            branch_fit_height = branch_height[
                [branch_height.argmin(), branch_height.argmax()]
            ]
            branch_fit_erad = (
                branch_normalization
                * (branch_fit_height / (1 * u.Rsun)) ** branch_power
            )
            axes[0].scatter(
                branch_height,
                radiation_profile[branch_mask],
                s=12,
                color=color,
                marker=marker,
            )
            axes[0].plot(
                branch_fit_height,
                branch_fit_erad,
                color=color,
                linestyle=linestyle,
                label=rf"${branch_label}$, {sample_label}: $H^{{{branch_power:.2f}}}$",
            )
        axes[0].scatter(
            column_height[seed_mask],
            radiation_profile[seed_mask],
            marker=marker,
            s=24,
            color=colors[3],
            label=r"$t=0$" if sample_label == "cell" else None,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncols=2,
        fontsize=10,
        columnspacing=1,
        handlelength=1.8,
        borderaxespad=0,
    )
    axes[0].set_xlabel(r"$H\ [R_\odot]$")
    axes[0].set_ylabel(r"$\epsilon_{\rm rad}\ [\mathrm{erg\,g^{-1}}]$")

    # Fit cell density and the column proxy Sigma/H against scale height.
    for density_profile, density_label, color in [
        (column_density_proxy, r"\Sigma/H", colors[4]),
        (cell_density, r"\rho_{\rm cell}", colors[0]),
    ]:
        log_density = np.log10(density_profile.to_value("g/cm**3"))
        density_power, log_density_norm = np.polyfit(log_height, log_density, 1)
        density_normalization = u.unyt_quantity(10**log_density_norm, "g/cm**3")
        fit_density = (
            density_normalization * (fit_height / (1 * u.Rsun)) ** density_power
        )
        axes[1].scatter(column_height, density_profile, s=12, color=color)
        axes[1].plot(
            fit_height,
            fit_density,
            color=color,
            linestyle="--",
            label=rf"${density_label}\propto H^{{{density_power:.2f}}}$",
        )
    axes[1].sharex(axes[0])
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$H\ [R_\odot]$")
    axes[1].set_ylabel(r"Density $[\mathrm{g\,cm^{-3}}]$")
    axes[1].legend()

    # Pair radiation energy with density from the same cell or vertical column.
    for (
        ax,
        density_profile,
        radiation_profile,
        density_label,
        sample_label,
        linestyle,
        marker,
    ) in [
        (axes[2], cell_density, cell_erad, r"\rho_{\rm cell}", "cell", "-", "o"),
        (axes[3], column_density_proxy, column_erad, r"\Sigma/H", "column", "--", "x"),
    ]:
        log_density = np.log10(density_profile.to_value("g/cm**3"))
        log_radiation = np.log10(radiation_profile.to_value("erg/g"))
        for branch_label, branch_mask, color in [
            ("t<0", stream_time < 0 * u.s, colors[0]),
            ("t>0", stream_time > 0 * u.s, colors[1]),
        ]:
            power, log_norm = np.polyfit(
                log_density[branch_mask], log_radiation[branch_mask], 1
            )
            branch_density = density_profile[branch_mask]
            fit_density = branch_density[
                [branch_density.argmin(), branch_density.argmax()]
            ]
            fit_radiation = (
                u.unyt_quantity(10**log_norm, "erg/g")
                * (fit_density / (1 * u.g / u.cm**3)) ** power
            )
            ax.scatter(
                branch_density,
                radiation_profile[branch_mask],
                s=12,
                color=color,
                marker=marker,
            )
            ax.plot(
                fit_density,
                fit_radiation,
                color=color,
                linestyle=linestyle,
                label=rf"${branch_label}:\ \epsilon_{{\rm rad}}\propto({density_label})^{{{power:.2f}}}$",
            )
        ax.scatter(
            density_profile[seed_mask],
            radiation_profile[seed_mask],
            marker=marker,
            s=24,
            color=colors[3],
            label=r"$t=0$",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(rf"${density_label}\ [\mathrm{{g\,cm^{{-3}}}}]$")
        ax.set_ylabel(
            rf"$\epsilon_{{\rm rad,{sample_label}}}\ [\mathrm{{erg\,g^{{-1}}}}]$"
        )
        ax.legend(fontsize=10)

    plt.show()
    plt.close(fig)
    return fig, axes
