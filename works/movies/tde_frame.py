#!/usr/bin/env python3
"""TDE reference-frame correction for movies (kept out of the richio library).

The simulation runs in the **star frame** (star at the origin) until the
returning debris triggers a switch to the **black-hole frame** at a known
snapshot; thereafter snapshots are stored in the BH frame.  To make a continuous
movie we transform the *pre-switch* snapshots into the BH frame by adding the
star's orbital position/velocity, exactly as the simulation does in
``UpdateReferenceFrame`` (see the run's main.cpp):

    x_BH = x_star + x0[0],  y_BH = y_star + x0[1]   (z unchanged)
    vx  += x0[2],           vy  += x0[3]

where ``x0 = (x, y, vx, vy)`` is the star's state on a **Paczynski–Wiita
parabolic orbit** integrated from pericenter to the snapshot time.

Usage (in a movie driver, before calling ``evolution_movie``)::

    import richio, tde_frame
    richio.load = tde_frame.make_bh_frame_loader(m_bh=1e4, m_star=0.5,
                                                 r_star=0.47, beta=1.0,
                                                 switch_snap=21)

Forked render workers inherit the monkeypatched ``richio.load``.  This is a
deliberately minimal, scripts-only shim; a proper TDE submodule may live in
richio later.
"""

import numpy as np


def _paczynski_x0(t, m_bh, rp, rg):
    """Star state ``[x, y, vx, vy]`` at time *t* on the PW parabolic orbit.

    Integrated from pericenter ``(rp, 0)`` with velocity ``(0, -vp)`` where
    ``vp = sqrt(2 (M/(rp-rg)))`` — matching ``GetTrueAnomaly`` in main.cpp
    (G = 1, code units).
    """
    from scipy.integrate import solve_ivp

    vp = np.sqrt(2.0 * (m_bh / (rp - rg)))
    if t == 0:
        return np.array([rp, 0.0, 0.0, -vp])

    def rhs(_t, x):
        r = np.hypot(x[0], x[1])
        f = m_bh / (r * (r - rg) ** 2)
        return [x[2], x[3], -x[0] * f, -x[1] * f]

    sol = solve_ivp(
        rhs,
        [0.0, t],
        [rp, 0.0, 0.0, -vp],
        rtol=1e-9,
        atol=1e-7,
        max_step=abs(t) / 200 + 1e-6,
    )
    return sol.y[:, -1]


def star_orbit_x0(t, m_bh=1e4, m_star=0.5, r_star=0.47, beta=1.0):
    """``[x, y, vx, vy]`` (code units) of the star at code-time *t*."""
    rt = r_star * (m_bh / m_star) ** (1.0 / 3.0)
    rp = rt / beta
    rg = 4.21 * m_bh / 1e6
    return _paczynski_x0(float(t), m_bh, rp, rg)


# Canonical position/velocity fields shifted into the BH frame.
_SHIFT_X = {"CMx", "X"}
_SHIFT_Y = {"CMy", "Y"}
_SHIFT_VX = {"Vx"}
_SHIFT_VY = {"Vy"}


class _BHFrameSnapshot:
    """Wrap a snapshot, shifting position/velocity fields into the BH frame.

    Delegates everything to the real snapshot except :meth:`_get_data`, which
    adds the orbital offset to the in-plane position and velocity fields.  z and
    vz are unchanged (orbit is in the x–y plane).
    """

    def __init__(self, snap, x0):
        self._snap = snap
        self._x0 = x0  # [dx, dy, dvx, dvy] in code units

    def __getattr__(self, name):
        return getattr(self._snap, name)

    def _get_data(self, data):
        arr = self._snap._get_data(data)
        if not isinstance(data, str):
            return arr
        canon = self._snap._resolve_field_name(data)
        dx, dy, dvx, dvy = self._x0
        if canon in _SHIFT_X:
            return arr + dx * arr.units
        if canon in _SHIFT_Y:
            return arr + dy * arr.units
        if canon in _SHIFT_VX:
            return arr + dvx * arr.units
        if canon in _SHIFT_VY:
            return arr + dvy * arr.units
        return arr


def select_unbound_outflow(
    snap,
    *,
    zr_max=None,
    x_sign=0,
    m_bh=1e4,
    m_star=0.5,
    r_star=0.47,
    coords=("CMx", "CMy", "CMz"),
):
    """Boolean cell mask: **unbound** (``B > 0``) and **radially outflowing** (``v_r > 0``).

    ``zr_max`` adds an **equatorial-wedge** cut ``|z|/r <= zr_max`` — e.g.
    ``0.5`` keeps gas with polar angle ``pi/3 <= theta <= 2 pi/3`` (within 30 deg
    of the xy plane), the disc-plane wind.  ``x_sign`` optionally restricts to one
    side of the BH along x (``+1`` pericenter side, ``-1`` far side, ``0`` off).

    Operates on whatever snapshot it is handed — under the BH-frame loader that is
    the frame-corrected snapshot, so the selection is in the BH frame like the
    rest of the movie.  Reuses :func:`richio.render.derived.bernoulli` for ``B``;
    ``v_r`` is ``(v·r)/|r|``.  Returned as a plain boolean ``ndarray`` ``(N,)``.
    """
    from richio.render.derived import bernoulli

    def cu(name):
        return np.asarray(snap._get_data(name).in_base("rich"), dtype="float64")

    b = np.asarray(
        bernoulli(snap, m_bh=m_bh, m_star=m_star, r_star=r_star, coords=coords),
        dtype="float64",
    )
    x, y, z = cu(coords[0]), cu(coords[1]), cu(coords[2])
    vx, vy, vz = cu("Vx"), cu("Vy"), cu("Vz")
    r = np.sqrt(x * x + y * y + z * z)
    r = np.where(r > 0, r, 1.0)
    vr = (x * vx + y * vy + z * vz) / r

    mask = (b > 0) & (vr > 0)
    if zr_max is not None:
        mask &= np.abs(z) <= zr_max * r
    if x_sign > 0:
        mask &= x > 0
    elif x_sign < 0:
        mask &= x < 0
    return mask


def make_bh_frame_loader(
    m_bh=1e4, m_star=0.5, r_star=0.47, beta=1.0, switch_snap=21, orig_load=None
):
    """Return a ``load(path)`` that puts pre-switch snapshots in the BH frame.

    Snapshots with ``snapnum < switch_snap`` are shifted by the star's orbital
    state at the snapshot time; later snapshots are returned unchanged (already
    in the BH frame).
    """
    if orig_load is None:
        import richio

        orig_load = richio.load

    def load(path):
        snap = orig_load(path)
        num = getattr(snap, "snapnum", -1)
        if num < 0 or num >= switch_snap:
            return snap
        t = getattr(snap, "time", None)
        if t is None:
            return snap  # no time → cannot place on the orbit; leave as-is
        tcode = float(np.atleast_1d(t)[0])  # snapshots here store code time
        x0 = star_orbit_x0(tcode, m_bh, m_star, r_star, beta)
        return _BHFrameSnapshot(snap, x0)

    return load
