#!/usr/bin/env python3
"""Geometry helpers for anisotropic ("pencil beam") boxes such as preset C.

A close-up of the pericentre region is **not** a smaller cube.  It keeps the wide
boxes' full line-of-sight (z) extent and only narrows x and y, so a face-on
projection integrates *exactly the same column* as the wide views.  The close-up
is then a true magnification — same physical values, so the wide colour limits
carry over — and a Rosseland projection stays a real optical depth instead of a
partial one through a truncated slab.

Two consequences need handling, and that is all this module does:

* **Framing.** ``richio.render.yt_backend._camera_vectors`` sets the camera width
  to ``max(domain extent) / zoom``.  For a pencil beam the largest extent is the
  line of sight, which would frame the beam hundreds of times too wide, so the
  requested zoom has to be scaled up — see :func:`camera_zoom_for_box`.
* **Cost.** Building the k-d tree from all ~57 M cells to fill a beam that holds
  ~12 % of them is wasted single-threaded work — see :func:`box_selection`.

Isotropic boxes (A, B) pass through both functions unchanged, so the same code
path serves every preset.
"""

def tidal_radius(m_bh=1e4, m_star=0.5, r_star=0.47):
    """Tidal radius ``r_t = R_* (M_BH / M_*)^(1/3)`` in code length (R_sun)."""
    return r_star * (m_bh / m_star) ** (1.0 / 3.0)


def pericentre_radius(m_bh=1e4, m_star=0.5, r_star=0.47, beta=1.0):
    """Pericentre distance ``r_p = r_t / beta`` in code length (R_sun)."""
    return tidal_radius(m_bh, m_star, r_star) / float(beta)


def box_extent(box):
    """``(dx, dy, dz)`` of a ``[x0, y0, z0, x1, y1, z1]`` box."""
    return (box[3] - box[0], box[4] - box[1], box[5] - box[2])


def is_pencil(box, ratio=2.0):
    """True when *box* is much longer along z than across, i.e. a beam not a cube.

    Used to decide whether the framing correction and the cell pre-selection are
    needed; a cube gives ``False`` and everything behaves as it always has.
    """
    dx, dy, dz = box_extent(box)
    return dz > ratio * max(dx, dy)


def camera_zoom_for_box(box, zoom):
    """Scale *zoom* so the camera frames the box's **transverse** extent.

    yt frames ``max(extent) / zoom``.  That is the intended behaviour for a cube,
    where the transverse extent *is* the largest one, so this returns *zoom*
    unchanged for presets A and B.  For a pencil beam the largest extent is the
    line of sight, so the zoom is scaled by ``max(extent) / max(dx, dy)`` — which
    is what makes preset C show a few ``r_p`` across instead of a few thousand.

    :param box: ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param zoom: The requested zoom (1.1 for the standard slight crop).
    :returns: The zoom to hand to ``volume_image(zoom=...)``.
    """
    dx, dy, dz = box_extent(box)
    return float(zoom) * max(dx, dy, dz) / max(dx, dy)


def box_selection(snap, coords, box, pad_frac=0.25):
    """Boolean mask of cells within the box's transverse footprint (plus margin).

    Passed to ``to_3dgrid(selection=...)`` so a narrow box builds its k-d tree
    from the cells that can matter instead of all ~57 M.  The tree build is
    single-threaded and dominates each frame, so for preset C this is what keeps
    the close-up from costing as much as a full-box render.

    The margin matters: a grid point just inside the edge may have its true
    nearest cell just *outside*, and cropping without a margin would snap it to
    the wrong cell.  ``pad_frac`` of the half-width is orders of magnitude larger
    than any cell in this region, so the mask cannot change the result.

    Returns ``None`` for a cube, meaning "use every cell" — the wide presets keep
    their existing nearest-neighbour behaviour exactly.

    :param snap: Snapshot to read coordinates from.
    :param coords: ``(x, y, z)`` field names, e.g. ``("CMx", "CMy", "CMz")``.
    :param box: ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param pad_frac: Margin as a fraction of each transverse half-width.
    :returns: Boolean array of shape ``(N,)``, or ``None``.
    """
    if not is_pencil(box):
        return None

    from richio import units

    dx, dy, _ = box_extent(box)
    x_lo, x_hi = box[0] - pad_frac * dx / 2, box[3] + pad_frac * dx / 2
    y_lo, y_hi = box[1] - pad_frac * dy / 2, box[4] + pad_frac * dy / 2
    x = snap._get_data(coords[0])
    y = snap._get_data(coords[1])
    return (
        (x >= x_lo * units.lscale)
        & (x <= x_hi * units.lscale)
        & (y >= y_lo * units.lscale)
        & (y <= y_hi * units.lscale)
    )


def scalebar_in_rp(box, zoom, r_p, target_frac=0.2):
    """Scale bar (fraction-of-width, label) measured in ``r_p`` rather than ``r_t``.

    Mirrors :func:`render_evolution._scalebar_for_box` — same rounding, same
    field-of-view convention — but labels the bar in pericentre radii, the natural
    ruler once the view is only a few ``r_p`` across.

    :param box: ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param zoom: The **effective** camera zoom (see :func:`camera_zoom_for_box`).
    :param r_p: Pericentre radius in code length.
    :param target_frac: Aim for a bar about this fraction of the image width.
    :returns: ``(frac, label)``.
    """
    import render_evolution

    fov = max(box_extent(box)) / float(zoom)
    n = render_evolution._nice_n(target_frac * fov / r_p)
    return float(n * r_p / fov), rf"${n}\,r_p$"


__all__ = [
    "tidal_radius",
    "pericentre_radius",
    "box_extent",
    "is_pencil",
    "camera_zoom_for_box",
    "box_selection",
    "scalebar_in_rp",
]

if __name__ == "__main__":  # quick geometry sanity check
    import render_evolution

    r_p = pericentre_radius()
    print(f"r_t = {tidal_radius():.4f}   r_p = {r_p:.4f}   2.5 r_p = {2.5 * r_p:.4f}")
    for name in ("A", "B", "C"):
        b = render_evolution.BOX_PRESETS[name]
        z = camera_zoom_for_box(b, 1.1)
        fov = max(box_extent(b)) / z
        bar = scalebar_in_rp(b, z, r_p) if is_pencil(b) else \
            render_evolution._scalebar_for_box(b, z, tidal_radius())
        print(f"  {name}: extent={tuple(round(e, 1) for e in box_extent(b))} "
              f"pencil={is_pencil(b)} zoom={z:.4f} "
              f"fov={fov:.2f} R_sun (+-{fov / 2:.2f} = {fov / 2 / r_p:.3f} r_p) bar={bar}")
