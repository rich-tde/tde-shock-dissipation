#!/usr/bin/env python3
r"""Geometry helpers for narrow transverse boxes with a full line of sight.

A face-on close-up narrows x/y while retaining the wide view's full z extent,
so the column integral remains comparable. Camera zoom is scaled to frame the
transverse extent rather than the longest box side. Optional padded x/y cell
selection reduces grid-building cost for pencil-shaped boxes. Verify that
padding preserves nearest neighbours for a new run or a sparse region; it is
an approximation. The framing convention is intended for face-on views.

Input files
-----------
No files are opened by the geometry functions. Movie drivers supply:

``box``
    Six bounds ``[x0,y0,z0,x1,y1,z1]`` in one consistent length unit;
    ``box_selection`` specifically expects plain RICH code lengths.
``m_bh``, ``m_star``, ``r_star``, ``beta``
    Scalar BH/stellar masses and stellar radius in code solar units, and
    dimensionless penetration factor. Defaults are 1e4, 0.5, 0.47 and 1.
``snap``, ``coords`` for ``box_selection``
    A loaded ``richio`` snapshot and a three-string tuple identifying the
    x/y/z coordinate fields, e.g. ``("CMx", "CMy", "CMz")``. Cell coordinates
    are unitful arrays of shape ``(N,)``, where ``N`` is the cell count.

Output files
------------
No files are written. Functions return these in-memory values:

``tidal_radius(...)``, ``pericentre_radius(...)``
    Floating-point radius in code length (solar-radius scale): respectively
    ``R_* (M_BH/M_*)**(1/3)`` and that value divided by ``beta``.
``box_extent(box)``
    Three-element tuple ``(dx,dy,dz)`` in the box's length unit.
``is_pencil(box, ratio=2.0)``
    Boolean, true when ``dz > ratio * max(dx,dy)``.
``camera_zoom_for_box(box, zoom)``
    Dimensionless float ``zoom * max(dx,dy,dz) / max(dx,dy)`` to pass to
    ``richio.render``. It equals the input zoom for isotropic boxes.
``box_selection(snap, coords, box, pad_frac=0.25)``
    Boolean ``ndarray (N,)`` in snapshot cell order; true selects cells inside
    the padded x/y footprint. z is not cut. Returns ``None`` for non-pencil
    boxes, meaning use all cells. This mask is not a physical wind selection.
``scalebar_in_rp(box, zoom, r_p, target_frac=0.2)``
    Tuple ``(fraction, label)``: dimensionless float fraction of screen width
    and a math-text string labelled in pericentre radii. Pass the effective
    zoom from ``camera_zoom_for_box`` and ``r_p`` in the box's length unit.

Usage
-----
Import the module from a sibling movie driver. Running it directly performs
small geometry checks and prints radii, extents, pencil flags, zoom, field of
view and scale-bar values for presets A/B/C::

    python works/movies/movie_zoom.py

Repeated calls simply recompute their in-memory results; there is no cache or
resume state.

Loading examples
----------------
Use the helper from a notebook or Python session at the repository root::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("works/movies").resolve()))
    import movie_zoom

    box = [-35, -35, -1400, 35, 35, 1400]
    zoom = movie_zoom.camera_zoom_for_box(box, 1.1)
    rp = movie_zoom.pericentre_radius()
    fraction, label = movie_zoom.scalebar_in_rp(box, zoom, rp)
    print(zoom, fraction, label)

For a real snapshot, pass a returned mask to the grid builder's
``selection`` argument; inspect ``mask.sum()`` and ``mask.shape`` before use.
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
    the wrong cell. ``pad_frac`` pads by a fraction of the transverse half-width.
    Compare with the uncropped grid when changing runs or zoom regions; padding
    alone does not guarantee identical nearest neighbours.

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
        bar = (
            scalebar_in_rp(b, z, r_p)
            if is_pencil(b)
            else render_evolution._scalebar_for_box(b, z, tidal_radius())
        )
        print(
            f"  {name}: extent={tuple(round(e, 1) for e in box_extent(b))} "
            f"pencil={is_pencil(b)} zoom={z:.4f} "
            f"fov={fov:.2f} R_sun (+-{fov / 2:.2f} = {fov / 2 / r_p:.3f} r_p) bar={bar}"
        )
