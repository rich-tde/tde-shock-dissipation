#!/usr/bin/env python3
"""Face-on close-up geometry: a pencil-beam box around the pericentre region.

A "zoom" here is **not** a smaller cube.  It keeps the wide box's full
line-of-sight (z) extent and only narrows x and y, so a face-on projection
integrates *exactly the same column* as the wide view.  The close-up is then a
true magnification of the wide frame — same physical values, so the wide movie's
colour limits carry over unchanged, and a Rosseland projection stays a real
optical depth instead of a partial one through a truncated slab.

Choosing the grid resolution equal to the wide grid's makes the z spacing come
out identical as well (same z extent, same number of cells), while dx and dy get
finer by the zoom factor.

The frames themselves are still drawn by :func:`richio.render.volume_image`;
nothing here touches plotting.
"""

import numpy as np


def tidal_radius(m_bh=1e4, m_star=0.5, r_star=0.47):
    """Tidal radius ``r_t = R_* (M_BH / M_*)^(1/3)`` in code length (R_sun)."""
    return r_star * (m_bh / m_star) ** (1.0 / 3.0)


def pericentre_radius(m_bh=1e4, m_star=0.5, r_star=0.47, beta=1.0):
    """Pericentre distance ``r_p = r_t / beta`` in code length (R_sun)."""
    return tidal_radius(m_bh, m_star, r_star) / float(beta)


def pencil_box(box, half_width):
    """*box* with x and y narrowed to ``±half_width`` about the origin.

    The z range is left alone on purpose — see the module docstring.  The beam is
    centred on the origin (the black hole in the BH frame), not on the wide box's
    centre, which matters for the off-centre preset B.

    :param box: Wide box ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param half_width: Transverse half-width of the beam, in code length.
    :returns: The pencil-beam box, same format.
    """
    h = float(half_width)
    return [-h, -h, float(box[2]), h, h, float(box[5])]


def camera_zoom_for(box, target_width):
    """The ``zoom`` value that makes the rendered field of view *target_width*.

    :func:`richio.render.yt_backend._camera_vectors` sets the camera width to
    ``max(domain extent) / zoom``.  For a pencil beam the largest extent is the
    *line of sight* (z), not the transverse width we actually want to frame, so
    the zoom factor has to undo that — hence this helper rather than a bare
    ratio.

    :param box: Box ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param target_width: Desired field of view in code length.
    :returns: The zoom factor to pass to ``volume_image(zoom=...)``.
    """
    extent = max(box[3] - box[0], box[4] - box[1], box[5] - box[2])
    return float(extent) / float(target_width)


def beam_selection(snap, coords, half_width, pad_frac=0.25):
    """Boolean mask of cells inside the pencil beam (with a margin).

    Passed to ``to_3dgrid(selection=...)`` so the zoom's k-d tree is built from
    the ~7 % of cells that can matter instead of all ~57 M — the tree build is
    single-threaded and dominates each frame, so this is what keeps the close-up
    from doubling the cost of a frame.

    The margin matters: a grid point just inside the beam edge may have its true
    nearest cell just *outside* it, and cropping without a margin would snap it
    to the wrong cell.  ``pad_frac`` of the half-width is orders of magnitude
    larger than any cell in this region, so the mask cannot change the result.

    :param snap: Snapshot to read coordinates from.
    :param coords: ``(x, y, z)`` field names, e.g. ``("CMx", "CMy", "CMz")``.
    :param half_width: Transverse half-width of the beam, in code length.
    :param pad_frac: Margin as a fraction of *half_width*.
    :returns: Boolean array of shape ``(N,)``.
    """
    from richio import units

    limit = float(half_width) * (1.0 + float(pad_frac)) * units.lscale
    x = snap._get_data(coords[0])
    y = snap._get_data(coords[1])
    return (np.abs(x) <= limit) & (np.abs(y) <= limit)


def scalebar_in_rp(box, zoom, r_p, target_frac=0.2):
    """Scale bar (fraction-of-width, label) measured in ``r_p`` rather than ``r_t``.

    Mirrors :func:`render_evolution._scalebar_for_box` — same rounding, same
    field-of-view convention — but labels the bar in pericentre radii, which is
    the natural ruler once the view is a few ``r_p`` across.

    :param box: Box ``[x0, y0, z0, x1, y1, z1]`` in code length.
    :param zoom: The camera zoom factor used for the render.
    :param r_p: Pericentre radius in code length.
    :param target_frac: Aim for a bar about this fraction of the image width.
    :returns: ``(frac, label)``.
    """
    import render_evolution

    extent = max(box[3] - box[0], box[4] - box[1], box[5] - box[2])
    fov = extent / float(zoom)
    n = render_evolution._nice_n(target_frac * fov / r_p)
    return float(n * r_p / fov), rf"${n}\,r_p$"
