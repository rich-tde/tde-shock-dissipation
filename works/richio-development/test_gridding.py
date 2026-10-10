r"""Check richio gridding and projection against small in-memory reference cases.

The unittest suite compares nearest-neighbour gridding, streamed integration,
thread/slab consistency, spacing rules, units and plot-worker forwarding.
These controlled cases verify implementation; use
``sinh-projection-convergence.py`` for snapshot-specific accuracy and
``benchmark_gridding.py`` for runtime/memory studies.

Input files
-----------
No simulation files are read. ``ArraySnapshot`` creates 343 synthetic cells
on a 7 by 7 by 7 Cartesian lattice. X/Y/Z are ``float64 (343,)`` arrays in cm;
density is a known linear function in ``g/cm**3``, volume is ``cm**3`` and
the box is ``[-1,-1,-1,1,1,1] cm``. Some tests mock the plotting interface.

Output files
------------
No files or figures are written. ``unittest`` reports test names, pass/fail
counts and tracebacks to stderr; exit status is nonzero on failure.
``ArraySnapshot`` can also be imported for an in-memory numerical example:
``snap.project("density", res=8)`` returns a tuple of three unitful arrays:

``projected``
    ``float64 (7, 7)`` density integral in ``g/cm**2``, axes x then y.
``xspace``, ``yspace``
    ``float64 (8,)`` coordinate arrays in cm defining the corresponding
    integration-map intervals. For an image, transpose ``projected`` and use
    ``origin="lower"`` so x is horizontal and y vertical.

Usage
-----
Run the suite from ``/home/hey4/rich_tde`` with the richanalysis environment::

    python works/richio-development/test_gridding.py -v

Every invocation repeats the checks; there is no stored output to resume.
Importing defines helpers/test cases but does not execute the test suite.

Loading examples
----------------
No output file needs loading. Inspect a synthetic projection in a Python
session from the repository root::

    import importlib.util
    path = "works/richio-development/test_gridding.py"
    spec = importlib.util.spec_from_file_location("gridding_checks", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    snap = checks.ArraySnapshot()
    projected, xspace, yspace = snap.project("density", res=8, workers=1)
    print(projected.shape, projected.units)  # (7, 7), g/cm**2
    values = projected.to_value("g/cm**2")  # float64 (7, 7), x/y axis order
"""

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".cache",
        "matplotlib",
    ),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import unyt as u
from richio.data import Snapshot, _iter_3d_nearest_slabs
from richio.plots import SnapshotPlotter
from richio.units import units
from scipy.spatial import KDTree


class ArraySnapshot(Snapshot):
    """Small in-memory snapshot used to exercise the public gridding API."""

    _field_aliases = {
        "X": [],
        "Y": [],
        "Z": [],
        "Density": ["density"],
        "Volume": ["volume"],
        "Box": ["box"],
    }
    _alias_to_canonical = {}

    def __init__(self):
        axis = np.linspace(-0.9, 0.9, 7)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        self._fields = {
            "X": x.ravel() * u.cm,
            "Y": y.ravel() * u.cm,
            "Z": z.ravel() * u.cm,
            "Density": (2.0 + x + 2.0 * y + 3.0 * z).ravel() * u.g / u.cm**3,
            # Large enough that volume_selection retains cells for every test plane.
            "Volume": np.full(x.size, 0.5) * u.cm**3,
            "Box": u.unyt_array([-1, -1, -1, 1, 1, 1], "cm"),
        }
        super().__init__("snap_0")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            field, index = key
        else:
            field, index = key, slice(None)
        value = self._fields[self._resolve_field_name(field)]
        return value if np.ndim(value) == 0 else value[index]

    def keys(self):
        return sorted(self._fields)


class GriddingTests(unittest.TestCase):
    def setUp(self):
        self.snap = ArraySnapshot()

    def assert_unyt_equal(self, left, right):
        self.assertEqual(left.units, right.units)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_slice_workers_are_identical_for_all_planes(self):
        for plane in ("xy", "xz", "yz"):
            with self.subTest(plane=plane):
                serial = self.snap.slice(
                    "density",
                    res=(13, 11),
                    plane=plane,
                    slice_coord=0.1 * u.cm,
                    volume_selection=False,
                    workers=1,
                )
                threaded = self.snap.slice(
                    "density",
                    res=(13, 11),
                    plane=plane,
                    slice_coord=0.1 * u.cm,
                    volume_selection=False,
                    workers=8,
                )
                for actual, expected in zip(threaded, serial):
                    self.assert_unyt_equal(actual, expected)

    def test_slice_selection_custom_box_and_all_workers(self):
        selection = np.asarray(self.snap.X) > -0.7
        serial = self.snap.slice(
            "density",
            res=9,
            box_size=[-0.5, -0.4, 0.7, 0.8],
            selection=selection,
            volume_selection=False,
            workers=1,
        )
        threaded = self.snap.slice(
            "density",
            res=9,
            box_size=[-0.5, -0.4, 0.7, 0.8],
            selection=selection,
            volume_selection=False,
            workers=-1,
        )
        for actual, expected in zip(threaded, serial):
            self.assert_unyt_equal(actual, expected)

    def test_streamed_projection_matches_full_cube_reference(self):
        selection = np.asarray(self.snap.Y) < 0.8
        for plane in (None, "xy", "xz", "yz"):
            with self.subTest(plane=plane):
                kwargs = {
                    "res": (10, 9, 8),
                    "box_size": [-0.8, -0.7, -0.6, 0.9, 0.8, 0.7],
                    "selection": selection,
                    "plane": plane,
                }
                indices, xspace, yspace, zspace = self.snap.to_3dgrid(
                    **kwargs, workers=1
                )
                coords, source_indices, _, _, _ = self.snap._prepare_3d_grid(**kwargs)
                streamed_indices = np.empty(
                    (len(xspace) - 1, len(yspace) - 1, len(zspace) - 1),
                    dtype=np.intp,
                )
                for start, local_indices in _iter_3d_nearest_slabs(
                    KDTree(coords),
                    xspace[:-1],
                    yspace[:-1],
                    zspace[:-1],
                    workers=1,
                ):
                    absolute_indices = (
                        source_indices[local_indices]
                        if source_indices is not None
                        else local_indices
                    )
                    streamed_indices[start : start + len(local_indices)] = (
                        absolute_indices
                    )
                np.testing.assert_array_equal(streamed_indices, indices[:-1, :-1, :-1])
                expected = np.sum(
                    self.snap.density[indices][:-1, :-1, :-1]
                    * (zspace[1:] - zspace[:-1]),
                    axis=-1,
                ).in_base("cgs")
                actual, actual_x, actual_y = self.snap.project(
                    "density", **kwargs, workers=8
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
                self.assertEqual(actual.units, expected.units)
                self.assert_unyt_equal(actual_x, xspace)
                self.assert_unyt_equal(actual_y, yspace)

    def test_scalar_projection_and_worker_counts_match(self):
        serial = self.snap.project("density", res=8, workers=1)
        threaded = self.snap.project("density", res=8, workers=-1)
        self.assertEqual(threaded[0].units, serial[0].units)
        np.testing.assert_allclose(threaded[0], serial[0], rtol=1e-14, atol=0)
        self.assert_unyt_equal(threaded[1], serial[1])
        self.assert_unyt_equal(threaded[2], serial[2])

    def test_explicit_linear_spacing_preserves_default(self):
        default = self.snap.project("density", res=(9, 8, 7), workers=1)
        explicit = self.snap.project(
            "density", res=(9, 8, 7), workers=1, spacing="linear"
        )
        self.assertEqual(explicit[0].units, default[0].units)
        np.testing.assert_allclose(explicit[0], default[0], rtol=1e-14, atol=0)
        self.assert_unyt_equal(explicit[1], default[1])
        self.assert_unyt_equal(explicit[2], default[2])

    def test_sinh_spacing_matches_definition_and_full_cube_projection(self):
        scale = 0.2 * u.cm
        kwargs = {
            "res": (9, 8, 11),
            "box_size": u.unyt_array([-1, -1, -1, 1, 1, 1], "cm"),
            "spacing": ("linear", "linear", "sinh"),
            "sinh_scale": scale,
        }
        indices, xspace, yspace, zspace = self.snap.to_3dgrid(**kwargs, workers=1)
        transformed = np.linspace(
            np.arcsinh(-1 / 0.2), np.arcsinh(1 / 0.2), 11, endpoint=False
        )
        np.testing.assert_allclose(zspace.to_value("cm"), 0.2 * np.sinh(transformed))
        self.assertFalse(np.allclose(np.diff(zspace), np.diff(zspace)[0]))

        expected = np.sum(
            self.snap.density[indices][:-1, :-1, :-1] * (zspace[1:] - zspace[:-1]),
            axis=-1,
        ).in_base("cgs")
        actual, actual_x, actual_y = self.snap.project("density", **kwargs, workers=8)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
        self.assert_unyt_equal(actual_x, xspace)
        self.assert_unyt_equal(actual_y, yspace)

    def test_spacing_tracks_physical_axes_when_plane_is_permuted(self):
        _, xspace, yspace, zspace = self.snap.to_3dgrid(
            res=(5, 7, 9),
            plane="yz",
            spacing=("linear", "linear", "sinh"),
            sinh_scale=(None, None, 0.2 * u.cm),
            workers=1,
        )
        self.assertTrue(np.allclose(np.diff(xspace), np.diff(xspace)[0]))
        self.assertFalse(np.allclose(np.diff(yspace), np.diff(yspace)[0]))
        self.assertTrue(np.allclose(np.diff(zspace), np.diff(zspace)[0]))

    def test_central_spacing_recovers_known_symmetric_scale(self):
        kwargs = {
            "res": (4, 4, 512),
            "box_size": u.unyt_array([-1, -1, -1000, 1, 1, 1000], "cm"),
            "spacing": ("linear", "linear", "sinh"),
            "workers": 1,
        }
        _, _, _, actual = self.snap.to_3dgrid(**kwargs, central_spacing=0.1 * u.cm)
        _, _, _, expected = self.snap.to_3dgrid(
            **kwargs, sinh_scale=4.142208627022638 * u.cm
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(np.min(np.diff(actual)).to_value("cm"), 0.1, places=11)
        self.assertEqual(actual.units, u.cm)

    def test_central_spacing_with_asymmetry_parity_and_endpoints(self):
        for bounds in ((-1, 1), (-0.3, 1), (0, 1), (-1, 0)):
            for nz in (10, 11):
                for endpoint in (False, True):
                    with self.subTest(bounds=bounds, nz=nz, endpoint=endpoint):
                        _, _, _, axis = self.snap.to_3dgrid(
                            res=(4, 4, nz),
                            box_size=u.unyt_array(
                                [-1, -1, bounds[0], 1, 1, bounds[1]], "cm"
                            ),
                            spacing=("linear", "linear", "sinh"),
                            central_spacing=0.02 * u.cm,
                            endpoint=endpoint,
                            workers=1,
                        )
                        widths = np.diff(axis)
                        self.assertAlmostEqual(
                            widths.min().to_value("cm"), 0.02, places=11
                        )
                        self.assertTrue(np.all(widths > 0))
                        self.assertAlmostEqual(axis[0].to_value("cm"), bounds[0])
                        if endpoint:
                            self.assertAlmostEqual(axis[-1].to_value("cm"), bounds[1])
                        else:
                            self.assertLess(axis[-1].to_value("cm"), bounds[1])

    def test_central_spacing_bare_lengths_and_physical_units_agree(self):
        kwargs = {"res": (6, 7, 8), "spacing": "sinh", "workers": 1}
        bare = self.snap.to_3dgrid(
            **kwargs, central_spacing=(0.02 * u.cm).to_value(units.lscale)
        )
        physical = self.snap.to_3dgrid(**kwargs, central_spacing=0.0002 * u.m)
        for actual, expected in zip(bare, physical):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
        self.assertTrue(all(axis.units == u.cm for axis in physical[1:]))

    def test_central_spacing_tracks_physical_axes_and_can_mix_with_scales(self):
        for plane, expected_widths in (
            ("xy", (0.04, 0.03, 0.02)),
            ("yz", (0.03, 0.02, 0.04)),
        ):
            with self.subTest(plane=plane):
                _, *axes = self.snap.to_3dgrid(
                    res=(7, 8, 9),
                    plane=plane,
                    spacing="sinh",
                    central_spacing=u.unyt_array([0.04, 0.03, 0.02], "cm"),
                    workers=1,
                )
                for axis, expected in zip(axes, expected_widths):
                    self.assertAlmostEqual(
                        np.min(np.diff(axis)).to_value("cm"), expected, places=11
                    )

        _, xspace, yspace, zspace = self.snap.to_3dgrid(
            res=9,
            spacing=("sinh", "linear", "sinh"),
            sinh_scale=(0.2 * u.cm, -1, None),
            central_spacing=(None, -1, 0.02 * u.cm),
            workers=1,
        )
        np.testing.assert_allclose(
            xspace,
            0.2
            * np.sinh(np.linspace(np.arcsinh(-5), np.arcsinh(5), 9, endpoint=False)),
        )
        np.testing.assert_allclose(yspace, np.linspace(-1, 1, 9, endpoint=False))
        self.assertAlmostEqual(np.min(np.diff(zspace)).to_value("cm"), 0.02, places=11)

    def test_central_spacing_projection_matches_full_cube(self):
        for plane in (None, "xz", "yz"):
            with self.subTest(plane=plane):
                kwargs = {
                    "res": (9, 8, 11),
                    "plane": plane,
                    "spacing": "sinh",
                    "central_spacing": u.unyt_array([0.04, 0.03, 0.02], "cm"),
                    "selection": np.asarray(self.snap.Y) < 0.8,
                }
                indices, xspace, yspace, zspace = self.snap.to_3dgrid(
                    **kwargs, workers=1
                )
                expected = np.sum(
                    self.snap.density[indices][:-1, :-1, :-1] * np.diff(zspace), axis=-1
                ).in_base("cgs")
                actual, actual_x, actual_y = self.snap.project(
                    "density", **kwargs, workers=1
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
                self.assertEqual(actual.units, expected.units)
                self.assert_unyt_equal(actual_x, xspace)
                self.assert_unyt_equal(actual_y, yspace)

    def test_central_spacing_linear_limit_and_default_are_unchanged(self):
        for endpoint in (False, True):
            with self.subTest(endpoint=endpoint):
                kwargs = {"res": 8, "endpoint": endpoint, "workers": 1}
                linear = self.snap.to_3dgrid(**kwargs)
                limit = self.snap.to_3dgrid(
                    **kwargs,
                    spacing="sinh",
                    central_spacing=2 / (7 if endpoint else 8) * u.cm,
                )
                ignored = self.snap.to_3dgrid(**kwargs, central_spacing=-1)
                for actual, expected in zip(limit, linear):
                    np.testing.assert_array_equal(actual, expected)
                for actual, expected in zip(ignored, linear):
                    np.testing.assert_array_equal(actual, expected)

    def test_invalid_central_spacing_is_rejected(self):
        kwargs = {"res": 8, "spacing": ("linear", "linear", "sinh"), "workers": 1}
        for target in (0, -0.1, np.nan, np.inf, 0.3):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.snap.to_3dgrid(**kwargs, central_spacing=target * u.cm)
        with self.assertRaises(ValueError):
            self.snap.to_3dgrid(
                **kwargs, central_spacing=0.1 * u.cm, sinh_scale=0.2 * u.cm
            )
        with self.assertRaises(ValueError):
            self.snap.to_3dgrid(**kwargs, central_spacing=(None, 0.1 * u.cm))
        with self.assertRaises(u.exceptions.UnitConversionError):
            self.snap.to_3dgrid(**kwargs, central_spacing=0.1 * u.s)
        for lower, upper in (
            (0.1, 1),
            (-1, -0.1),
            (1, -1),
            (0, 0),
            (-np.inf, 1),
            (-1, np.nan),
        ):
            with self.subTest(bounds=(lower, upper)), self.assertRaises(ValueError):
                self.snap.to_3dgrid(
                    **kwargs,
                    box_size=u.unyt_array([-1, -1, lower, 1, 1, upper], "cm"),
                    central_spacing=0.1 * u.cm,
                )

    def test_invalid_spacing_and_scale_are_clear(self):
        with self.assertRaisesRegex(ValueError, "three-element"):
            self.snap.project("density", res=8, spacing=("linear", "sinh"))
        with self.assertRaisesRegex(ValueError, "Unsupported grid spacing"):
            self.snap.project("density", res=8, spacing=("linear", "linear", "log"))
        with self.assertRaisesRegex(ValueError, "sinh_scale|central_spacing"):
            self.snap.project("density", res=8, spacing=("linear", "linear", "sinh"))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.snap.project(
                "density",
                res=8,
                spacing=("linear", "linear", "sinh"),
                sinh_scale=-1 * u.cm,
            )

    def test_invalid_and_empty_selections_are_clear(self):
        with self.assertRaisesRegex(ValueError, "selection must have shape"):
            self.snap.project("density", res=8, selection=np.ones(3, dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty cell selection"):
            self.snap.project(
                "density", res=8, selection=np.zeros(len(self.snap.X), dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "empty cell selection"):
            self.snap.slice(
                "density",
                res=8,
                selection=np.zeros(len(self.snap.X), dtype=bool),
                volume_selection=False,
            )

    def test_projection_resolution_must_support_integration(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.snap.project("density", res=(8, 8, 1))


class PlotWorkerForwardingTests(unittest.TestCase):
    @patch("richio.plots.scalar_map")
    def test_slice_forwards_workers(self, scalar_map):
        snap = Mock()
        grid = np.ones((2, 2)) * u.g / u.cm**3
        axis = np.arange(2) * u.cm
        snap.slice.return_value = grid, axis, axis
        scalar_map.return_value = Mock(), Mock()

        SnapshotPlotter(snap).slice("density", 2, workers=-1)

        self.assertEqual(snap.slice.call_args.kwargs["workers"], -1)

    @patch("richio.plots.scalar_map")
    def test_projection_forwards_workers_and_spacing(self, scalar_map):
        snap = Mock()
        grid = np.ones((2, 2)) * u.g / u.cm**2
        axis = np.arange(3) * u.cm
        snap.project.return_value = grid, axis, axis
        scalar_map.return_value = Mock(), Mock()

        SnapshotPlotter(snap).projection(
            "density",
            3,
            workers=4,
            spacing=("sinh", "linear", "sinh"),
            sinh_scale=(2 * u.cm, None, None),
            central_spacing=(None, None, 0.1 * u.cm),
        )

        self.assertEqual(snap.project.call_args.kwargs["workers"], 4)
        self.assertEqual(
            snap.project.call_args.kwargs["spacing"],
            ("sinh", "linear", "sinh"),
        )
        self.assertEqual(
            snap.project.call_args.kwargs["sinh_scale"], (2 * u.cm, None, None)
        )
        self.assertEqual(
            snap.project.call_args.kwargs["central_spacing"], (None, None, 0.1 * u.cm)
        )
        self.assertNotIn("central_spacing", scalar_map.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
