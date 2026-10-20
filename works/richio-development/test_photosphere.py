"""Check native-cell optical-depth rays against small analytic examples.

Run from the repository root with the richanalysis environment::

    python works/richio-development/test_photosphere.py -v

No snapshot files are read and no outputs are written. Constant-opacity rays
test geometry, units, both observer sides, transparent columns and deep
integration. Two-cell examples check the final trapezoid interpolation and
an opacity discontinuity. These verify the algorithm, not resolution of an
arbitrary simulation's nearest-cell opacity field.
"""

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".cache",
        "matplotlib",
    ),
)

import numpy as np
import unyt as u
from richio.analysis import find_photosphere, integrate_optical_depth
from richio.units import units
from scipy.spatial import KDTree
from unyt.exceptions import UnitConversionError


class PhotosphereTests(unittest.TestCase):
    def setUp(self):
        self.coordinates = u.unyt_array([[0, 0, 0]], "cm")
        self.alpha = u.unyt_array([2], "1/cm")
        self.volume = u.unyt_array([1], "cm**3")
        self.box = u.unyt_array([-2, -2, -2, 2, 2, 2], "cm")

    def test_both_observer_sides_have_analytic_photospheres(self):
        # Directions point inward: +z starts at the lower box face.
        directions = np.array([[0, 0, 1], [0, 0, -1]])
        result = find_photosphere(
            self.coordinates,
            self.alpha,
            self.volume,
            self.box,
            np.zeros((2, 3)) * u.cm,
            directions,
        )
        expected = np.array([[0, 0, -5 / 3], [0, 0, 5 / 3]])
        np.testing.assert_allclose(result["position"].to_value("cm"), expected)
        np.testing.assert_allclose(result["distance"].to_value("cm"), 1 / 3)
        np.testing.assert_allclose(result["tau"], 2 / 3)
        np.testing.assert_array_equal(result["found"], [True, True])
        np.testing.assert_allclose(
            result["entry"].to_value("cm"), [[0, 0, -2], [0, 0, 2]]
        )
        np.testing.assert_allclose(
            result["exit"].to_value("cm"), [[0, 0, 2], [0, 0, -2]]
        )

    def test_oblique_direction_is_normalized_and_clipped_to_box(self):
        direction = np.array([1, 2, 3])
        normal = direction / np.linalg.norm(direction)
        entry = np.array([-2 / 3, -4 / 3, -2])
        result = find_photosphere(
            self.coordinates,
            self.alpha,
            self.volume,
            self.box,
            np.zeros(3) * u.cm,
            direction,
        )
        self.assertTrue(result["found"])
        np.testing.assert_allclose(result["entry"].to_value("cm"), entry)
        np.testing.assert_allclose(result["exit"].to_value("cm"), -entry)
        np.testing.assert_allclose(
            result["position"].to_value("cm"), entry + normal / 3
        )

    def test_parallel_offset_rays_broadcast_and_match_individual_calls(self):
        points = u.unyt_array([[0, 0, 0], [0.5, -0.3, 0]], "cm")
        batch = find_photosphere(
            self.coordinates,
            self.alpha,
            self.volume,
            self.box,
            points,
            [0, 0, -1],
        )
        for row, point in enumerate(points):
            single = find_photosphere(
                self.coordinates,
                self.alpha,
                self.volume,
                self.box,
                point,
                [0, 0, -1],
            )
            np.testing.assert_allclose(batch["position"][row], single["position"])
            self.assertEqual(batch["steps"][row], single["steps"])
        np.testing.assert_allclose(batch["position"][:, :2], points[:, :2])

    def test_transparent_and_optically_thin_columns_have_no_photosphere(self):
        for opacity in (0, 0.1):
            with self.subTest(opacity=opacity):
                result = find_photosphere(
                    self.coordinates,
                    u.unyt_array([opacity], "1/cm"),
                    self.volume,
                    self.box,
                    np.zeros(3) * u.cm,
                    [0, 0, 1],
                )
                self.assertFalse(result["found"])
                self.assertTrue(np.isnan(result["position"]).all())
                self.assertAlmostEqual(float(result["tau"]), 4 * opacity)
                self.assertAlmostEqual(float(result["distance"].to_value("cm")), 4)

    def test_ray_missing_box_is_not_extrapolated_to_a_photosphere(self):
        result = find_photosphere(
            self.coordinates,
            self.alpha,
            self.volume,
            self.box,
            u.unyt_array([3, 0, 0], "cm"),
            [0, 0, 1],
        )
        self.assertFalse(result["found"])
        self.assertTrue(np.isnan(result["position"]).all())
        self.assertEqual(result["steps"], 0)

    def test_last_trapezoid_inverts_linear_opacity_not_linear_tau(self):
        # One interval has endpoint opacities a0,a1. Its integral to x=1/2
        # is a0/2 + (a1-a0)/8, including increasing, decreasing and zero a0.
        coordinates = u.unyt_array([[0, 0, 0], [1, 0, 0]], "cm")
        volume = u.unyt_array([8**3, 8**3], "cm**3")
        for a0, a1 in [(1, 3), (3, 1), (0, 2), (2, 2)]:
            target = a0 / 2 + (a1 - a0) / 8
            with self.subTest(alpha=(a0, a1)):
                result = integrate_optical_depth(
                    coordinates,
                    u.unyt_array([a0, a1], "1/cm"),
                    volume,
                    coordinates[0],
                    coordinates[1],
                    target_tau=target,
                    max_delta_tau=None,
                )
                self.assertTrue(result["reached"])
                self.assertEqual(result["steps"], 1)
                self.assertAlmostEqual(float(result["tau"]), target)
                np.testing.assert_allclose(
                    result["position"].to_value("cm"), [0.5, 0, 0]
                )

    def test_entering_opaque_cell_does_not_smear_surface_across_whole_ray(self):
        # Nearest-cell opacity jumps at x=0.5 cm. The exact piecewise-constant
        # photosphere is x=0.5 + (2/3 - 0.1*0.5)/10, within this step tolerance.
        coordinates = u.unyt_array([[0, 0, 0], [1, 0, 0]], "cm")
        result = integrate_optical_depth(
            coordinates,
            u.unyt_array([0.1, 10], "1/cm"),
            u.unyt_array([8**3, 8**3], "cm**3"),
            coordinates[0],
            coordinates[1],
            target_tau=2 / 3,
            max_delta_tau=0.01,
        )
        expected = 0.5 + (2 / 3 - 0.1 * 0.5) / 10
        self.assertTrue(result["reached"])
        self.assertAlmostEqual(
            float(result["position"][0].to_value("cm")), expected, delta=0.001
        )

    def test_code_length_inputs_physical_units_and_reused_tree_agree(self):
        coordinates = np.array([[0.0, 0, 0], [0, 0, 1]]) * units.lscale
        alpha = np.array([2.0, 2]) / units.lscale
        volume = np.ones(2) * units.lscale**3
        box = [-2, -2, -2, 2, 2, 2]
        point = [0, 0, 0]
        code = find_photosphere(coordinates, alpha, volume, box, point, [0, 0, 1])
        physical_coordinates = coordinates.in_units("Rsun")
        tree = KDTree(physical_coordinates.to_value("Rsun"))
        physical = find_photosphere(
            physical_coordinates,
            alpha.in_units("1/cm"),
            volume.in_units("cm**3"),
            (np.array(box) * units.lscale).in_units("Rsun"),
            (np.array(point) * units.lscale).in_units("Rsun"),
            [0, 0, 1],
            tree=tree,
        )
        np.testing.assert_allclose(
            code["position"].to_value("cm"), physical["position"].to_value("cm")
        )
        np.testing.assert_allclose(
            code["distance"].to_value("cm"),
            (1 * units.lscale / 3).to_value("cm"),
            rtol=1e-12,
        )

    def test_deep_integral_without_opacity_cap_does_not_resolve_every_tau(self):
        result = integrate_optical_depth(
            self.coordinates,
            u.unyt_array([1e8], "1/cm"),
            self.volume,
            np.zeros(3) * u.cm,
            u.unyt_array([1, 0, 0], "cm"),
            max_delta_tau=None,
        )
        self.assertAlmostEqual(float(result["tau"]), 1e8)
        self.assertLessEqual(result["steps"], 5)
        np.testing.assert_allclose(result["position"].to_value("cm"), [1, 0, 0])

    def test_max_steps_reports_incomplete_integration(self):
        with self.assertRaises(RuntimeError):
            integrate_optical_depth(
                self.coordinates,
                self.alpha,
                self.volume,
                np.zeros(3) * u.cm,
                u.unyt_array([1, 0, 0], "cm"),
                max_steps=2,
            )

    def test_negative_opacity_and_wrong_units_are_rejected(self):
        invalid_fields = [
            (u.unyt_array([-1], "1/cm"), self.volume),
            (u.unyt_array([1], "g"), self.volume),
            (self.alpha, u.unyt_array([1], "cm")),
        ]
        for alpha, volume in invalid_fields:
            with self.subTest(alpha=alpha, volume=volume):
                with self.assertRaises((ValueError, UnitConversionError)):
                    integrate_optical_depth(
                        self.coordinates,
                        alpha,
                        volume,
                        np.zeros(3) * u.cm,
                        u.unyt_array([1, 0, 0], "cm"),
                    )


if __name__ == "__main__":
    unittest.main()
