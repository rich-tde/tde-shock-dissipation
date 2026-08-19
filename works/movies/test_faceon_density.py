import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import unyt as u


MODULE_PATH = Path(__file__).with_name("faceon_density.py")
SPEC = importlib.util.spec_from_file_location("faceon_density", MODULE_PATH)
movie = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = movie
SPEC.loader.exec_module(movie)


class FaceOnDensityTests(unittest.TestCase):
    def test_mass_settings_and_fps(self):
        self.assertEqual(movie.mode_settings(1).fps, 8)
        self.assertEqual(movie.mode_settings(2).fps, 16)
        self.assertEqual(movie.mode_settings(3).fps, 24)
        self.assertAlmostEqual(
            movie.mode_settings(3).r_amin.to_value("code_length"), 1e4
        )

    def test_fixed_box_uses_ramin(self):
        config = movie.mode_settings(1)
        expected = np.asarray(
            (-2.0, -0.875, -0.7, 0.5, 0.875, 0.7)
        ) * config.r_amin
        np.testing.assert_allclose(movie.fixed_box(config), expected)

        other = movie.mode_settings(2)
        expected = np.asarray((-1.5, -0.7, -0.7, 0.5, 0.7, 0.7)) * other.r_amin
        np.testing.assert_allclose(movie.fixed_box(other), expected)

    def test_reference_frame_is_dataset_specific(self):
        self.assertTrue(movie.needs_reference_frame("1e4", Path("/run/snap_20.h5")))
        self.assertFalse(movie.needs_reference_frame("1e4", Path("/run/snap_full_21.h5")))
        self.assertTrue(movie.needs_reference_frame("1e6", Path("/run/TEMPTDE/snap_513.h5")))
        self.assertFalse(movie.needs_reference_frame("1e6", Path("/run/TEMPTDE4/snap_626.h5")))
        self.assertFalse(movie.needs_reference_frame("1e6", Path("/run/TEMPTDE4_new/snap_820.h5")))

    def test_unified_limits_span_exactly_six_decades(self):
        vmin, vmax = movie.six_decade_limits()
        self.assertAlmostEqual(np.log10(vmax / vmin), 6.0)
        self.assertEqual((vmin, vmax), movie.COLOR_LIMITS)

    def test_projected_pixel_shape(self):
        quality = movie.QUALITIES["preview"]
        self.assertEqual(movie.grid_samples(quality), (257, 181, 129))

    def test_black_hole_position_in_fixed_window(self):
        self.assertEqual(movie.bh_axes_position(movie.RUNS[1]), (0.8, 0.5))
        self.assertEqual(movie.bh_axes_position(movie.RUNS[2]), (0.75, 0.5))

    def test_ambient_suppression_uses_star_tracer_only_before_cutoff(self):
        class Snapshot:
            density = u.unyt_array([1.0, 2.0], "g/cm**3")
            box = np.asarray([-5, -5, -5, 5, 5, 5]) * movie.richio.units.lscale

            def __getitem__(self, key):
                self.assert_key = key
                return u.unyt_array([0.5, 1.0], "dimensionless")

        snapshot = Snapshot()
        config = movie.RUNS[3]
        early = movie.projection_density(
            snapshot, 0.1 * config.t_fb, config, ambient_factor=1e-4
        )
        np.testing.assert_allclose(early.to_value("g/cm**3"), [1e-4, 2.0])
        np.testing.assert_allclose(snapshot.density.to_value("g/cm**3"), [1.0, 2.0])
        self.assertEqual(snapshot.assert_key, "tracers/Star")
        late = movie.projection_density(
            snapshot, 0.3 * config.t_fb, config, ambient_factor=1e-4
        )
        self.assertIs(late, snapshot.density)

    def test_box_scaled_suppression_cancels_inverse_volume_ambient_density(self):
        class Snapshot:
            def __init__(self, width, density):
                half = width / 2
                self.box = np.asarray(
                    [-half, -half, -half, half, half, half]
                ) * movie.richio.units.lscale
                self.density = u.unyt_array([density, 1.0], "g/cm**3")

            def __getitem__(self, key):
                return u.unyt_array([0.0, 1.0], "dimensionless")

        config = movie.RUNS[3]
        small = Snapshot(10, 8e-8)
        large = Snapshot(20, 1e-8)
        small_rho = movie.projection_density(
            small, 0.1 * config.t_fb, config, ambient_factor=1e-5
        )[0]
        large_rho = movie.projection_density(
            large, 0.1 * config.t_fb, config, ambient_factor=1e-5
        )[0]
        self.assertEqual(small_rho.units, large_rho.units)
        self.assertAlmostEqual(
            small_rho.to_value("g/cm**3"), large_rho.to_value("g/cm**3")
        )

    def test_snapshot_time_uses_registered_physical_units(self):
        self.assertEqual(movie._time_days(u.unyt_quantity(2.5, "day")), 2.5)

    def test_annotation_contains_time_but_not_black_hole_mass(self):
        config = movie.RUNS[1]
        annotation = movie.time_annotation(0.25 * config.t_fb, config)
        self.assertIn(r"t_{\rm fb}", annotation)
        self.assertIn(r"\mathrm{d}", annotation)
        self.assertNotIn("M_", annotation)
        self.assertNotIn("10^", annotation)

    def test_frame_durations_follow_physical_cadence(self):
        times = u.unyt_array([0.0, 1.0, 3.0], "day")
        durations = movie.frame_durations(times, fps=2)
        self.assertEqual(str(durations.units), "s")
        np.testing.assert_allclose(durations / durations[0], [1.0, 2.0, 1.5])
        self.assertAlmostEqual(float(durations.sum().to_value("s")), 1.0)

    def test_projection_forwards_anisotropic_sinh_grid(self):
        class Snapshot:
            t = u.unyt_array([1.0], "s")
            X = u.unyt_array([0.0], "cm")
            Y = u.unyt_array([0.0], "cm")
            Z = u.unyt_array([0.0], "cm")
            density = u.unyt_array([1.0], "g/cm**3")
            box = np.asarray([-5, -5, -5, 5, 5, 5]) * movie.richio.units.lscale

            def __getitem__(self, key):
                return u.unyt_array([1.0], "dimensionless")

            def project(self, *args, **kwargs):
                self.call = (args, kwargs)
                return (
                    u.unyt_array(np.ones((256, 180)), "g/cm**2"),
                    u.unyt_array(np.linspace(-1, 1, 257), "cm"),
                    u.unyt_array(np.linspace(-1, 1, 181), "cm"),
                )

        snapshot = Snapshot()
        path = Path("/run/snap_full_21.h5")
        with mock.patch.object(movie.richio, "load", return_value=snapshot):
            movie.project_snapshot(
                path, movie.RUNS[1], movie.QUALITIES["preview"], "proposed", 7
            )
        _, kwargs = snapshot.call
        self.assertEqual(kwargs["res"], (257, 181, 129))
        self.assertEqual(kwargs["spacing"], ("linear", "linear", "sinh"))
        self.assertEqual(kwargs["workers"], 7)
        self.assertIsNone(kwargs["sinh_scale"][0])
        self.assertIsNone(kwargs["sinh_scale"][1])
        self.assertAlmostEqual(float(kwargs["sinh_scale"][2]), 0.1)

    def test_frame_complete_rejects_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.touch()
            self.assertFalse(movie.frame_complete(path))
            path.write_bytes(b"png")
            self.assertTrue(movie.frame_complete(path))

    def test_snapshot_inventory(self):
        expected = {"1e4": 152, "1e5": 162, "1e6": 842}
        for run, count in expected.items():
            numbers, paths = movie.DATAPATHS(run)
            self.assertEqual(len(numbers), count)
            self.assertEqual(len(set(numbers)), count)
            self.assertEqual(numbers, sorted(numbers))
            self.assertEqual(len(paths), count)

    def test_h264_encoding_has_expected_frames_and_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            for index in range(2):
                imageio.imwrite(
                    frames / f"frame_{index:05d}.png",
                    np.full((16, 16, 3), index * 255, dtype="uint8"),
                )
            output = root / "movie.mp4"
            movie.richrender.encode_movie(frames, 2, output, fps=8)
            reader = imageio.get_reader(output)
            try:
                self.assertEqual(reader.count_frames(), 2)
                self.assertAlmostEqual(reader.get_meta_data()["fps"], 8.0)
            finally:
                reader.close()

    def test_variable_timing_preserves_source_frames_and_holds_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            for index in range(2):
                imageio.imwrite(
                    frames / f"frame_{index:05d}.png",
                    np.full((16, 16, 3), index * 255, dtype="uint8"),
                )
            output = root / "movie-vfr.mp4"
            movie.richrender.encode_movie(
                frames, 2, output, fps=8,
                durations=u.unyt_array([0.05, 0.10], "s"),
            )
            count, seconds = imageio_ffmpeg.count_frames_and_secs(str(output))
            self.assertEqual(count, 3)  # repeated final timestamp closes its hold
            self.assertGreaterEqual(seconds, 0.15)


if __name__ == "__main__":
    unittest.main()
