"""Render Air Rhythm's interface without opening a camera or window."""

import unittest
from unittest.mock import patch

import numpy as np

import ui


class UserInterfaceTests(unittest.TestCase):
    def frame(self, width=640, height=480):
        return np.full((height, width, 3), 96, dtype=np.uint8)

    def assert_rendered(self, before, after):
        self.assertIs(after.dtype, before.dtype)
        self.assertEqual(after.shape, before.shape)
        self.assertFalse(np.array_equal(before, after))

    def test_title_screen_renders_at_common_camera_aspect_ratios(self):
        for width, height in ((640, 480), (1280, 720), (1920, 1080)):
            with self.subTest(size=(width, height)):
                frame = self.frame(width, height)
                before = frame.copy()
                result = ui.draw_title_screen(
                    frame,
                    hand_count=2,
                    privacy_label="Camera",
                    sound_label="Sound on",
                    current_time=1.25,
                )
                self.assertIs(result, frame)
                self.assert_rendered(before, result)

    def test_every_component_is_safe_on_a_small_frame(self):
        drawers = (
            lambda frame: ui.draw_brand_badge(frame),
            lambda frame: ui.draw_glass_panel(frame, (-20, -10), (300, 190)),
            lambda frame: ui.draw_glow_circle(frame, (80, 60), 20, pulse=1.5),
            lambda frame: ui.draw_countdown(frame, "GO!", progress=2.0),
            lambda frame: ui.draw_help_overlay(frame),
            lambda frame: ui.draw_debug_overlay(frame, {"hands": 1, "landmarks": 21}),
            lambda frame: ui.draw_results(
                frame,
                score=12500,
                accuracy=87.5,
                rank="A",
                perfect=8,
                great=4,
                good=2,
                misses=1,
                max_combo=9,
            ),
        )
        for drawer in drawers:
            with self.subTest(drawer=drawer):
                frame = self.frame(160, 120)
                before = frame.copy()
                result = drawer(frame)
                self.assertIs(result, frame)
                self.assert_rendered(before, result)

    def test_game_hud_keeps_shape_and_dtype_at_common_resolutions(self):
        for width, height in ((640, 480), (1280, 720), (1920, 1080)):
            with self.subTest(size=(width, height)):
                frame = self.frame(width, height)
                before = frame.copy()
                result = ui.draw_game_hud(
                    frame,
                    hands=2,
                    score=18700,
                    combo=11,
                    progress=0.47,
                    hits=18,
                    misses=2,
                    mode_label="Challenge",
                    song_label="Fur Elise",
                    privacy_label="Camera",
                    sound_label="Sound on",
                    debug_info={
                        "fps": "29.8",
                        "landmarks": 42,
                        "confidence": (0.91, 0.87),
                        "gesture": "DOWN",
                    },
                )
                self.assertIs(result, frame)
                self.assert_rendered(before, result)

    def test_results_screen_renders_at_showcase_resolutions(self):
        for width, height in ((1280, 720), (1920, 1080)):
            with self.subTest(size=(width, height)):
                frame = self.frame(width, height)
                before = frame.copy()
                result = ui.draw_results(
                    frame,
                    score=18700,
                    accuracy=87.5,
                    rank="A",
                    perfect=12,
                    great=5,
                    good=1,
                    misses=2,
                    max_combo=11,
                    song_label="Fur Elise challenge",
                )
                self.assertIs(result, frame)
                self.assert_rendered(before, result)

    def test_game_progress_is_clamped_to_zero_and_one(self):
        def rendered(progress):
            frame = self.frame()
            return ui.draw_game_hud(
                frame,
                hands=1,
                score=100,
                combo=2,
                progress=progress,
                hits=3,
                misses=1,
                mode_label="Challenge",
                song_label="Demo song",
                privacy_label="Camera",
                sound_label="Sound on",
            )

        np.testing.assert_array_equal(rendered(-5), rendered(0))
        np.testing.assert_array_equal(rendered(7), rendered(1))
        np.testing.assert_array_equal(rendered(float("nan")), rendered(0))

    def test_ready_progress_is_clamped(self):
        def rendered(progress):
            frame = self.frame()
            return ui.draw_title_screen(
                frame, 0, "Camera", "Sound on", 2.0, ready_progress=progress
            )

        np.testing.assert_array_equal(rendered(-1), rendered(0))
        np.testing.assert_array_equal(rendered(3), rendered(1))

    def test_text_fitting_and_alignment_draw_inside_frame(self):
        frame = self.frame(240, 100)
        scale = ui.fit_text_scale("A very long status message", 100, 1.0)
        self.assertGreaterEqual(scale, 0.2)
        self.assertLess(scale, 1.0)
        box = ui.draw_text(
            frame,
            "A very long status message",
            (120, 50),
            1.0,
            align="center",
            max_width=100,
        )
        x, y, width, height = box
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, frame.shape[1])
        self.assertLessEqual(y + height, frame.shape[0] + 2)

    def test_invalid_alignment_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "align"):
            ui.draw_text(self.frame(), "test", (10, 20), align="diagonal")

    def test_small_panel_text_skips_outline_but_large_text_keeps_a_thin_edge(self):
        frame = self.frame()
        with patch("ui.cv2.putText") as put_text:
            ui.draw_text(frame, "Readable", (20, 40), 0.4, thickness=1)

        self.assertEqual(put_text.call_count, 1)

        with patch("ui.cv2.putText") as put_text:
            ui.draw_text(frame, "Large label", (20, 40), 1.2, thickness=1)
        self.assertEqual(put_text.call_count, 1)

        with patch("ui.cv2.putText") as put_text:
            ui.draw_text(frame, "GO", (20, 60), 1.0, thickness=2)
        self.assertEqual(put_text.call_count, 2)
        self.assertEqual(put_text.call_args_list[0].args[5], ui.TEXT_SHADOW)
        self.assertEqual(put_text.call_args_list[0].args[6], 3)
        self.assertEqual(put_text.call_args_list[1].args[6], 2)

    def test_glow_clips_safely_at_every_frame_edge(self):
        for center in ((0, 0), (199, 0), (0, 119), (199, 119), (-5, 60), (204, 60)):
            with self.subTest(center=center):
                frame = self.frame(200, 120)
                before = frame.copy()
                result = ui.draw_glow_circle(frame, center, 20, pulse=0.75)
                self.assertIs(result, frame)
                self.assert_rendered(before, result)

    def test_portfolio_and_debug_wording_describes_each_technology_precisely(self):
        title_frame = self.frame(1280, 720)
        with patch("ui.draw_text", wraps=ui.draw_text) as title_text:
            ui.draw_title_screen(title_frame, 2, "Camera", "Sound on", 1.0)
        title_labels = [call.args[1] for call in title_text.call_args_list]
        self.assertIn(
            "MediaPipe pretrained landmarks + custom gesture / collision logic",
            title_labels,
        )

        debug_frame = self.frame(1280, 720)
        with patch("ui.draw_text", wraps=ui.draw_text) as debug_text:
            ui.draw_debug_overlay(
                debug_frame,
                {
                    "hands": 2,
                    "confidence": (0.8, 1.0),
                    "gesture": "NO DIRECTIONAL STRIKE",
                },
            )
        debug_labels = [call.args[1] for call in debug_text.call_args_list]
        self.assertIn("Mean hand-class confidence  90%", debug_labels)
        self.assertIn("Motion rule  NO DIRECTIONAL STRIKE", debug_labels)
        self.assertIn("OpenCV: camera capture + rendering", debug_labels)
        self.assertIn("App logic: fingertip path collision", debug_labels)


if __name__ == "__main__":
    unittest.main()
