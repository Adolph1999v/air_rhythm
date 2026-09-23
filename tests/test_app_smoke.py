"""Exercise one complete app frame without opening a camera, window, or speaker."""

from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app


class FakeCamera:
    def __init__(self):
        self.released = False
        self.properties = {
            app.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            app.cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            app.cv2.CAP_PROP_FPS: 30.0,
        }
        self.set_calls = []

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((480, 640, 3), 220, dtype=np.uint8)

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        return True

    def get(self, property_id):
        return self.properties.get(property_id, 0.0)

    def release(self):
        self.released = True


class FakeLandmarker:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def detect_for_video(self, _image, _timestamp):
        return SimpleNamespace(hand_landmarks=[], handedness=[])


class AppSmokeTests(unittest.TestCase):
    def run_app(self, camera, audio, pressed_keys, shown_frames=None):
        shown_frames = shown_frames if shown_frames is not None else []
        wait_key = Mock(side_effect=pressed_keys)

        with (
            patch("app.cv2.VideoCapture", return_value=camera),
            patch("app.AudioEngine", return_value=audio),
            patch("app.create_hand_landmarker", return_value=FakeLandmarker()),
            patch("app.cv2.namedWindow"),
            patch("app.cv2.setWindowProperty"),
            patch("app.cv2.imshow", side_effect=lambda _title, frame: shown_frames.append(frame.copy())),
            patch("app.cv2.waitKey", wait_key),
            patch("app.cv2.destroyAllWindows"),
            patch("app.time.monotonic", return_value=100.0),
        ):
            app.main()

        return shown_frames

    def test_main_opens_on_title_screen_and_cleans_up(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True
        shown_frames = []

        self.run_app(camera, audio, [ord("q")], shown_frames)

        self.assertTrue(camera.released)
        audio.close.assert_called_once()
        self.assertEqual(len(shown_frames), 1)
        self.assertEqual(shown_frames[0].shape, (480, 768, 3))
        self.assertFalse(np.all(shown_frames[0] == 220))

    def test_main_composes_a_generated_stage_with_a_live_input_inset(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True

        with (
            patch(
                "app.create_performance_stage",
                wraps=app.create_performance_stage,
            ) as create_stage,
            patch(
                "app.draw_camera_inset",
                wraps=app.draw_camera_inset,
            ) as draw_inset,
        ):
            self.run_app(camera, audio, [ord("q")])

        create_stage.assert_called_once()
        draw_inset.assert_called_once()

    def test_display_window_preserves_the_stage_aspect_ratio(self):
        with (
            patch("app.cv2.namedWindow") as named_window,
            patch("app.cv2.setWindowProperty") as set_window_property,
        ):
            app.configure_display_window()

        named_window.assert_called_once_with(
            app.WINDOW_TITLE,
            app.cv2.WINDOW_NORMAL | app.cv2.WINDOW_KEEPRATIO,
        )
        set_window_property.assert_called_once_with(
            app.WINDOW_TITLE,
            app.cv2.WND_PROP_ASPECT_RATIO,
            app.cv2.WINDOW_KEEPRATIO,
        )

    def test_camera_requests_a_responsive_capture_mode(self):
        camera = FakeCamera()

        configuration = app.configure_camera_capture(camera)

        self.assertIn(
            (app.cv2.CAP_PROP_FRAME_WIDTH, float(app.CAMERA_REQUEST_WIDTH)),
            camera.set_calls,
        )
        self.assertIn(
            (app.cv2.CAP_PROP_FRAME_HEIGHT, float(app.CAMERA_REQUEST_HEIGHT)),
            camera.set_calls,
        )
        self.assertIn(
            (app.cv2.CAP_PROP_FPS, app.CAMERA_REQUEST_FPS),
            camera.set_calls,
        )
        self.assertEqual(configuration["width"], 640.0)
        self.assertEqual(configuration["height"], 480.0)
        self.assertEqual(configuration["fps"], 30.0)

    def test_camera_frames_are_downscaled_without_upscaling(self):
        large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        small_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = app.resize_frame_for_processing(large_frame)
        unchanged = app.resize_frame_for_processing(small_frame)

        self.assertEqual(resized.shape, (720, 1280, 3))
        self.assertIs(unchanged, small_frame)

    def test_stage_dimensions_match_a_maximized_sixteen_by_ten_display(self):
        self.assertEqual(app.stage_dimensions_for_camera(1280, 720), (1280, 800))
        self.assertEqual(app.stage_dimensions_for_camera(640, 480), (768, 480))

    def test_space_moves_from_title_to_gameplay_on_the_next_frame(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True

        with (
            patch("app.ui.draw_title_screen") as draw_title,
            patch("app.ui.draw_game_hud") as draw_hud,
        ):
            self.run_app(camera, audio, [ord(" "), ord("q")])

        draw_title.assert_called_once()
        draw_hud.assert_called_once()
        audio.stop_all.assert_called_once()
        audio.close.assert_called_once()

    def test_help_overlay_pauses_game_updates_until_it_is_closed(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True

        with patch(
            "app.update_challenge_nodes",
            wraps=app.update_challenge_nodes,
        ) as update_challenge:
            self.run_app(
                camera,
                audio,
                [ord(" "), ord("h"), ord("h"), ord("q")],
            )

        self.assertEqual(update_challenge.call_count, 2)
        self.assertEqual(audio.stop_all.call_count, 2)

    def test_benchmark_key_records_and_saves_a_privacy_safe_report(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True

        with patch(
            "app.save_benchmark_report",
            return_value=(Path("report.json"), Path("report.md")),
        ) as save_report:
            self.run_app(camera, audio, [ord("b"), ord("b"), ord("q")])

        save_report.assert_called_once()
        report = save_report.call_args.args[0]
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(
            report["metadata"]["landmark_filter"],
            "Adaptive One Euro-style",
        )
        self.assertEqual(report["metadata"]["visual_dropout_grace_ms"], 140)
        self.assertFalse(report["privacy"]["camera_images_saved"])
        self.assertFalse(report["privacy"]["landmark_coordinates_saved"])

    def test_handedness_score_is_not_mislabeled_as_tracking_accuracy(self):
        result = SimpleNamespace(
            handedness=[
                [SimpleNamespace(category_name="Left", score=0.93)],
                [SimpleNamespace(category_name="Right", score=1.4)],
                [],
            ]
        )

        self.assertEqual(app.handedness_confidences(result), (0.93, 1.0))

if __name__ == "__main__":
    unittest.main()
