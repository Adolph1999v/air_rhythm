"""Exercise one complete app frame without opening a camera, window, or speaker."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app


class FakeCamera:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((480, 640, 3), 220, dtype=np.uint8)

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
        self.assertEqual(shown_frames[0].shape, (480, 640, 3))
        self.assertFalse(np.all(shown_frames[0] == 220))

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
