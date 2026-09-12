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
    def test_main_starts_with_camera_runs_one_frame_and_cleans_up(self):
        camera = FakeCamera()
        audio = Mock(enabled=True, muted=False, error_message=None)
        audio.start.return_value = True
        shown_frames = []

        with (
            patch("app.cv2.VideoCapture", return_value=camera),
            patch("app.AudioEngine", return_value=audio),
            patch("app.create_hand_landmarker", return_value=FakeLandmarker()),
            patch("app.cv2.imshow", side_effect=lambda _title, frame: shown_frames.append(frame.copy())),
            patch("app.cv2.waitKey", return_value=ord("q")),
            patch("app.cv2.destroyAllWindows"),
            patch("app.time.monotonic", return_value=100.0),
        ):
            app.main()

        self.assertTrue(camera.released)
        audio.close.assert_called_once()
        self.assertEqual(len(shown_frames), 1)
        # The normal camera is the default. This point contains no overlay.
        self.assertTrue(np.all(shown_frames[0][250, 500] == 220))


if __name__ == "__main__":
    unittest.main()
