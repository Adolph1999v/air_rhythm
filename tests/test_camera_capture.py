"""Deterministic checks for non-blocking latest-frame camera capture."""

from queue import Queue
import unittest

import numpy as np

from camera_capture import LatestFrameCamera


class ControlledCamera:
    def __init__(self) -> None:
        self.frames: Queue = Queue()
        self.released = False

    def read(self):
        frame = self.frames.get(timeout=1.0)
        self.frames.task_done()
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.released = True
        self.frames.put(None)


class LatestFrameCameraTests(unittest.TestCase):
    def test_returns_only_frames_newer_than_the_consumed_sequence(self):
        camera = ControlledCamera()
        stream = LatestFrameCamera(camera).start()
        try:
            first_image = np.full((4, 6, 3), 10, dtype=np.uint8)
            camera.frames.put(first_image)
            first = stream.read_latest(timeout=0.5)

            self.assertIsNotNone(first)
            self.assertIs(first.image, first_image)
            self.assertEqual(first.sequence, 1)
            self.assertEqual(first.skipped_since_previous, 0)
            self.assertIsNone(
                stream.read_latest(after_sequence=first.sequence, timeout=0.01)
            )

            second_image = np.full((4, 6, 3), 20, dtype=np.uint8)
            camera.frames.put(second_image)
            second = stream.read_latest(
                after_sequence=first.sequence,
                timeout=0.5,
            )

            self.assertIsNotNone(second)
            self.assertIs(second.image, second_image)
            self.assertEqual(second.sequence, 2)
            self.assertEqual(second.skipped_since_previous, 0)
        finally:
            stream.close()

        self.assertTrue(camera.released)

    def test_reports_replaced_stale_frames(self):
        camera = ControlledCamera()
        stream = LatestFrameCamera(camera).start()
        try:
            camera.frames.put(np.zeros((2, 2, 3), dtype=np.uint8))
            first = stream.read_latest(timeout=0.5)
            self.assertIsNotNone(first)

            camera.frames.put(np.full((2, 2, 3), 1, dtype=np.uint8))
            camera.frames.put(np.full((2, 2, 3), 2, dtype=np.uint8))
            newest = stream.read_latest(
                after_sequence=first.sequence + 1,
                timeout=0.5,
            )
            second = stream.read_latest(
                after_sequence=first.sequence,
                timeout=0.0,
            )

            self.assertEqual(newest.sequence, first.sequence + 2)
            self.assertEqual(second.sequence, first.sequence + 2)
            self.assertEqual(second.skipped_since_previous, 1)
            self.assertTrue(np.all(second.image == 2))
        finally:
            stream.close()

    def test_rejects_invalid_wait_arguments(self):
        stream = LatestFrameCamera(ControlledCamera())

        with self.assertRaises(ValueError):
            stream.read_latest(after_sequence=-1)
        with self.assertRaises(ValueError):
            stream.read_latest(timeout=float("nan"))

        stream.close()


if __name__ == "__main__":
    unittest.main()
