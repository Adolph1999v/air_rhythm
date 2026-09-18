"""Camera-free checks for the person-free performance-stage renderer."""

from types import SimpleNamespace
import unittest

import numpy as np

from performance_stage import (
    create_performance_stage,
    draw_camera_inset,
    draw_collision_points,
    draw_virtual_drumsticks,
)


def hand_at(x: float, y: float):
    """Return one simple MediaPipe-like hand with a clear index direction."""
    landmarks = [SimpleNamespace(x=x, y=y) for _ in range(21)]
    landmarks[5] = SimpleNamespace(x=x - 0.07, y=y + 0.08)
    landmarks[8] = SimpleNamespace(x=x + 0.06, y=y - 0.10)
    landmarks[9] = SimpleNamespace(x=x, y=y + 0.10)
    landmarks[12] = SimpleNamespace(x=x + 0.04, y=y - 0.12)
    return landmarks


class PerformanceStageTests(unittest.TestCase):
    def camera_frame(self, width=1280, height=720):
        return np.full((height, width, 3), (0, 250, 0), dtype=np.uint8)

    def test_main_stage_is_generated_without_copying_camera_pixels(self):
        camera = self.camera_frame()
        stage = create_performance_stage(camera, current_time=2.5)

        self.assertEqual(stage.shape, camera.shape)
        self.assertEqual(stage.dtype, camera.dtype)
        self.assertFalse(np.any(np.all(stage == (0, 250, 0), axis=2)))

    def test_virtual_drumsticks_follow_landmark_positions_at_showcase_sizes(self):
        for width, height in ((640, 480), (1280, 720), (1920, 1080)):
            with self.subTest(size=(width, height)):
                stage = create_performance_stage(self.camera_frame(width, height))
                before = stage.copy()
                draw_virtual_drumsticks(stage, [hand_at(0.35, 0.55), hand_at(0.70, 0.48)])

                self.assertEqual(stage.shape, before.shape)
                self.assertFalse(np.array_equal(stage, before))

    def test_collision_points_show_every_active_fingertip(self):
        stage = create_performance_stage(self.camera_frame())
        before = stage.copy()
        hand = hand_at(0.45, 0.55)

        result = draw_collision_points(stage, [hand])

        self.assertIs(result, stage)
        self.assertFalse(np.array_equal(stage, before))
        for fingertip_index in (4, 8, 12, 16, 20):
            x = min(int(hand[fingertip_index].x * stage.shape[1]), stage.shape[1] - 1)
            y = min(int(hand[fingertip_index].y * stage.shape[0]), stage.shape[0] - 1)
            self.assertFalse(np.array_equal(stage[y, x], before[y, x]))

    def test_live_input_inset_is_bottom_right_and_keeps_camera_pixels_local(self):
        stage = create_performance_stage(self.camera_frame())
        camera = self.camera_frame()
        left, top, right, bottom = draw_camera_inset(
            stage,
            camera,
            mode_label="Camera",
            hand_count=2,
        )

        self.assertGreater(left, stage.shape[1] // 2)
        self.assertGreater(top, stage.shape[0] // 2)
        self.assertLess(right, stage.shape[1])
        self.assertLess(bottom, stage.shape[0])
        self.assertTrue(
            np.any(
                np.all(stage[top:bottom, left:right] == (0, 250, 0), axis=2)
            )
        )
        self.assertFalse(np.any(np.all(stage[:top, :left] == (0, 250, 0), axis=2)))


if __name__ == "__main__":
    unittest.main()
