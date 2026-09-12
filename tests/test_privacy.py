"""Checks for the privacy-safe camera renderer."""

from types import SimpleNamespace
import unittest

import numpy as np

from privacy import PrivacyMode, PrivacyRenderer, next_privacy_mode


def landmark(x, y):
    return SimpleNamespace(x=x, y=y)


def centered_hand():
    """Return a recognizable open-hand layout with all 21 landmarks."""
    return [
        landmark(0.50, 0.80),  # Wrist
        landmark(0.44, 0.69), landmark(0.38, 0.64),
        landmark(0.33, 0.58), landmark(0.28, 0.51),  # Thumb
        landmark(0.43, 0.63), landmark(0.42, 0.48),
        landmark(0.42, 0.34), landmark(0.42, 0.20),  # Index
        landmark(0.50, 0.61), landmark(0.50, 0.43),
        landmark(0.50, 0.27), landmark(0.50, 0.12),  # Middle
        landmark(0.57, 0.63), landmark(0.58, 0.47),
        landmark(0.59, 0.33), landmark(0.60, 0.20),  # Ring
        landmark(0.64, 0.67), landmark(0.67, 0.55),
        landmark(0.69, 0.45), landmark(0.71, 0.37),  # Pinky
    ]


class PrivacyRendererTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((160, 200, 3), (210, 170, 130), dtype=np.uint8)

    def test_camera_mode_returns_an_unchanged_independent_frame(self):
        renderer = PrivacyRenderer(PrivacyMode.CAMERA)
        result = renderer.apply(self.frame, [centered_hand()])
        np.testing.assert_array_equal(result, self.frame)
        self.assertIsNot(result, self.frame)

    def test_no_hands_reveals_no_camera_pixels(self):
        renderer = PrivacyRenderer()
        private_view = renderer.apply(self.frame, [])
        stage = PrivacyRenderer(PrivacyMode.SKELETON_ONLY).apply(
            self.frame, [centered_hand()]
        )
        np.testing.assert_array_equal(private_view, stage)

    def test_hands_only_reveals_hand_but_hides_far_face_and_body_pixels(self):
        renderer = PrivacyRenderer()
        private_view = renderer.apply(self.frame, [centered_hand()])
        stage = PrivacyRenderer(PrivacyMode.SKELETON_ONLY).apply(self.frame, [])

        # The palm center shows the original camera; distant corners and lower
        # body areas exactly match the opaque privacy stage.
        np.testing.assert_array_equal(private_view[100, 100], self.frame[100, 100])
        np.testing.assert_array_equal(private_view[10, 10], stage[10, 10])
        np.testing.assert_array_equal(private_view[150, 100], stage[150, 100])
        self.assertFalse(np.array_equal(private_view, stage))

    def test_skeleton_only_reveals_none_of_the_original_camera(self):
        renderer = PrivacyRenderer(PrivacyMode.SKELETON_ONLY)
        with_hand = renderer.apply(self.frame, [centered_hand()])
        without_hand = renderer.apply(self.frame, [])
        np.testing.assert_array_equal(with_hand, without_hand)
        self.assertEqual(with_hand.shape, self.frame.shape)
        self.assertEqual(with_hand.dtype, self.frame.dtype)

    def test_cycle_order_moves_through_all_modes(self):
        renderer = PrivacyRenderer()
        self.assertEqual(renderer.mode, PrivacyMode.HANDS_ONLY)
        self.assertEqual(renderer.cycle(), PrivacyMode.SKELETON_ONLY)
        self.assertEqual(renderer.cycle(), PrivacyMode.CAMERA)
        self.assertEqual(renderer.cycle(), PrivacyMode.HANDS_ONLY)
        self.assertEqual(
            next_privacy_mode(PrivacyMode.HANDS_ONLY),
            PrivacyMode.SKELETON_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
