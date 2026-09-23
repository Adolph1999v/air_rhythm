"""Checks for smooth poses, stable identities, and safe tracking dropouts."""

from types import SimpleNamespace
import unittest

import numpy as np

from hand_stabilizer import HandLandmarkStabilizer


def hand_at(x: float, y: float = 0.55, z: float = -0.02):
    return [
        SimpleNamespace(x=x + index * 0.0005, y=y, z=z)
        for index in range(21)
    ]


def handedness(label: str, score: float = 0.95):
    return [SimpleNamespace(category_name=label, score=score)]


class HandLandmarkStabilizerTests(unittest.TestCase):
    def test_handedness_keeps_output_order_stable_when_input_order_reverses(self):
        stabilizer = HandLandmarkStabilizer()
        first = stabilizer.update(
            [hand_at(0.25), hand_at(0.75)],
            [handedness("Left"), handedness("Right")],
            1.0,
        )
        second = stabilizer.update(
            [hand_at(0.74), hand_at(0.26)],
            [handedness("Right"), handedness("Left")],
            1.033,
        )

        self.assertEqual(
            [hand.identity for hand in first.active.hand_landmarks],
            ["Left", "Right"],
        )
        self.assertEqual(
            [hand.identity for hand in second.active.hand_landmarks],
            ["Left", "Right"],
        )
        self.assertLess(second.active.hand_landmarks[0][0].x, 0.5)
        self.assertGreater(second.active.hand_landmarks[1][0].x, 0.5)

    def test_adaptive_filter_reduces_stationary_landmark_jitter(self):
        stabilizer = HandLandmarkStabilizer()
        raw_positions = []
        filtered_positions = []
        for frame_index in range(30):
            raw_x = 0.49 if frame_index % 2 == 0 else 0.51
            raw_positions.append(raw_x)
            result = stabilizer.update(
                [hand_at(raw_x)],
                [handedness("Left")],
                2.0 + frame_index / 30.0,
            )
            filtered_positions.append(result.active.hand_landmarks[0][0].x)

        self.assertLess(
            np.std(filtered_positions[5:]),
            np.std(raw_positions[5:]) * 0.6,
        )

    def test_filter_reduces_the_angle_jitter_amplified_by_a_long_stick(self):
        stabilizer = HandLandmarkStabilizer()
        raw_angles = []
        filtered_angles = []
        for frame_index in range(30):
            jitter = -0.014 if frame_index % 2 == 0 else 0.014
            landmarks = hand_at(0.5)
            landmarks[5] = SimpleNamespace(
                x=0.45 - jitter,
                y=0.66,
                z=-0.02,
            )
            landmarks[8] = SimpleNamespace(
                x=0.55 + jitter,
                y=0.36,
                z=-0.02,
            )
            raw_angles.append(
                np.arctan2(
                    landmarks[8].y - landmarks[5].y,
                    landmarks[8].x - landmarks[5].x,
                )
            )
            result = stabilizer.update(
                [landmarks],
                [handedness("Left")],
                2.0 + frame_index / 30.0,
            )
            filtered = result.active.hand_landmarks[0]
            filtered_angles.append(
                np.arctan2(
                    filtered[8].y - filtered[5].y,
                    filtered[8].x - filtered[5].x,
                )
            )

        self.assertLess(
            np.std(filtered_angles[5:]),
            np.std(raw_angles[5:]) * 0.6,
        )

    def test_confident_handedness_preserves_identity_when_hands_cross(self):
        stabilizer = HandLandmarkStabilizer()
        stabilizer.update(
            [hand_at(0.20), hand_at(0.80)],
            [handedness("Left"), handedness("Right")],
            2.0,
        )
        crossed = stabilizer.update(
            [hand_at(0.78), hand_at(0.22)],
            [handedness("Left"), handedness("Right")],
            2.033,
        )

        identities = [hand.identity for hand in crossed.active.hand_landmarks]
        self.assertEqual(identities, ["Left", "Right"])
        self.assertGreater(crossed.active.hand_landmarks[0][0].x, 0.20)
        self.assertLess(crossed.active.hand_landmarks[1][0].x, 0.80)

    def test_filter_still_moves_quickly_during_a_deliberate_strike(self):
        stabilizer = HandLandmarkStabilizer()
        stabilizer.update(
            [hand_at(0.25)],
            [handedness("Left")],
            3.0,
        )
        moved = stabilizer.update(
            [hand_at(0.75)],
            [handedness("Left")],
            3.033,
        )

        filtered_x = moved.active.hand_landmarks[0][0].x
        self.assertGreater(filtered_x, 0.50)
        self.assertLessEqual(filtered_x, 0.75)

        forward = stabilizer.update(
            [hand_at(0.75, z=-0.55)],
            [handedness("Left")],
            3.066,
        )
        self.assertLess(forward.active.hand_landmarks[0][0].z, -0.20)

    def test_dropout_grace_is_visual_only_and_expires_quickly(self):
        stabilizer = HandLandmarkStabilizer(dropout_grace_seconds=0.14)
        stabilizer.update(
            [hand_at(0.4)],
            [handedness("Left")],
            4.0,
        )

        brief_dropout = stabilizer.update([], [], 4.10)
        expired_dropout = stabilizer.update([], [], 4.20)

        self.assertEqual(len(brief_dropout.active.hand_landmarks), 0)
        self.assertEqual(len(brief_dropout.visible.hand_landmarks), 1)
        self.assertEqual(brief_dropout.stale_identities, ("Left",))
        self.assertEqual(len(expired_dropout.active.hand_landmarks), 0)
        self.assertEqual(len(expired_dropout.visible.hand_landmarks), 0)

    def test_invalid_landmark_data_is_ignored(self):
        stabilizer = HandLandmarkStabilizer()
        result = stabilizer.update(
            [[SimpleNamespace(x=float("nan"), y=0.5, z=0.0)] * 21],
            [handedness("Left")],
            5.0,
        )

        self.assertEqual(len(result.active.hand_landmarks), 0)


if __name__ == "__main__":
    unittest.main()
