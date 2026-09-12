"""Checks for hit-by-hit melody progress and the sound bank's pitch coverage."""

import unittest

from music import (
    ALL_PITCHES,
    INSTRUMENTS,
    MELODY_NOTES,
    MELODY_STEP_SECONDS,
    MelodyPlayer,
    note_name,
)


class MelodyPlayerTests(unittest.TestCase):
    def test_previewing_does_not_consume_the_next_hit(self):
        player = MelodyPlayer()
        self.assertEqual(player.next_note, 76)
        self.assertEqual(player.next_note, player.advance())
        self.assertEqual(player.next_note, 75)

    def test_complete_phrase_loops_and_reset_restarts(self):
        player = MelodyPlayer()
        phrase = tuple(player.advance() for _ in range(len(MELODY_NOTES)))
        self.assertEqual(phrase[:9], (76, 75, 76, 75, 76, 71, 74, 72, 69))
        self.assertEqual(phrase[-4:], (64, 72, 71, 69))
        self.assertEqual(player.position, 0)
        self.assertEqual(player.advance(), phrase[0])
        player.advance()
        player.reset()
        self.assertEqual(player.next_note, phrase[0])

    def test_preloaded_pitches_cover_both_modes_and_preview_timings(self):
        self.assertTrue(set(MELODY_NOTES).issubset(ALL_PITCHES))
        self.assertTrue(all(item.freestyle_note in ALL_PITCHES for item in INSTRUMENTS))
        self.assertEqual(len(MELODY_NOTES), len(MELODY_STEP_SECONDS))
        self.assertTrue(all(seconds > 0 for seconds in MELODY_STEP_SECONDS))

    def test_pitch_names_and_midi_limits(self):
        self.assertEqual(note_name(60), "C4")
        self.assertEqual(note_name(75), "D#5")
        self.assertEqual(note_name(0), "C-1")
        self.assertEqual(note_name(127), "G9")
        with self.assertRaises(ValueError):
            note_name(128)


if __name__ == "__main__":
    unittest.main()
