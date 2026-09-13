"""Check that the accepted hybrid contacts reliably become musical hits."""

import unittest
from unittest.mock import Mock, patch

from app import (
    CHALLENGE_TARGET_HEIGHT_RATIO,
    FallingNode,
    FingertipMotion,
    NodeHit,
    challenge_node_y,
    create_challenge_node,
    node_radius_for_frame,
    play_node_hits,
    run_sound_test,
    update_challenge_nodes,
    update_falling_nodes,
)
from music import INSTRUMENTS, MELODY_NOTES, MelodyPlayer
from rhythm_game import ChartEvent, RhythmRound, RoundPhase, TimingGrade


class HitAudioTests(unittest.TestCase):
    def setUp(self):
        self.audio = Mock()
        self.melody = MelodyPlayer()

    def node(self, instrument=INSTRUMENTS[0], y=200):
        return FallingNode(
            x=200, y=y, radius=34, speed=0,
            color=instrument.color, instrument=instrument.key,
        )

    def test_stationary_contact_plays_once_and_removes_node(self):
        motion = FingertipMotion((200, 200), (200, 200), 0, None)
        remaining, hits, misses = update_falling_nodes(
            [self.node()], 0.03, [motion], 720
        )
        play_node_hits(hits, self.audio, self.melody, True)
        self.assertEqual((len(remaining), len(hits), misses), (0, 1, 0))
        self.assertEqual(hits[0].rating, "GOOD")
        self.audio.play_note.assert_called_once_with(
            MELODY_NOTES[0], instrument="keys", velocity=0.75
        )
        _, next_hits, _ = update_falling_nodes(remaining, 0.03, [motion], 720)
        play_node_hits(next_hits, self.audio, self.melody, True)
        self.assertEqual(self.audio.play_note.call_count, 1)

    def test_fast_sweep_plays_even_when_endpoints_miss(self):
        motion = FingertipMotion((80, 200), (320, 200), 1.0, None)
        _, hits, _ = update_falling_nodes([self.node()], 0.03, [motion], 720)
        play_node_hits(hits, self.audio, self.melody, True)
        self.assertEqual(hits[0].rating, "PERFECT")
        self.audio.play_note.assert_called_once_with(
            MELODY_NOTES[0], instrument="keys", velocity=1.0
        )

    def test_misses_are_silent_and_keep_next_melody_note(self):
        remaining, hits, misses = update_falling_nodes(
            [self.node(y=800)], 0.03, [], 720
        )
        play_node_hits(hits, self.audio, self.melody, True)
        self.assertEqual((remaining, misses), ([], 1))
        self.assertEqual(self.melody.position, 0)
        self.audio.play_note.assert_not_called()

    def test_simultaneous_hits_share_one_step_then_next_hit_advances(self):
        hits = [NodeHit(self.node(instrument), "GOOD") for instrument in INSTRUMENTS]
        play_node_hits(hits, self.audio, self.melody, True)
        self.assertEqual(self.melody.position, 1)
        self.assertEqual(self.audio.play_note.call_count, 4)
        self.assertTrue(all(
            call.args[0] == MELODY_NOTES[0]
            for call in self.audio.play_note.call_args_list
        ))
        play_node_hits(hits[:1], self.audio, self.melody, True)
        self.assertEqual(self.audio.play_note.call_args.args[0], MELODY_NOTES[1])

    def test_same_instrument_in_one_frame_uses_strongest_hit_once(self):
        hits = [NodeHit(self.node(), rating) for rating in ("GOOD", "PERFECT")]
        play_node_hits(hits, self.audio, self.melody, True)
        self.audio.play_note.assert_called_once_with(
            MELODY_NOTES[0], instrument="keys", velocity=1.0
        )

    def test_free_play_uses_instrument_pitches_without_advancing_melody(self):
        for instrument in INSTRUMENTS:
            play_node_hits(
                [NodeHit(self.node(instrument), "GREAT")],
                self.audio, self.melody, False,
            )
            self.audio.play_note.assert_called_with(
                instrument.freestyle_note, instrument=instrument.key, velocity=0.9
            )
        self.assertEqual(self.melody.position, 0)

    def test_sound_check_reports_output_failure_mid_melody_and_closes(self):
        self.audio.start.return_value = True
        self.audio.enabled = True
        self.audio.error_message = "Output disconnected"

        def disconnect_output(_seconds):
            self.audio.enabled = False

        with (
            patch("app.AudioEngine", return_value=self.audio),
            patch("app.time.sleep", side_effect=disconnect_output),
            patch("builtins.print") as printed,
        ):
            self.assertFalse(run_sound_test())
        self.audio.play_note.assert_called_once()
        self.audio.close.assert_called_once()
        printed.assert_any_call("Output disconnected")

    def test_song_node_uses_absolute_clock_and_plays_its_attached_pitch(self):
        event = ChartEvent(0, 64, 0.65)
        rhythm_round = RhythmRound(started_at=100.0, chart=(event,))
        target_time = rhythm_round.target_time(event)
        node = create_challenge_node(
            event,
            rhythm_round,
            1280,
            720,
            rhythm_round.spawn_time(event),
            [],
        )
        node.x = 300
        target_position = (300, round(720 * CHALLENGE_TARGET_HEIGHT_RATIO))
        motion = FingertipMotion(
            target_position,
            target_position,
            0.0,
            None,
        )
        remaining, hits, misses = update_challenge_nodes(
            [node], target_time, [motion], rhythm_round, 720
        )
        self.assertEqual((remaining, misses), ([], 0))
        self.assertIs(hits[0].timing_grade, TimingGrade.PERFECT)
        self.assertAlmostEqual(node.y, 720 * CHALLENGE_TARGET_HEIGHT_RATIO)
        play_node_hits(hits, self.audio, self.melody, True)
        self.audio.play_note.assert_called_once_with(
            64, instrument="keys", velocity=0.75
        )

    def test_song_contact_waits_for_window_then_late_node_becomes_miss(self):
        event = ChartEvent(0, 60, 0.65)
        rhythm_round = RhythmRound(started_at=100.0, chart=(event,))
        target_time = rhythm_round.target_time(event)
        node = create_challenge_node(
            event,
            rhythm_round,
            1280,
            720,
            rhythm_round.spawn_time(event),
            [],
        )
        node.x = 300
        early_time = target_time - 0.5
        early_y = round(node.radius + node.speed * (early_time - node.spawn_time))
        motion = FingertipMotion((300, early_y), (300, early_y), 0.0, None)
        remaining, hits, misses = update_challenge_nodes(
            [node], early_time, [motion], rhythm_round, 720
        )
        self.assertEqual((len(remaining), hits, misses), (1, [], 0))
        self.assertEqual(rhythm_round.resolved_count, 0)

        remaining, hits, misses = update_challenge_nodes(
            remaining, target_time + 0.421, [], rhythm_round, 720
        )
        self.assertEqual((len(remaining), hits, misses), (1, [], 1))
        self.assertEqual(rhythm_round.score.misses, 1)

        remaining, hits, misses = update_challenge_nodes(
            remaining,
            node.spawn_time + 720 / node.speed + 0.01,
            [],
            rhythm_round,
            720,
        )
        self.assertEqual((remaining, hits, misses), ([], [], 0))

    def test_a_complete_unplayed_challenge_reaches_results(self):
        rhythm_round = RhythmRound(started_at=0.0)
        active_nodes = []
        frame_width, frame_height = 640, 480
        radius = node_radius_for_frame(frame_width, frame_height)
        speed = max(
            1.0,
            (frame_height * CHALLENGE_TARGET_HEIGHT_RATIO - radius)
            / rhythm_round.node_travel_seconds,
        )
        final_time = (
            rhythm_round.spawn_time(rhythm_round.chart[-1])
            + frame_height / speed
            + 0.1
        )
        current_time = 0.0
        while current_time <= final_time:
            for event in rhythm_round.due_events(current_time):
                active_nodes.append(
                    create_challenge_node(
                        event,
                        rhythm_round,
                        frame_width,
                        frame_height,
                        current_time,
                        active_nodes,
                    )
                )
            active_nodes, _, _ = update_challenge_nodes(
                active_nodes, current_time, [], rhythm_round, frame_height
            )
            current_time += 0.05

        self.assertEqual(active_nodes, [])
        self.assertEqual(rhythm_round.score.misses, len(rhythm_round.chart))
        self.assertIs(rhythm_round.phase_at(final_time), RoundPhase.RESULTS)

    def test_song_node_keeps_one_speed_before_and_after_its_beat(self):
        event = ChartEvent(0, 64, 0.65)
        rhythm_round = RhythmRound(started_at=100.0, chart=(event,))
        node = create_challenge_node(
            event,
            rhythm_round,
            1280,
            720,
            rhythm_round.spawn_time(event),
            [],
        )
        target_time = rhythm_round.target_time(event)
        sample_seconds = 0.35

        before = challenge_node_y(node, target_time - sample_seconds)
        at_beat = challenge_node_y(node, target_time)
        after = challenge_node_y(node, target_time + sample_seconds)

        self.assertAlmostEqual(at_beat - before, node.speed * sample_seconds)
        self.assertAlmostEqual(after - at_beat, node.speed * sample_seconds)


if __name__ == "__main__":
    unittest.main()
