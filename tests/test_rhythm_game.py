"""Checks for Air Rhythm's frame-independent song clock and scoring rules."""

import unittest

from music import MELODY_NOTES, MELODY_STEP_SECONDS
from rhythm_game import (
    DEFAULT_LEAD_IN_SECONDS,
    GOOD_WINDOW_SECONDS,
    NODE_TRAVEL_SECONDS,
    ChartEvent,
    RhythmRound,
    RoundPhase,
    RoundScore,
    TimingGrade,
    build_melody_chart,
    grade_timing,
)


class MelodyChartTests(unittest.TestCase):
    def test_chart_preserves_pitches_and_stretches_the_original_rhythm(self):
        chart = build_melody_chart(tempo_scale=2.0)

        self.assertEqual(tuple(event.pitch for event in chart), MELODY_NOTES)
        self.assertEqual(chart[0].target_offset, DEFAULT_LEAD_IN_SECONDS)
        self.assertAlmostEqual(
            chart[1].target_offset,
            DEFAULT_LEAD_IN_SECONDS + MELODY_STEP_SECONDS[0] * 2.0,
        )
        self.assertAlmostEqual(
            chart[9].target_offset - chart[8].target_offset,
            MELODY_STEP_SECONDS[8] * 2.0,
        )
        self.assertTrue(all(event.instrument == "keys" for event in chart))

    def test_chart_rejects_invalid_speed_and_lead_in(self):
        for tempo_scale in (0, -1, float("inf")):
            with self.subTest(tempo_scale=tempo_scale):
                with self.assertRaises(ValueError):
                    build_melody_chart(tempo_scale=tempo_scale)
        with self.assertRaises(ValueError):
            build_melody_chart(lead_in_seconds=-0.1)


class TimingGradeTests(unittest.TestCase):
    def test_every_timing_boundary_is_inclusive_for_early_and_late_hits(self):
        expectations = (
            (0.0, TimingGrade.PERFECT),
            (0.10, TimingGrade.PERFECT),
            (-0.10, TimingGrade.PERFECT),
            (0.1001, TimingGrade.GREAT),
            (0.22, TimingGrade.GREAT),
            (-0.22, TimingGrade.GREAT),
            (0.2201, TimingGrade.GOOD),
            (0.42, TimingGrade.GOOD),
            (-0.42, TimingGrade.GOOD),
            (0.4201, None),
        )
        for error, expected in expectations:
            with self.subTest(error=error):
                self.assertIs(grade_timing(error), expected)


class RoundScoreTests(unittest.TestCase):
    def test_combo_and_movement_add_small_bonuses_while_miss_breaks_streak(self):
        score = RoundScore()

        self.assertEqual(score.record_hit(TimingGrade.PERFECT, "GOOD"), 1000)
        self.assertEqual(score.record_hit(TimingGrade.GREAT, "PERFECT"), 800)
        self.assertEqual(score.combo, 2)
        score.record_miss()
        self.assertEqual(score.combo, 0)
        self.assertEqual(score.max_combo, 2)
        self.assertEqual(score.record_hit(TimingGrade.GOOD, "GREAT"), 440)
        self.assertEqual(score.score, 2240)
        self.assertEqual(score.total_judged, 4)
        self.assertAlmostEqual(score.accuracy, 56.25)
        self.assertEqual(score.rank, "C")

    def test_ranks_follow_accuracy_thresholds(self):
        perfect = RoundScore()
        perfect.record_hit(TimingGrade.PERFECT)
        self.assertEqual(perfect.rank, "S")

        great = RoundScore()
        great.record_hit(TimingGrade.GREAT)
        self.assertEqual(great.rank, "A")

        mixed = RoundScore()
        mixed.record_hit(TimingGrade.GREAT)
        mixed.record_hit(TimingGrade.GOOD)
        self.assertEqual(mixed.rank, "B")


class RhythmRoundTests(unittest.TestCase):
    def setUp(self):
        self.chart = (
            ChartEvent(0, 60, 0.65),
            ChartEvent(1, 62, 1.05),
            ChartEvent(2, 64, 1.45),
        )
        self.round = RhythmRound(started_at=100.0, chart=self.chart)

    def test_countdown_boundary_and_first_target_leave_a_clear_go_moment(self):
        self.assertIs(self.round.phase_at(102.999), RoundPhase.COUNTDOWN)
        self.assertIs(self.round.phase_at(103.0), RoundPhase.PLAYING)
        self.assertEqual(self.round.countdown_remaining(101.5), 1.5)
        self.assertEqual(self.round.countdown_remaining(103.0), 0.0)
        self.assertAlmostEqual(self.round.target_time(0), 103.65)
        self.assertAlmostEqual(
            self.round.spawn_time(0),
            self.round.target_time(0) - NODE_TRAVEL_SECONDS,
        )
        self.assertLess(self.round.spawn_time(0), self.round.song_start_time)

    def test_one_slow_camera_frame_emits_every_node_that_became_due(self):
        self.assertEqual(self.round.due_events(101.64), ())
        self.assertEqual(
            self.round.due_events(102.06),
            self.chart[:2],
        )
        self.assertEqual(self.round.due_events(102.06), ())
        self.assertEqual(self.round.due_events(999.0), self.chart[2:])

    def test_hits_use_absolute_target_time_and_results_wait_for_resolution(self):
        self.round.due_events(999.0)
        self.assertIs(
            self.round.judge_hit(0, self.round.target_time(0), "PERFECT"),
            TimingGrade.PERFECT,
        )
        self.assertIs(
            self.round.judge_hit(1, self.round.target_time(1) + 0.2, "GOOD"),
            TimingGrade.GREAT,
        )
        self.assertIs(self.round.phase_at(999.0), RoundPhase.PLAYING)
        self.assertIsNone(
            self.round.judge_hit(2, self.round.target_time(2) + 0.5)
        )
        self.assertIs(self.round.phase_at(999.0), RoundPhase.RESULTS)
        self.assertEqual(self.round.score.total_judged, 3)
        self.assertEqual(self.round.score.misses, 1)

    def test_expired_windows_become_misses_once(self):
        cutoff = self.round.target_time(1) + GOOD_WINDOW_SECONDS
        self.assertEqual(self.round.expire_misses(cutoff), (self.chart[0],))
        self.assertEqual(
            self.round.expire_misses(cutoff + 0.001),
            (self.chart[1],),
        )
        self.assertEqual(self.round.score.misses, 2)
        self.assertFalse(self.round.record_miss(0))
        self.assertEqual(self.round.score.misses, 2)

    def test_reset_restarts_schedule_and_score_at_a_new_time(self):
        self.round.due_events(999.0)
        self.round.judge_hit(0, self.round.target_time(0))
        self.round.reset(started_at=200.0)

        self.assertEqual(self.round.score, RoundScore())
        self.assertEqual(self.round.resolved_count, 0)
        self.assertEqual(self.round.remaining_count, 3)
        self.assertIs(self.round.phase_at(200.0), RoundPhase.COUNTDOWN)
        self.assertEqual(self.round.due_events(201.64), ())
        self.assertEqual(self.round.due_events(201.65), (self.chart[0],))


if __name__ == "__main__":
    unittest.main()
