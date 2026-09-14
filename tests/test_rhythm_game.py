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
        self.assertEqual(DEFAULT_LEAD_IN_SECONDS, NODE_TRAVEL_SECONDS)
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

        self.assertEqual(score.record_hit(TimingGrade.PERFECT, "GOOD"), 1500)
        self.assertEqual(score.record_hit(TimingGrade.GREAT, "PERFECT"), 1350)
        self.assertEqual(score.combo, 2)
        score.record_miss()
        self.assertEqual(score.combo, 0)
        self.assertEqual(score.max_combo, 2)
        self.assertEqual(score.record_hit(TimingGrade.GOOD, "GREAT"), 1140)
        self.assertEqual(score.score, 3990)
        self.assertEqual(score.total_judged, 4)
        self.assertAlmostEqual(score.completion_accuracy, 75.0)
        self.assertAlmostEqual(score.timing_accuracy, 56.25)
        self.assertEqual(score.rank, "A")

    def test_basic_contact_is_a_hit_and_continues_the_combo(self):
        score = RoundScore()
        score.record_hit(TimingGrade.PERFECT)

        self.assertEqual(score.record_hit(TimingGrade.HIT, "PERFECT"), 1100)
        self.assertEqual(score.total_hits, 2)
        self.assertEqual(score.basic_hits, 1)
        self.assertEqual(score.misses, 0)
        self.assertEqual(score.combo, 2)
        self.assertAlmostEqual(score.completion_accuracy, 100.0)
        self.assertAlmostEqual(score.timing_accuracy, 62.5)

    def test_catching_the_full_melody_gives_full_completion_and_a_strong_score(self):
        score = RoundScore()
        for _ in MELODY_NOTES:
            score.record_hit(TimingGrade.HIT)

        self.assertEqual(score.total_hits, len(MELODY_NOTES))
        self.assertEqual(score.score, 40_900)
        self.assertEqual(score.completion_accuracy, 100.0)
        self.assertEqual(score.rank, "S")

    def test_ranks_follow_circle_completion_thresholds(self):
        scores = []
        for hits, misses in ((9, 1), (3, 1), (3, 2), (1, 1)):
            score = RoundScore()
            for _ in range(hits):
                score.record_hit(TimingGrade.HIT)
            for _ in range(misses):
                score.record_miss()
            scores.append(score)

        self.assertEqual([score.rank for score in scores], ["S", "A", "B", "C"])


class RhythmRoundTests(unittest.TestCase):
    def setUp(self):
        self.chart = (
            ChartEvent(0, 60, 2.0),
            ChartEvent(1, 62, 2.4),
            ChartEvent(2, 64, 2.8),
        )
        self.round = RhythmRound(started_at=100.0, chart=self.chart)

    def test_first_circle_enters_exactly_when_the_countdown_reaches_go(self):
        self.assertIs(self.round.phase_at(102.999), RoundPhase.COUNTDOWN)
        self.assertIs(self.round.phase_at(103.0), RoundPhase.PLAYING)
        self.assertEqual(self.round.countdown_remaining(101.5), 1.5)
        self.assertEqual(self.round.countdown_remaining(103.0), 0.0)
        self.assertAlmostEqual(self.round.target_time(0), 105.0)
        self.assertAlmostEqual(
            self.round.spawn_time(0),
            self.round.target_time(0) - NODE_TRAVEL_SECONDS,
        )
        self.assertEqual(self.round.spawn_time(0), self.round.song_start_time)
        self.assertEqual(self.round.due_events(102.999), ())
        self.assertEqual(self.round.due_events(103.0), (self.chart[0],))

    def test_default_song_chart_also_holds_its_first_circle_until_go(self):
        default_round = RhythmRound(started_at=100.0)

        self.assertEqual(default_round.due_events(102.999), ())
        self.assertEqual(
            default_round.due_events(default_round.song_start_time),
            (default_round.chart[0],),
        )

    def test_ui_progress_uses_song_clock_and_clamps(self):
        self.assertEqual(self.round.progress_at(99.0), 0.0)
        self.assertEqual(self.round.progress_at(self.round.song_start_time), 0.0)
        midway = self.round.song_start_time + (
            self.chart[-1].target_offset + GOOD_WINDOW_SECONDS
        ) / 2
        self.assertAlmostEqual(self.round.progress_at(midway), 0.5)
        self.assertEqual(self.round.progress_at(999.0), 1.0)
        with self.assertRaises(ValueError):
            self.round.progress_at(float("nan"))

    def test_help_pause_moves_future_beats_without_resetting_progress(self):
        original_target = self.round.target_time(1)
        self.round.due_events(103.06)
        self.round.judge_hit(0, self.round.target_time(0))

        self.round.delay_timeline(2.5)

        self.assertAlmostEqual(self.round.target_time(1), original_target + 2.5)
        self.assertEqual(self.round.resolved_count, 1)
        self.assertEqual(self.round.score.perfect, 1)
        self.assertEqual(self.round.due_events(105.89), ())
        self.assertEqual(self.round.due_events(105.90), (self.chart[1],))
        with self.assertRaises(ValueError):
            self.round.delay_timeline(-0.1)

    def test_one_slow_camera_frame_emits_every_node_that_became_due(self):
        self.assertEqual(self.round.due_events(102.999), ())
        self.assertEqual(
            self.round.due_events(103.45),
            self.chart[:2],
        )
        self.assertEqual(self.round.due_events(103.45), ())
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
        self.assertIs(
            self.round.judge_hit(2, self.round.target_time(2) + 0.5),
            TimingGrade.HIT,
        )
        self.assertIs(self.round.phase_at(999.0), RoundPhase.RESULTS)
        self.assertEqual(self.round.score.total_judged, 3)
        self.assertEqual(self.round.score.total_hits, 3)
        self.assertEqual(self.round.score.basic_hits, 1)
        self.assertEqual(self.round.score.misses, 0)

    def test_unhit_event_becomes_a_miss_only_when_explicitly_recorded(self):
        self.assertFalse(self.round.is_resolved(0))
        self.assertTrue(self.round.record_miss(0))
        self.assertEqual(self.round.score.misses, 1)
        self.assertFalse(self.round.record_miss(0))
        self.assertEqual(self.round.score.misses, 1)

    def test_reset_restarts_schedule_and_score_at_a_new_time(self):
        self.round.due_events(999.0)
        self.round.judge_hit(0, self.round.target_time(0))
        self.round.reset(started_at=200.0)

        self.assertEqual(self.round.score, RoundScore())
        self.assertEqual(self.round.resolved_count, 0)
        self.assertEqual(self.round.remaining_count, 3)
        self.assertIs(self.round.phase_at(200.0), RoundPhase.COUNTDOWN)
        self.assertEqual(self.round.due_events(202.999), ())
        self.assertEqual(self.round.due_events(203.0), (self.chart[0],))


if __name__ == "__main__":
    unittest.main()
