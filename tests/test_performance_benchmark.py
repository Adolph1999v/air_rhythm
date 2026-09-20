"""Deterministic checks for privacy-safe performance measurement."""

import json
from pathlib import Path
import tempfile
import unittest

from performance_benchmark import (
    PerformanceBenchmark,
    benchmark_markdown,
    save_benchmark_report,
    summarize_values,
)


class PerformanceBenchmarkTests(unittest.TestCase):
    def test_summary_reports_average_median_p95_best_and_worst(self):
        summary = summarize_values([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["average"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertEqual(summary["best"], 1.0)
        self.assertEqual(summary["worst"], 4.0)

    def test_session_counts_slow_frames_and_estimated_drops(self):
        benchmark = PerformanceBenchmark(target_fps=30.0)
        benchmark.start(
            now=10.0,
            started_at_utc="2026-09-19T10:00:00+00:00",
            metadata={"camera_resolution": "1280x720"},
        )
        for interval in (20.0, 40.0, 70.0):
            benchmark.record_frame(
                camera_capture_ms=2.0,
                camera_wait_ms=0.4,
                camera_preprocessing_ms=1.0,
                mediapipe_inference_ms=8.0,
                game_update_ms=1.5,
                rendering_ms=5.0,
                complete_frame_ms=16.5,
                frame_interval_ms=interval,
                camera_frames_skipped=1 if interval == 70.0 else 0,
            )
        benchmark.record_audio_request(12.5)

        live = benchmark.live_summary(now=11.0)
        self.assertEqual(live["frames"], 3)
        self.assertEqual(live["slow_frames"], 2)
        self.assertEqual(live["estimated_dropped_frames"], 1)
        self.assertEqual(live["camera_frames_skipped"], 1)
        self.assertEqual(live["audio_samples"], 1)

        report = benchmark.stop(now=12.0)
        self.assertFalse(benchmark.active)
        self.assertEqual(report["duration_seconds"], 2.0)
        self.assertEqual(report["frame_count"], 3)
        self.assertEqual(report["slow_frame_count"], 2)
        self.assertEqual(report["estimated_dropped_frames"], 1)
        self.assertEqual(report["camera_frames_skipped"], 1)
        self.assertEqual(
            report["timings_ms"]["camera_capture_ms"]["average"],
            2.0,
        )
        self.assertAlmostEqual(
            report["timings_ms"]["camera_wait_ms"]["average"],
            0.4,
        )
        self.assertEqual(
            report["timings_ms"]["camera_preprocessing_ms"]["average"],
            1.0,
        )
        self.assertEqual(
            report["timings_ms"]["frame_start_to_audio_request_ms"]["samples"],
            1,
        )
        self.assertFalse(report["privacy"]["camera_images_saved"])
        self.assertFalse(report["privacy"]["landmark_coordinates_saved"])

    def test_inactive_session_ignores_samples_and_validates_values(self):
        benchmark = PerformanceBenchmark()
        benchmark.record_audio_request(10.0)
        self.assertEqual(benchmark.live_summary()["audio_samples"], 0)
        with self.assertRaises(RuntimeError):
            benchmark.stop()

        benchmark.start(now=1.0)
        with self.assertRaises(ValueError):
            benchmark.record_frame(
                camera_capture_ms=-1.0,
                camera_wait_ms=1.0,
                camera_preprocessing_ms=1.0,
                mediapipe_inference_ms=1.0,
                game_update_ms=1.0,
                rendering_ms=1.0,
                complete_frame_ms=1.0,
                frame_interval_ms=1.0,
            )

    def test_reports_save_as_readable_json_and_markdown(self):
        benchmark = PerformanceBenchmark()
        benchmark.start(now=1.0, started_at_utc="2026-09-19T10:00:00+00:00")
        benchmark.record_frame(
            camera_capture_ms=2.0,
            camera_wait_ms=0.5,
            camera_preprocessing_ms=1.0,
            mediapipe_inference_ms=7.0,
            game_update_ms=1.0,
            rendering_ms=4.0,
            complete_frame_ms=14.0,
            frame_interval_ms=33.0,
        )
        report = benchmark.stop(now=2.0)
        markdown = benchmark_markdown(report)
        self.assertIn("Average FPS", markdown)
        self.assertIn("MediaPipe inference", markdown)
        self.assertIn("Main-loop wait for fresh camera frame", markdown)
        self.assertIn("Camera preparation", markdown)
        self.assertIn("No camera images are stored", markdown)

        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = save_benchmark_report(report, directory)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            saved_report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["frame_count"], 1)
            self.assertIn(
                "Pipeline timings",
                Path(markdown_path).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
