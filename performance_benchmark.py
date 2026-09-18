"""Privacy-safe performance measurement for Air Rhythm.

Only durations, counters, resolutions, and environment details are stored.
Camera images and MediaPipe landmark coordinates never enter a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any


DEFAULT_TARGET_FPS = 30.0
METRIC_NAMES = (
    "camera_capture_ms",
    "mediapipe_inference_ms",
    "game_update_ms",
    "rendering_ms",
    "complete_frame_ms",
    "frame_start_to_audio_request_ms",
)
METRIC_LABELS = {
    "camera_capture_ms": "Camera capture",
    "mediapipe_inference_ms": "MediaPipe inference",
    "game_update_ms": "Game update",
    "rendering_ms": "Rendering and display submission",
    "complete_frame_ms": "Complete frame pipeline",
    "frame_start_to_audio_request_ms": "Frame start to audio request (hit frames)",
}


def _finite_non_negative(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative finite number") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return an interpolated percentile without requiring NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Summarize one timing series for reports and tests."""
    if not values:
        return {
            "samples": 0,
            "average": None,
            "median": None,
            "p95": None,
            "best": None,
            "worst": None,
        }
    return {
        "samples": len(values),
        "average": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "best": min(values),
        "worst": max(values),
    }


@dataclass
class PerformanceBenchmark:
    """Collect timing samples only while an explicit session is active."""

    target_fps: float = DEFAULT_TARGET_FPS
    active: bool = field(default=False, init=False)
    started_at: float | None = field(default=None, init=False)
    started_at_utc: str | None = field(default=None, init=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False)
    metrics: dict[str, list[float]] = field(default_factory=dict, init=False)
    frame_intervals_ms: list[float] = field(default_factory=list, init=False)
    slow_frame_count: int = field(default=0, init=False)
    estimated_dropped_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.target_fps = _finite_non_negative(self.target_fps, "target_fps")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self._reset_samples()

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.target_fps

    @property
    def frame_count(self) -> int:
        return len(self.metrics["complete_frame_ms"])

    def _reset_samples(self) -> None:
        self.metrics = {name: [] for name in METRIC_NAMES}
        self.frame_intervals_ms = []
        self.slow_frame_count = 0
        self.estimated_dropped_frames = 0

    def start(
        self,
        *,
        now: float | None = None,
        metadata: dict[str, Any] | None = None,
        started_at_utc: str | None = None,
    ) -> None:
        """Start a fresh session, discarding samples from an older session."""
        if self.active:
            raise RuntimeError("benchmark is already active")
        started = time.perf_counter() if now is None else float(now)
        if not math.isfinite(started):
            raise ValueError("start time must be finite")
        self._reset_samples()
        self.started_at = started
        self.started_at_utc = started_at_utc or datetime.now(timezone.utc).isoformat()
        self.metadata = dict(metadata or {})
        self.active = True

    def record_frame(
        self,
        *,
        camera_capture_ms: float,
        mediapipe_inference_ms: float,
        game_update_ms: float,
        rendering_ms: float,
        complete_frame_ms: float,
        frame_interval_ms: float,
    ) -> None:
        """Record one complete frame when the session is active."""
        if not self.active:
            return
        samples = {
            "camera_capture_ms": camera_capture_ms,
            "mediapipe_inference_ms": mediapipe_inference_ms,
            "game_update_ms": game_update_ms,
            "rendering_ms": rendering_ms,
            "complete_frame_ms": complete_frame_ms,
        }
        validated_samples = {
            name: _finite_non_negative(value, name)
            for name, value in samples.items()
        }
        interval = _finite_non_negative(frame_interval_ms, "frame_interval_ms")
        for name, value in validated_samples.items():
            self.metrics[name].append(value)
        if interval > 0:
            self.frame_intervals_ms.append(interval)
            if interval > self.frame_budget_ms:
                self.slow_frame_count += 1
            self.estimated_dropped_frames += max(
                0,
                int(interval / self.frame_budget_ms) - 1,
            )

    def record_audio_request(self, latency_ms: float) -> None:
        """Record approximate frame-start-to-audio-request latency for a hit."""
        if not self.active:
            return
        self.metrics["frame_start_to_audio_request_ms"].append(
            _finite_non_negative(
                latency_ms,
                "frame_start_to_audio_request_ms",
            )
        )

    def elapsed_seconds(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        current = time.perf_counter() if now is None else float(now)
        if not math.isfinite(current):
            return 0.0
        return max(0.0, current - self.started_at)

    def live_summary(self, now: float | None = None) -> dict[str, Any]:
        """Return compact counters suitable for the on-screen badge."""
        frame_summary = summarize_values(self.frame_intervals_ms)
        average_interval = frame_summary["average"]
        average_fps = (
            1000.0 / average_interval
            if isinstance(average_interval, (int, float)) and average_interval > 0
            else None
        )
        return {
            "active": self.active,
            "elapsed_seconds": self.elapsed_seconds(now),
            "frames": self.frame_count,
            "average_fps": average_fps,
            "slow_frames": self.slow_frame_count,
            "estimated_dropped_frames": self.estimated_dropped_frames,
            "audio_samples": len(self.metrics["frame_start_to_audio_request_ms"]),
        }

    def stop(self, *, now: float | None = None) -> dict[str, Any]:
        """Stop and return a serializable benchmark report."""
        if not self.active or self.started_at is None:
            raise RuntimeError("benchmark is not active")
        stopped = time.perf_counter() if now is None else float(now)
        if not math.isfinite(stopped):
            raise ValueError("stop time must be finite")
        duration_seconds = max(0.0, stopped - self.started_at)
        self.active = False

        timing_summaries = {
            name: summarize_values(self.metrics[name])
            for name in METRIC_NAMES
        }
        fps_values = [
            1000.0 / interval
            for interval in self.frame_intervals_ms
            if interval > 0
        ]
        fps_summary = summarize_values(fps_values)
        fps_summary["minimum"] = min(fps_values) if fps_values else None
        fps_summary["maximum"] = max(fps_values) if fps_values else None
        slow_percent = (
            self.slow_frame_count / len(self.frame_intervals_ms) * 100.0
            if self.frame_intervals_ms
            else 0.0
        )
        environment = {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        return {
            "schema_version": 1,
            "started_at_utc": self.started_at_utc,
            "duration_seconds": duration_seconds,
            "target_fps": self.target_fps,
            "frame_budget_ms": self.frame_budget_ms,
            "frame_count": self.frame_count,
            "slow_frame_count": self.slow_frame_count,
            "slow_frame_percent": slow_percent,
            "estimated_dropped_frames": self.estimated_dropped_frames,
            "fps": fps_summary,
            "timings_ms": timing_summaries,
            "environment": environment,
            "metadata": dict(self.metadata),
            "privacy": {
                "camera_images_saved": False,
                "landmark_coordinates_saved": False,
            },
            "limitations": [
                "Frame start to audio request is software-path timing, not measured speaker output latency.",
                "Estimated dropped frames are inferred from frame intervals against the target frame budget.",
            ],
        }


def _format_number(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def benchmark_markdown(report: dict[str, Any]) -> str:
    """Create a readable portfolio report from a benchmark dictionary."""
    fps = report.get("fps", {})
    lines = [
        "# Air Rhythm Performance Benchmark",
        "",
        f"- Started: {report.get('started_at_utc') or '--'}",
        f"- Duration: {_format_number(report.get('duration_seconds'))} seconds",
        f"- Recorded frames: {int(report.get('frame_count', 0))}",
        f"- Target: {_format_number(report.get('target_fps'), 1)} FPS",
        f"- Average FPS: {_format_number(fps.get('average'))}",
        f"- Minimum FPS: {_format_number(fps.get('minimum'))}",
        f"- Slow frames: {int(report.get('slow_frame_count', 0))} "
        f"({_format_number(report.get('slow_frame_percent'))}%)",
        f"- Estimated dropped frames: {int(report.get('estimated_dropped_frames', 0))}",
        "",
        "## Pipeline timings",
        "",
        "| Stage | Samples | Average ms | Median ms | P95 ms | Worst ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    timings = report.get("timings_ms", {})
    for name in METRIC_NAMES:
        summary = timings.get(name, {})
        lines.append(
            f"| {METRIC_LABELS[name]} | {int(summary.get('samples', 0))} | "
            f"{_format_number(summary.get('average'))} | "
            f"{_format_number(summary.get('median'))} | "
            f"{_format_number(summary.get('p95'))} | "
            f"{_format_number(summary.get('worst'))} |"
        )

    lines.extend(("", "## Test settings", ""))
    combined_settings = {
        **dict(report.get("environment", {})),
        **dict(report.get("metadata", {})),
    }
    for key, value in combined_settings.items():
        label = str(key).replace("_", " ").title()
        lines.append(f"- {label}: {value}")

    lines.extend(
        (
            "",
            "## Privacy and limitations",
            "",
            "- No camera images are stored.",
            "- No hand-landmark coordinates are stored.",
        )
    )
    for limitation in report.get("limitations", ()):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def save_benchmark_report(
    report: dict[str, Any],
    directory: str | Path = "benchmark_reports",
) -> tuple[Path, Path]:
    """Write matching JSON and Markdown reports and return both paths."""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    json_path = output_directory / f"performance-{timestamp}.json"
    markdown_path = output_directory / f"performance-{timestamp}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(benchmark_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "DEFAULT_TARGET_FPS",
    "METRIC_LABELS",
    "METRIC_NAMES",
    "PerformanceBenchmark",
    "benchmark_markdown",
    "save_benchmark_report",
    "summarize_values",
]
