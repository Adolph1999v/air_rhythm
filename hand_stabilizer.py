"""Adaptive landmark smoothing and stable hand identities for Air Rhythm."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


DEFAULT_DROPOUT_GRACE_SECONDS = 0.14
DEFAULT_TRACK_RETENTION_SECONDS = 1.0
DEFAULT_FILTER_RESET_SECONDS = 0.25
DEFAULT_MIN_CUTOFF = 2.2
DEFAULT_SPEED_COEFFICIENT = 1.2
DERIVATIVE_CUTOFF = 1.0
NEAREST_TRACK_DISTANCE = 0.28
LABEL_REASSIGN_DISTANCE = 0.45
TRUSTED_LABEL_CONFIDENCE = 0.65


@dataclass(frozen=True)
class StabilizedLandmark:
    """The coordinates used by rendering, motion, and collision logic."""

    x: float
    y: float
    z: float


class StabilizedLandmarks(list[StabilizedLandmark]):
    """A MediaPipe-compatible landmark list carrying a persistent identity."""

    def __init__(
        self,
        points: Sequence[StabilizedLandmark],
        identity: str,
    ) -> None:
        super().__init__(points)
        self.identity = identity


@dataclass(frozen=True)
class HandDetectionView:
    """The small part of a MediaPipe result consumed by the application."""

    hand_landmarks: tuple[StabilizedLandmarks, ...]
    handedness: tuple[Any, ...]


@dataclass(frozen=True)
class StabilizedHandFrame:
    """Fresh hands for interaction plus briefly retained hands for rendering."""

    active: HandDetectionView
    visible: HandDetectionView
    stale_identities: tuple[str, ...]


class _AdaptiveLandmarkFilter:
    """One Euro-style low-pass filter that relaxes during fast movement."""

    def __init__(
        self,
        *,
        min_cutoff: float,
        speed_coefficient: float,
        reset_seconds: float,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.speed_coefficient = speed_coefficient
        self.reset_seconds = reset_seconds
        self._last_time: float | None = None
        self._last_raw: np.ndarray | None = None
        self._last_filtered: np.ndarray | None = None
        self._last_derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, elapsed_seconds: float):
        time_constant = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + time_constant / elapsed_seconds)

    def update(self, coordinates: np.ndarray, measured_at: float) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if (
            self._last_time is None
            or self._last_raw is None
            or self._last_filtered is None
            or self._last_derivative is None
            or coordinates.shape != self._last_raw.shape
            or measured_at - self._last_time <= 0
            or measured_at - self._last_time > self.reset_seconds
        ):
            self._last_time = measured_at
            self._last_raw = coordinates.copy()
            self._last_filtered = coordinates.copy()
            self._last_derivative = np.zeros_like(coordinates)
            return coordinates.copy()

        elapsed_seconds = min(0.1, measured_at - self._last_time)
        derivative = (coordinates - self._last_raw) / elapsed_seconds
        derivative_alpha = self._alpha(DERIVATIVE_CUTOFF, elapsed_seconds)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._last_derivative
        )
        planar_speed = np.linalg.norm(filtered_derivative[:, :2], axis=1)
        depth_speed = np.abs(filtered_derivative[:, 2])
        planar_cutoff = self.min_cutoff + self.speed_coefficient * planar_speed
        depth_cutoff = self.min_cutoff + self.speed_coefficient * depth_speed
        planar_alpha = self._alpha(planar_cutoff, elapsed_seconds)[:, None]
        depth_alpha = self._alpha(depth_cutoff, elapsed_seconds)[:, None]
        position_alpha = np.concatenate(
            (np.repeat(planar_alpha, 2, axis=1), depth_alpha),
            axis=1,
        )
        filtered = (
            position_alpha * coordinates
            + (1.0 - position_alpha) * self._last_filtered
        )

        self._last_time = measured_at
        self._last_raw = coordinates.copy()
        self._last_filtered = filtered
        self._last_derivative = filtered_derivative
        return filtered.copy()


@dataclass
class _HandTrack:
    identity: str
    landmark_filter: _AdaptiveLandmarkFilter
    landmarks: StabilizedLandmarks
    handedness: Any
    wrist_position: np.ndarray
    last_seen: float


@dataclass(frozen=True)
class _Candidate:
    coordinates: np.ndarray
    handedness: Any
    label: str | None
    label_confidence: float

    @property
    def wrist_position(self) -> np.ndarray:
        return self.coordinates[0, :2]


def _normalized_label(handedness: Any) -> str | None:
    if not handedness:
        return None
    category = handedness[0]
    label = getattr(category, "category_name", None)
    if not isinstance(label, str):
        return None
    normalized = label.strip().casefold()
    if normalized == "left":
        return "Left"
    if normalized == "right":
        return "Right"
    return None


def _handedness_confidence(handedness: Any) -> float:
    if not handedness:
        return 0.0
    try:
        score = float(getattr(handedness[0], "score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _coordinates(hand_landmarks: Sequence[Any]) -> np.ndarray | None:
    if len(hand_landmarks) < 21:
        return None
    values = []
    for landmark in hand_landmarks:
        try:
            point = (
                float(landmark.x),
                float(landmark.y),
                float(getattr(landmark, "z", 0.0)),
            )
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in point):
            return None
        values.append(point)
    return np.asarray(values, dtype=np.float64)


def _identity_order(identity: str) -> tuple[int, str]:
    if identity == "Left":
        return 0, identity
    if identity == "Right":
        return 1, identity
    return 2, identity


class HandLandmarkStabilizer:
    """Stabilize hand poses without allowing stale hands to trigger hits."""

    def __init__(
        self,
        *,
        dropout_grace_seconds: float = DEFAULT_DROPOUT_GRACE_SECONDS,
        track_retention_seconds: float = DEFAULT_TRACK_RETENTION_SECONDS,
        filter_reset_seconds: float = DEFAULT_FILTER_RESET_SECONDS,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        speed_coefficient: float = DEFAULT_SPEED_COEFFICIENT,
    ) -> None:
        settings = {
            "dropout_grace_seconds": dropout_grace_seconds,
            "track_retention_seconds": track_retention_seconds,
            "filter_reset_seconds": filter_reset_seconds,
            "min_cutoff": min_cutoff,
            "speed_coefficient": speed_coefficient,
        }
        for name, value in settings.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if track_retention_seconds < dropout_grace_seconds:
            raise ValueError("track retention must cover the dropout grace period")

        self.dropout_grace_seconds = float(dropout_grace_seconds)
        self.track_retention_seconds = float(track_retention_seconds)
        self.filter_reset_seconds = float(filter_reset_seconds)
        self.min_cutoff = float(min_cutoff)
        self.speed_coefficient = float(speed_coefficient)
        self._tracks: dict[str, _HandTrack] = {}
        self._next_unknown_identity = 1

    def _new_filter(self) -> _AdaptiveLandmarkFilter:
        return _AdaptiveLandmarkFilter(
            min_cutoff=self.min_cutoff,
            speed_coefficient=self.speed_coefficient,
            reset_seconds=self.filter_reset_seconds,
        )

    def _next_identity(self, preferred_label: str | None) -> str:
        if preferred_label is not None and preferred_label not in self._tracks:
            return preferred_label
        while True:
            identity = f"Hand-{self._next_unknown_identity}"
            self._next_unknown_identity += 1
            if identity not in self._tracks:
                return identity

    def _select_identity(
        self,
        candidate: _Candidate,
        assigned_identities: set[str],
    ) -> str:
        available = {
            identity: track
            for identity, track in self._tracks.items()
            if identity not in assigned_identities
        }
        nearest_identity = None
        nearest_distance = math.inf
        for identity, track in available.items():
            distance = float(
                np.linalg.norm(candidate.wrist_position - track.wrist_position)
            )
            if distance < nearest_distance:
                nearest_identity = identity
                nearest_distance = distance

        if (
            candidate.label in available
            and candidate.label_confidence >= TRUSTED_LABEL_CONFIDENCE
        ):
            return candidate.label
        if candidate.label in available:
            label_distance = float(
                np.linalg.norm(
                    candidate.wrist_position
                    - available[candidate.label].wrist_position
                )
            )
            if (
                label_distance <= LABEL_REASSIGN_DISTANCE
                or nearest_identity is None
                or nearest_distance > NEAREST_TRACK_DISTANCE
            ):
                return candidate.label
        if nearest_identity is not None and nearest_distance <= NEAREST_TRACK_DISTANCE:
            return nearest_identity
        return self._next_identity(candidate.label)

    def update(
        self,
        hand_landmarks: Sequence[Sequence[Any]],
        handedness: Sequence[Any],
        measured_at: float,
    ) -> StabilizedHandFrame:
        """Return fresh interaction hands and short-lived rendering hands."""
        measured_at = float(measured_at)
        if not math.isfinite(measured_at):
            raise ValueError("measured_at must be finite")

        expired = [
            identity
            for identity, track in self._tracks.items()
            if measured_at - track.last_seen > self.track_retention_seconds
        ]
        for identity in expired:
            del self._tracks[identity]

        candidates = []
        for index, landmarks in enumerate(hand_landmarks):
            coordinates = _coordinates(landmarks)
            if coordinates is None:
                continue
            classifications = handedness[index] if index < len(handedness) else ()
            candidates.append(
                _Candidate(
                    coordinates=coordinates,
                    handedness=classifications,
                    label=_normalized_label(classifications),
                    label_confidence=_handedness_confidence(classifications),
                )
            )

        # Confident Left/Right labels are considered first, so reversing
        # MediaPipe's result-list order cannot swap the visible stick colours.
        candidates.sort(
            key=lambda candidate: (
                0 if candidate.label in ("Left", "Right") else 1,
                _identity_order(candidate.label or ""),
            )
        )
        assigned: set[str] = set()
        active_entries: list[tuple[str, StabilizedLandmarks, Any]] = []
        for candidate in candidates:
            identity = self._select_identity(candidate, assigned)
            assigned.add(identity)
            track = self._tracks.get(identity)
            if track is None:
                track = _HandTrack(
                    identity=identity,
                    landmark_filter=self._new_filter(),
                    landmarks=StabilizedLandmarks((), identity),
                    handedness=candidate.handedness,
                    wrist_position=candidate.wrist_position.copy(),
                    last_seen=measured_at,
                )
                self._tracks[identity] = track

            filtered = track.landmark_filter.update(
                candidate.coordinates,
                measured_at,
            )
            landmarks = StabilizedLandmarks(
                tuple(
                    StabilizedLandmark(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                    )
                    for point in filtered
                ),
                identity,
            )
            track.landmarks = landmarks
            track.handedness = candidate.handedness
            track.wrist_position = candidate.wrist_position.copy()
            track.last_seen = measured_at
            active_entries.append((identity, landmarks, candidate.handedness))

        active_entries.sort(key=lambda entry: _identity_order(entry[0]))
        visible_entries = list(active_entries)
        stale_identities = []
        for identity, track in self._tracks.items():
            if identity in assigned or not track.landmarks:
                continue
            if measured_at - track.last_seen <= self.dropout_grace_seconds:
                visible_entries.append(
                    (identity, track.landmarks, track.handedness)
                )
                stale_identities.append(identity)
        visible_entries.sort(key=lambda entry: _identity_order(entry[0]))

        return StabilizedHandFrame(
            active=HandDetectionView(
                hand_landmarks=tuple(entry[1] for entry in active_entries),
                handedness=tuple(entry[2] for entry in active_entries),
            ),
            visible=HandDetectionView(
                hand_landmarks=tuple(entry[1] for entry in visible_entries),
                handedness=tuple(entry[2] for entry in visible_entries),
            ),
            stale_identities=tuple(
                sorted(stale_identities, key=_identity_order)
            ),
        )

    def reset(self) -> None:
        """Forget all previous landmarks and hand identities."""
        self._tracks.clear()
        self._next_unknown_identity = 1


__all__ = [
    "DEFAULT_DROPOUT_GRACE_SECONDS",
    "HandDetectionView",
    "HandLandmarkStabilizer",
    "StabilizedHandFrame",
    "StabilizedLandmark",
    "StabilizedLandmarks",
]
