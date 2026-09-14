"""Pure timing and scoring rules for Air Rhythm's short song challenge.

This module does not know about cameras, drawing, or speakers.  Keeping the
song clock here makes its behaviour deterministic and easy to test, even when
the camera occasionally takes longer to produce a frame.
"""

from dataclasses import dataclass, field
from enum import Enum
import math

from music import MELODY_NOTES, MELODY_STEP_SECONDS


DEFAULT_TEMPO_SCALE = 1.75
DEFAULT_COUNTDOWN_SECONDS = 3.0
NODE_TRAVEL_SECONDS = 2.0
# The first circle must enter only once the countdown reaches GO.  Its target
# therefore sits one full travel duration after GO, giving it the same two
# seconds of on-screen travel as every later circle.
DEFAULT_LEAD_IN_SECONDS = NODE_TRAVEL_SECONDS

PERFECT_WINDOW_SECONDS = 0.10
GREAT_WINDOW_SECONDS = 0.22
GOOD_WINDOW_SECONDS = 0.42


class RoundPhase(Enum):
    """The three timing phases inside one song challenge."""

    COUNTDOWN = "COUNTDOWN"
    PLAYING = "PLAYING"
    RESULTS = "RESULTS"


@dataclass(frozen=True)
class ChartEvent:
    """One scheduled circle and the note it should play."""

    index: int
    pitch: int
    target_offset: float
    instrument: str = "keys"


def build_melody_chart(
    tempo_scale: float = DEFAULT_TEMPO_SCALE,
    lead_in_seconds: float = DEFAULT_LEAD_IN_SECONDS,
) -> tuple[ChartEvent, ...]:
    """Turn the melody into absolute offsets from the song's GO moment.

    ``tempo_scale`` stretches every gap without changing the tune.  A value
    above 1 makes this first challenge slower and easier to play.  By default,
    the first target is one node-travel duration after GO, so its circle enters
    at the top of the screen at the exact moment the countdown ends.
    """
    if not math.isfinite(tempo_scale) or tempo_scale <= 0:
        raise ValueError("tempo_scale must be a positive, finite number")
    if not math.isfinite(lead_in_seconds) or lead_in_seconds < 0:
        raise ValueError("lead_in_seconds must be a non-negative, finite number")

    target_offset = lead_in_seconds
    chart = []
    for index, pitch in enumerate(MELODY_NOTES):
        chart.append(
            ChartEvent(
                index=index,
                pitch=pitch,
                target_offset=target_offset,
            )
        )
        # The final duration is useful for an audio preview, but there is no
        # following game node that needs it.
        if index < len(MELODY_NOTES) - 1:
            target_offset += MELODY_STEP_SECONDS[index] * tempo_scale
    return tuple(chart)


class TimingGrade(Enum):
    """Timing feedback, score bonus, and OpenCV-friendly BGR color."""

    PERFECT = ("PERFECT", 500, (80, 235, 110), 1.0)
    GREAT = ("GREAT", 250, (40, 210, 255), 0.75)
    GOOD = ("GOOD", 100, (255, 210, 80), 0.5)
    HIT = ("HIT", 0, (210, 145, 255), 0.25)

    def __init__(
        self,
        label: str,
        score_bonus: int,
        color: tuple[int, int, int],
        accuracy_weight: float,
    ) -> None:
        self.label = label
        self.score_bonus = score_bonus
        self.color = color
        self.accuracy_weight = accuracy_weight


def grade_timing(error_seconds: float) -> TimingGrade | None:
    """Grade how far a hit was from its target time, early or late."""
    if not math.isfinite(error_seconds):
        raise ValueError("error_seconds must be finite")

    distance = abs(error_seconds)
    if distance <= PERFECT_WINDOW_SECONDS:
        return TimingGrade.PERFECT
    if distance <= GREAT_WINDOW_SECONDS:
        return TimingGrade.GREAT
    if distance <= GOOD_WINDOW_SECONDS:
        return TimingGrade.GOOD
    return None


MOVEMENT_BONUS = {
    "GOOD": 0,
    "GREAT": 40,
    "PERFECT": 80,
}
BASE_HIT_POINTS = 1000
COMBO_BONUS_PER_HIT = 20
MAX_COMBO_BONUS = 200


@dataclass
class RoundScore:
    """Score and streak totals for one play-through."""

    score: int = 0
    combo: int = 0
    max_combo: int = 0
    perfect: int = 0
    great: int = 0
    good: int = 0
    basic_hits: int = 0
    misses: int = 0

    def record_hit(
        self,
        grade: TimingGrade,
        movement_rating: str | None = None,
    ) -> int:
        """Record a timed hit and return the points awarded for that hit."""
        if not isinstance(grade, TimingGrade):
            raise TypeError("grade must be a TimingGrade")

        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)

        if grade is TimingGrade.PERFECT:
            self.perfect += 1
        elif grade is TimingGrade.GREAT:
            self.great += 1
        elif grade is TimingGrade.GOOD:
            self.good += 1
        else:
            self.basic_hits += 1

        combo_bonus = min(
            (self.combo - 1) * COMBO_BONUS_PER_HIT,
            MAX_COMBO_BONUS,
        )
        movement_key = (movement_rating or "").upper()
        points = (
            BASE_HIT_POINTS
            + grade.score_bonus
            + combo_bonus
            + MOVEMENT_BONUS.get(movement_key, 0)
        )
        self.score += points
        return points

    def record_miss(self) -> None:
        """Break the current streak and count an unresolved note as missed."""
        self.misses += 1
        self.combo = 0

    @property
    def total_judged(self) -> int:
        """Return the number of notes that have received a final result."""
        return self.total_hits + self.misses

    @property
    def total_hits(self) -> int:
        """Return every touched note, including contacts outside the beat window."""
        return self.perfect + self.great + self.good + self.basic_hits

    @property
    def completion_accuracy(self) -> float:
        """Return the percentage of circles that were successfully touched."""
        if self.total_judged == 0:
            return 0.0
        return self.total_hits / self.total_judged * 100.0

    @property
    def timing_accuracy(self) -> float:
        """Return optional timing quality without reducing hit completion."""
        if self.total_judged == 0:
            return 0.0
        weighted_hits = (
            self.perfect * TimingGrade.PERFECT.accuracy_weight
            + self.great * TimingGrade.GREAT.accuracy_weight
            + self.good * TimingGrade.GOOD.accuracy_weight
            + self.basic_hits * TimingGrade.HIT.accuracy_weight
        )
        return weighted_hits / self.total_judged * 100.0

    @property
    def rank(self) -> str:
        """Turn circle completion into a simple showcase-friendly rank."""
        if self.completion_accuracy >= 90:
            return "S"
        if self.completion_accuracy >= 75:
            return "A"
        if self.completion_accuracy >= 60:
            return "B"
        return "C"


@dataclass
class RhythmRound:
    """Own the clock, emitted events, judgements, and score for one round."""

    started_at: float
    countdown_seconds: float = DEFAULT_COUNTDOWN_SECONDS
    tempo_scale: float = DEFAULT_TEMPO_SCALE
    node_travel_seconds: float = NODE_TRAVEL_SECONDS
    chart: tuple[ChartEvent, ...] | None = None
    score: RoundScore = field(init=False)
    _next_spawn_position: int = field(init=False, repr=False)
    _resolved_indices: set[int] = field(init=False, repr=False)
    _events_by_index: dict[int, ChartEvent] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.started_at):
            raise ValueError("started_at must be finite")
        if not math.isfinite(self.countdown_seconds) or self.countdown_seconds < 0:
            raise ValueError("countdown_seconds must be non-negative and finite")
        if not math.isfinite(self.node_travel_seconds) or self.node_travel_seconds <= 0:
            raise ValueError("node_travel_seconds must be positive and finite")

        if self.chart is None:
            self.chart = build_melody_chart(self.tempo_scale)
        else:
            self.chart = tuple(self.chart)
        self._validate_chart()
        self.reset(self.started_at)

    def _validate_chart(self) -> None:
        previous_offset = -1.0
        indices = set()
        for event in self.chart:
            if event.index in indices:
                raise ValueError("chart event indices must be unique")
            if not math.isfinite(event.target_offset) or event.target_offset < 0:
                raise ValueError("chart target offsets must be non-negative and finite")
            if event.target_offset < previous_offset:
                raise ValueError("chart events must be ordered by target time")
            indices.add(event.index)
            previous_offset = event.target_offset
        self._events_by_index = {event.index: event for event in self.chart}

    @property
    def song_start_time(self) -> float:
        """The absolute time when the countdown changes to GO."""
        return self.started_at + self.countdown_seconds

    @property
    def resolved_count(self) -> int:
        return len(self._resolved_indices)

    @property
    def remaining_count(self) -> int:
        return len(self.chart) - self.resolved_count

    def phase_at(self, now: float) -> RoundPhase:
        """Return the round phase at an absolute monotonic-clock time."""
        if self.resolved_count == len(self.chart):
            return RoundPhase.RESULTS
        if now < self.song_start_time:
            return RoundPhase.COUNTDOWN
        return RoundPhase.PLAYING

    def countdown_remaining(self, now: float) -> float:
        """Return countdown time left, stopping cleanly at zero."""
        return max(0.0, self.song_start_time - now)

    def progress_at(self, now: float) -> float:
        """Return smooth song progress from zero to one for the UI."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if not self.chart:
            return 1.0
        if self.phase_at(now) is RoundPhase.RESULTS:
            return 1.0
        challenge_duration = self.chart[-1].target_offset + GOOD_WINDOW_SECONDS
        if challenge_duration <= 0:
            return 1.0
        elapsed = now - self.song_start_time
        return max(0.0, min(1.0, elapsed / challenge_duration))

    def delay_timeline(self, pause_seconds: float) -> None:
        """Move all future beat times forward after an interface pause."""
        if not math.isfinite(pause_seconds) or pause_seconds < 0:
            raise ValueError("pause_seconds must be non-negative and finite")
        self.started_at += pause_seconds

    def _event(self, event_or_index: ChartEvent | int) -> ChartEvent:
        index = (
            event_or_index.index
            if isinstance(event_or_index, ChartEvent)
            else event_or_index
        )
        try:
            return self._events_by_index[index]
        except KeyError as error:
            raise KeyError(f"unknown chart event index: {index}") from error

    def target_time(self, event_or_index: ChartEvent | int) -> float:
        """Return when an event should ideally be touched."""
        return self.song_start_time + self._event(event_or_index).target_offset

    def spawn_time(self, event_or_index: ChartEvent | int) -> float:
        """Return when an event should enter at the top of the frame."""
        return self.target_time(event_or_index) - self.node_travel_seconds

    def due_events(self, now: float) -> tuple[ChartEvent, ...]:
        """Return every newly due event, including any skipped between frames."""
        due = []
        while self._next_spawn_position < len(self.chart):
            event = self.chart[self._next_spawn_position]
            if self.spawn_time(event) > now:
                break
            due.append(event)
            self._next_spawn_position += 1
        return tuple(due)

    def judge_hit(
        self,
        event_or_index: ChartEvent | int,
        hit_at: float,
        movement_rating: str | None = None,
    ) -> TimingGrade:
        """Resolve a touched node from its timing and update the round score.

        Every visible touch is a hit. Contacts outside the optional timing-bonus
        window receive the neutral HIT result, because the interface does not
        show a fixed timing target that the player is expected to follow.
        """
        event = self._event(event_or_index)
        if event.index in self._resolved_indices:
            raise ValueError(f"chart event {event.index} is already resolved")
        if not math.isfinite(hit_at):
            raise ValueError("hit_at must be finite")

        grade = (
            grade_timing(hit_at - self.target_time(event))
            or TimingGrade.HIT
        )
        self._resolved_indices.add(event.index)
        self.score.record_hit(grade, movement_rating)
        return grade

    def record_miss(self, event_or_index: ChartEvent | int) -> bool:
        """Resolve one unhit node as missed; return False if already resolved."""
        event = self._event(event_or_index)
        if event.index in self._resolved_indices:
            return False
        self._resolved_indices.add(event.index)
        self.score.record_miss()
        return True

    def is_resolved(self, event_or_index: ChartEvent | int) -> bool:
        """Return whether an event already became a hit or miss."""
        return self._event(event_or_index).index in self._resolved_indices

    def reset(self, started_at: float | None = None) -> None:
        """Clear the score and schedule, optionally starting from a new time."""
        if started_at is not None:
            if not math.isfinite(started_at):
                raise ValueError("started_at must be finite")
            self.started_at = started_at
        self.score = RoundScore()
        self._next_spawn_position = 0
        self._resolved_indices = set()
