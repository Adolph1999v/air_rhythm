"""Air Rhythm: catch falling circles to play a melody with your hands."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time

import cv2
import mediapipe as mp

from audio_engine import AudioEngine
from music import (
    ALL_PITCHES,
    INSTRUMENTS,
    MELODY_NOTES,
    MELODY_STEP_SECONDS,
    MELODY_TITLE,
    MelodyPlayer,
    note_name,
)
from privacy import PrivacyMode, PrivacyRenderer
from rhythm_game import (
    ChartEvent,
    GOOD_WINDOW_SECONDS,
    NODE_TRAVEL_SECONDS,
    RhythmRound,
    RoundPhase,
    TimingGrade,
    grade_timing,
)


CAMERA_INDEX = 0
WINDOW_TITLE = "Air Rhythm | Play with your hands"
MAX_HANDS = 2
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
NODE_SPAWN_INTERVAL_SECONDS = 0.55
NODE_FALL_SPEED_PER_FRAME_HEIGHT = 0.28
MAX_ACTIVE_NODES = 6
MAX_FRAME_TIME_SECONDS = 0.1
FINGERTIP_TOUCH_RADIUS = 10
HIT_EFFECT_DURATION_SECONDS = 0.35
MIN_NODE_RADIUS = 34
NODE_RADIUS_PER_FRAME_MIN_DIMENSION = 0.0585
CHALLENGE_TARGET_HEIGHT_RATIO = 0.68
DOWNWARD_STRIKE_SPEED = 0.55
FORWARD_STRIKE_SPEED = 0.4
GREAT_MOVEMENT_SPEED = 0.28
PERFECT_MOVEMENT_SPEED = 0.65
MOTION_SMOOTHING = 0.6
MAX_MOTION_SAMPLE_GAP_SECONDS = 0.2

# Each pair says which two hand landmarks should be joined by a line.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (5, 9),
    (9, 13),
    (13, 17),  # Palm
)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

LANDMARK_COLOR = (255, 210, 0)
CONNECTION_COLOR = (255, 150, 0)
FINGERTIP_COLOR = (0, 255, 0)

GAME_MODE_CHALLENGE = "challenge"
GAME_MODE_FREE_PLAY = "free_play"
INSTRUMENT_BY_KEY = {instrument.key: instrument for instrument in INSTRUMENTS}
RATING_VELOCITY = {"GOOD": 0.75, "GREAT": 0.9, "PERFECT": 1.0}
PRIVACY_LABELS = {
    PrivacyMode.HANDS_ONLY: "Hands only",
    PrivacyMode.SKELETON_ONLY: "Skeleton only",
    PrivacyMode.CAMERA: "Camera",
}


@dataclass
class FallingNode:
    """A circular rhythm target moving down the camera frame."""

    x: float
    y: float
    radius: int
    speed: float
    color: tuple[int, int, int]
    instrument: str = "keys"
    chart_index: int | None = None
    midi_note: int | None = None
    target_time: float | None = None
    spawn_time: float | None = None


@dataclass
class HitEffect:
    """A short visual burst left behind when a node is touched."""

    position: tuple[int, int]
    color: tuple[int, int, int]
    started_at: float
    rating: str
    detail: str | None = None


@dataclass
class NodeHit:
    """A successful node collision and the quality of its movement."""

    node: FallingNode
    rating: str
    timing_grade: TimingGrade | None = None
    timing_error: float | None = None


@dataclass
class FingertipHistory:
    """The previous position and smoothed speed of one fingertip."""

    x: float
    y: float
    z: float
    pixel_position: tuple[int, int]
    measured_at: float
    movement_speed: float = 0.0
    downward_speed: float = 0.0
    forward_speed: float = 0.0


@dataclass
class FingertipMotion:
    """A fingertip's current screen position and detected strike direction."""

    previous_position: tuple[int, int]
    position: tuple[int, int]
    movement_speed: float
    strike_direction: str | None


def create_hand_landmarker() -> mp.tasks.vision.HandLandmarker:
    """Create the locally stored MediaPipe model in video-tracking mode."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the hand model at {MODEL_PATH}. "
            "Download hand_landmarker.task into the models folder."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def landmark_to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Convert one normalized MediaPipe landmark to a safe pixel position."""
    pixel_x = max(0, min(int(landmark.x * frame_width), frame_width - 1))
    pixel_y = max(0, min(int(landmark.y * frame_height), frame_height - 1))
    return pixel_x, pixel_y


def draw_hand_skeleton(frame, hand_landmarks) -> None:
    """Draw one hand's landmark points, connecting lines, and fingertips."""
    frame_height, frame_width = frame.shape[:2]
    pixel_points = [
        landmark_to_pixel(landmark, frame_width, frame_height)
        for landmark in hand_landmarks
    ]

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pixel_points[start_index],
            pixel_points[end_index],
            CONNECTION_COLOR,
            2,
            cv2.LINE_AA,
        )

    for landmark_index, point in enumerate(pixel_points):
        is_fingertip = landmark_index in FINGERTIP_INDICES
        color = FINGERTIP_COLOR if is_fingertip else LANDMARK_COLOR
        radius = 7 if is_fingertip else 4
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def collect_fingertip_motions(
    detection_result,
    frame_width: int,
    frame_height: int,
    current_time: float,
    previous_history: dict[tuple[str, int], FingertipHistory],
) -> tuple[list[FingertipMotion], dict[tuple[str, int], FingertipHistory]]:
    """Measure downward and forward movement for every detected fingertip."""
    fingertip_motions = []
    updated_history = {}

    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_identity = f"hand-{hand_index}"
        if hand_index < len(detection_result.handedness):
            handedness = detection_result.handedness[hand_index]
            if handedness and handedness[0].category_name:
                hand_identity = handedness[0].category_name

        for fingertip_index in FINGERTIP_INDICES:
            landmark = hand_landmarks[fingertip_index]
            history_key = (hand_identity, fingertip_index)
            previous = previous_history.get(history_key)
            current_position = landmark_to_pixel(
                landmark,
                frame_width,
                frame_height,
            )
            previous_position = current_position
            movement_speed = 0.0
            downward_speed = 0.0
            forward_speed = 0.0

            if previous is not None:
                elapsed_seconds = current_time - previous.measured_at
                if 0 < elapsed_seconds <= MAX_MOTION_SAMPLE_GAP_SECONDS:
                    previous_position = previous.pixel_position
                    movement_x = landmark.x - previous.x
                    movement_y = landmark.y - previous.y
                    raw_movement_speed = (
                        movement_x**2 + movement_y**2
                    ) ** 0.5 / elapsed_seconds
                    raw_downward_speed = (landmark.y - previous.y) / elapsed_seconds
                    raw_forward_speed = (previous.z - landmark.z) / elapsed_seconds
                    movement_speed = (
                        MOTION_SMOOTHING * raw_movement_speed
                        + (1 - MOTION_SMOOTHING) * previous.movement_speed
                    )
                    downward_speed = (
                        MOTION_SMOOTHING * raw_downward_speed
                        + (1 - MOTION_SMOOTHING) * previous.downward_speed
                    )
                    forward_speed = (
                        MOTION_SMOOTHING * raw_forward_speed
                        + (1 - MOTION_SMOOTHING) * previous.forward_speed
                    )

            strike_directions = []
            if downward_speed >= DOWNWARD_STRIKE_SPEED:
                strike_directions.append("DOWN")
            if forward_speed >= FORWARD_STRIKE_SPEED:
                strike_directions.append("FORWARD")
            strike_direction = " + ".join(strike_directions) or None

            fingertip_motions.append(
                FingertipMotion(
                    previous_position=previous_position,
                    position=current_position,
                    movement_speed=movement_speed,
                    strike_direction=strike_direction,
                )
            )
            updated_history[history_key] = FingertipHistory(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                pixel_position=current_position,
                measured_at=current_time,
                movement_speed=movement_speed,
                downward_speed=downward_speed,
                forward_speed=forward_speed,
            )

    return fingertip_motions, updated_history


def point_overlaps_node(
    point: tuple[int, int],
    node_center: tuple[int, int],
    collision_radius: int,
) -> bool:
    """Return True when a fingertip point touches a circular node."""
    difference_x = point[0] - node_center[0]
    difference_y = point[1] - node_center[1]
    distance_squared = difference_x**2 + difference_y**2
    return distance_squared <= collision_radius**2


def movement_path_overlaps_node(
    motion: FingertipMotion,
    node_center: tuple[int, int],
    collision_radius: int,
) -> bool:
    """Check the full fingertip path between frames against a node."""
    start_x, start_y = motion.previous_position
    end_x, end_y = motion.position
    path_x = end_x - start_x
    path_y = end_y - start_y
    path_length_squared = path_x**2 + path_y**2

    if path_length_squared == 0:
        return point_overlaps_node(
            motion.position,
            node_center,
            collision_radius,
        )

    node_from_start_x = node_center[0] - start_x
    node_from_start_y = node_center[1] - start_y
    projection = (
        node_from_start_x * path_x + node_from_start_y * path_y
    ) / path_length_squared
    projection = max(0.0, min(1.0, projection))
    closest_point = (
        start_x + projection * path_x,
        start_y + projection * path_y,
    )

    return point_overlaps_node(
        closest_point,
        node_center,
        collision_radius,
    )


def rate_fingertip_motion(motion: FingertipMotion) -> str:
    """Turn movement strength into feedback without rejecting the touch."""
    if (
        motion.strike_direction is not None
        or motion.movement_speed >= PERFECT_MOVEMENT_SPEED
    ):
        return "PERFECT"
    if motion.movement_speed >= GREAT_MOVEMENT_SPEED:
        return "GREAT"
    return "GOOD"


def node_radius_for_frame(frame_width: int, frame_height: int) -> int:
    """Keep circles large enough to touch at every common camera resolution."""
    return max(
        MIN_NODE_RADIUS,
        int(min(frame_width, frame_height) * NODE_RADIUS_PER_FRAME_MIN_DIMENSION),
    )


def choose_node_x(
    frame_width: int,
    radius: int,
    active_nodes: list[FallingNode],
) -> int:
    """Choose a full-frame horizontal position without creating a tight cluster."""
    minimum_x = radius
    maximum_x = max(radius, frame_width - radius)
    candidate_x = random.randint(minimum_x, maximum_x)

    for _ in range(12):
        if all(abs(candidate_x - node.x) >= radius * 3 for node in active_nodes):
            break
        candidate_x = random.randint(minimum_x, maximum_x)
    return candidate_x


def create_falling_node(
    frame_width: int,
    frame_height: int,
    active_nodes: list[FallingNode],
) -> FallingNode:
    """Spawn one free-play node at the top of the camera frame."""
    radius = node_radius_for_frame(frame_width, frame_height)
    nodes_near_top = [node for node in active_nodes if node.y < radius * 4]
    candidate_x = choose_node_x(frame_width, radius, nodes_near_top)

    instrument = random.choice(INSTRUMENTS)
    return FallingNode(
        x=float(candidate_x),
        y=float(radius),
        radius=radius,
        speed=frame_height * NODE_FALL_SPEED_PER_FRAME_HEIGHT,
        color=instrument.color,
        instrument=instrument.key,
    )


def create_challenge_node(
    event: ChartEvent,
    rhythm_round: RhythmRound,
    frame_width: int,
    frame_height: int,
    current_time: float,
    active_nodes: list[FallingNode],
) -> FallingNode:
    """Create a song node whose position follows the absolute music clock."""
    radius = node_radius_for_frame(frame_width, frame_height)
    nodes_near_top = [node for node in active_nodes if node.y < radius * 5]
    earlier_song_nodes = [
        node for node in active_nodes if node.chart_index is not None
    ]
    if earlier_song_nodes:
        previous_node = max(earlier_song_nodes, key=lambda node: node.chart_index)
        reachable_distance = int(frame_width * 0.20)
        minimum_x = max(radius, round(previous_node.x) - reachable_distance)
        maximum_x = min(frame_width - radius, round(previous_node.x) + reachable_distance)
        candidate_x = random.randint(minimum_x, max(minimum_x, maximum_x))
        for _ in range(12):
            if all(abs(candidate_x - node.x) >= radius * 2.5 for node in nodes_near_top):
                break
            candidate_x = random.randint(minimum_x, max(minimum_x, maximum_x))
    else:
        candidate_x = choose_node_x(frame_width, radius, nodes_near_top)
    spawn_time = rhythm_round.spawn_time(event)
    target_time = rhythm_round.target_time(event)
    target_y = frame_height * CHALLENGE_TARGET_HEIGHT_RATIO
    speed = max(1.0, (target_y - radius) / rhythm_round.node_travel_seconds)
    elapsed_since_spawn = max(0.0, current_time - spawn_time)
    instrument = INSTRUMENT_BY_KEY[event.instrument]

    return FallingNode(
        x=float(candidate_x),
        y=float(radius + speed * elapsed_since_spawn),
        radius=radius,
        speed=speed,
        color=instrument.color,
        instrument=event.instrument,
        chart_index=event.index,
        midi_note=event.pitch,
        target_time=target_time,
        spawn_time=spawn_time,
    )


def update_falling_nodes(
    active_nodes: list[FallingNode],
    elapsed_seconds: float,
    fingertip_motions: list[FingertipMotion],
    frame_height: int,
) -> tuple[list[FallingNode], list[NodeHit], int]:
    """Move nodes and separate them into active, hit, and missed groups."""
    remaining_nodes = []
    node_hits = []
    missed_node_count = 0
    rating_priority = {"GOOD": 1, "GREAT": 2, "PERFECT": 3}

    for node in active_nodes:
        node.y += node.speed * elapsed_seconds
        node_center = (round(node.x), round(node.y))
        collision_radius = node.radius + FINGERTIP_TOUCH_RADIUS

        touching_motions = [
            motion
            for motion in fingertip_motions
            if movement_path_overlaps_node(
                motion,
                node_center,
                collision_radius,
            )
        ]
        if touching_motions:
            rating = max(
                (rate_fingertip_motion(motion) for motion in touching_motions),
                key=rating_priority.get,
            )
            node_hits.append(NodeHit(node=node, rating=rating))
        elif node.y - node.radius > frame_height:
            missed_node_count += 1
        else:
            remaining_nodes.append(node)

    return remaining_nodes, node_hits, missed_node_count


def update_challenge_nodes(
    active_nodes: list[FallingNode],
    current_time: float,
    fingertip_motions: list[FingertipMotion],
    rhythm_round: RhythmRound,
) -> tuple[list[FallingNode], list[NodeHit], int]:
    """Position song nodes from the beat clock and judge fingertip contacts."""
    remaining_nodes = []
    node_hits = []
    missed_node_count = 0
    rating_priority = {"GOOD": 1, "GREAT": 2, "PERFECT": 3}

    for node in active_nodes:
        if node.chart_index is None or node.target_time is None or node.spawn_time is None:
            raise ValueError("A challenge node is missing its chart timing.")

        # Absolute time prevents the music and circles from drifting if one
        # camera frame is slow. Both jump to the correct point on the same clock.
        node.y = node.radius + node.speed * max(0.0, current_time - node.spawn_time)
        timing_error = current_time - node.target_time
        timing_grade = grade_timing(timing_error)
        collision_radius = node.radius + FINGERTIP_TOUCH_RADIUS
        node_center = (round(node.x), round(node.y))

        touching_motions = []
        if timing_grade is not None:
            touching_motions = [
                motion
                for motion in fingertip_motions
                if movement_path_overlaps_node(
                    motion,
                    node_center,
                    collision_radius,
                )
            ]

        if touching_motions:
            movement_rating = max(
                (rate_fingertip_motion(motion) for motion in touching_motions),
                key=rating_priority.get,
            )
            judged_grade = rhythm_round.judge_hit(
                node.chart_index,
                current_time,
                movement_rating,
            )
            node_hits.append(
                NodeHit(
                    node=node,
                    rating=movement_rating,
                    timing_grade=judged_grade,
                    timing_error=timing_error,
                )
            )
        elif timing_error > GOOD_WINDOW_SECONDS:
            if rhythm_round.record_miss(node.chart_index):
                missed_node_count += 1
        else:
            remaining_nodes.append(node)

    return remaining_nodes, node_hits, missed_node_count


def play_node_hits(
    node_hits: list[NodeHit],
    audio: AudioEngine,
    melody: MelodyPlayer,
    melody_mode: bool,
) -> None:
    """Play chart notes in challenge mode or color notes in free play."""
    if not node_hits:
        return

    # Older hit-paced nodes have no attached pitch. Keep their shared melody
    # step as a safe fallback while scheduled song nodes carry their own pitch.
    fallback_melody_note = None
    if melody_mode and any(hit.node.midi_note is None for hit in node_hits):
        fallback_melody_note = melody.advance()

    requested_notes: dict[tuple[str, int], float] = {}
    for hit in node_hits:
        instrument_key = hit.node.instrument
        instrument = INSTRUMENT_BY_KEY[instrument_key]
        pitch = (
            hit.node.midi_note
            if melody_mode and hit.node.midi_note is not None
            else fallback_melody_note
            if melody_mode
            else instrument.freestyle_note
        )
        note_key = (instrument_key, pitch)
        requested_notes[note_key] = max(
            requested_notes.get(note_key, 0.0),
            RATING_VELOCITY[hit.rating],
        )

    for (instrument_key, pitch), velocity in requested_notes.items():
        audio.play_note(pitch, instrument=instrument_key, velocity=velocity)


def draw_falling_nodes(
    frame,
    active_nodes: list[FallingNode],
    current_time: float | None = None,
) -> None:
    """Draw falling circles and a halo that closes on each song beat."""
    transparent_overlay = frame.copy()

    for node in active_nodes:
        center = (round(node.x), round(node.y))
        cv2.circle(
            transparent_overlay,
            center,
            node.radius,
            node.color,
            -1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(transparent_overlay, 0.32, frame, 0.68, 0, frame)

    for node in active_nodes:
        center = (round(node.x), round(node.y))
        if current_time is not None and node.target_time is not None:
            seconds_until_target = node.target_time - current_time
            approaching_progress = max(
                0.0,
                min(1.0, seconds_until_target / NODE_TRAVEL_SECONDS),
            )
            if abs(seconds_until_target) <= 0.10:
                halo_color = (80, 255, 110)
                halo_thickness = 5
            elif seconds_until_target < 0:
                halo_color = (80, 190, 255)
                halo_thickness = 3
            else:
                halo_color = (245, 245, 245)
                halo_thickness = 3
            halo_radius = node.radius + 5 + int(node.radius * 1.45 * approaching_progress)
            cv2.circle(
                frame,
                center,
                halo_radius,
                halo_color,
                halo_thickness,
                cv2.LINE_AA,
            )
        cv2.circle(frame, center, node.radius, node.color, 4, cv2.LINE_AA)
        label = (
            note_name(node.midi_note)
            if node.midi_note is not None
            else INSTRUMENT_BY_KEY[node.instrument].label
        )
        label_scale = min(0.48, node.radius / 90)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1
        )
        text_position = (center[0] - text_width // 2, center[1] + text_height // 2)
        for color, thickness in (((20, 20, 20), 3), ((255, 255, 255), 1)):
            cv2.putText(
                frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                label_scale, color, thickness, cv2.LINE_AA,
            )


def draw_hit_effects(
    frame,
    hit_effects: list[HitEffect],
    current_time: float,
) -> None:
    """Draw a briefly expanding ring with timing or movement feedback."""
    rating_colors = {
        "GOOD": (255, 255, 255),
        "GREAT": (40, 210, 255),
        "PERFECT": (80, 255, 80),
        "TOUCH": (255, 255, 255),
        "STRONG": (40, 210, 255),
        "POWER": (80, 255, 80),
    }

    for effect in hit_effects:
        progress = (current_time - effect.started_at) / HIT_EFFECT_DURATION_SECONDS
        ring_radius = int(25 + progress * 45)
        cv2.circle(
            frame,
            effect.position,
            ring_radius,
            effect.color,
            4,
            cv2.LINE_AA,
        )
        text_size, _ = cv2.getTextSize(
            effect.rating,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        text_position = (
            effect.position[0] - text_size[0] // 2,
            max(28, effect.position[1] - ring_radius - 10),
        )
        cv2.putText(
            frame,
            effect.rating,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            5,
            cv2.LINE_AA,
        )
        if effect.detail:
            detail_size, _ = cv2.getTextSize(
                effect.detail,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                1,
            )
            detail_position = (
                effect.position[0] - detail_size[0] // 2,
                min(frame.shape[0] - 15, effect.position[1] + ring_radius + 24),
            )
            cv2.putText(
                frame,
                effect.detail,
                detail_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                effect.detail,
                detail_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            effect.rating,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            rating_colors[effect.rating],
            2,
            cv2.LINE_AA,
        )


def draw_strike_feedback(
    frame,
    fingertip_motions: list[FingertipMotion],
) -> None:
    """Highlight fingertips only while a deliberate strike is detected."""
    for motion in fingertip_motions:
        if motion.strike_direction is None:
            continue

        if motion.strike_direction == "DOWN":
            strike_color = (0, 165, 255)
        elif motion.strike_direction == "FORWARD":
            strike_color = (255, 80, 220)
        else:
            strike_color = (255, 255, 255)

        cv2.circle(
            frame,
            motion.position,
            15,
            strike_color,
            3,
            cv2.LINE_AA,
        )


def draw_outlined_text(
    frame,
    text: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw readable text on either a camera image or the privacy stage."""
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (15, 15, 18), thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


def draw_status_overlay(
    frame,
    hand_count: int,
    hit_count: int,
    miss_count: int,
    active_strikes: set[str],
    audio: AudioEngine,
    privacy_mode: PrivacyMode,
    game_mode: str,
    rhythm_round: RhythmRound,
    current_time: float,
) -> None:
    """Show compact timing, score, privacy, and control information."""
    if game_mode == GAME_MODE_CHALLENGE:
        score = rhythm_round.score
        status_text = (
            f"Hands: {hand_count}/{MAX_HANDS}   "
            f"Score: {score.score:,}   Combo: {score.combo}"
        )
        phase = rhythm_round.phase_at(current_time)
        if phase is RoundPhase.COUNTDOWN:
            instruction_text = "Get ready - the halo meets the circle on the beat"
        elif phase is RoundPhase.RESULTS:
            instruction_text = "Round complete - press R to replay"
        elif active_strikes:
            instruction_text = f"Timing + motion bonus: {', '.join(sorted(active_strikes))}"
        else:
            instruction_text = "Touch when the shrinking halo reaches the circle"
        mode_status = (
            f"{MELODY_TITLE} Challenge | Notes left: {rhythm_round.remaining_count}"
        )
    else:
        status_text = (
            f"Hands: {hand_count}/{MAX_HANDS}   Hits: {hit_count}   Misses: {miss_count}"
        )
        instruction_text = (
            f"Motion bonus: {', '.join(sorted(active_strikes))}"
            if active_strikes
            else "Touch or sweep through a falling circle"
        )
        mode_status = "Free play | Each color is an instrument"

    instruction_color = (80, 255, 110) if active_strikes else (225, 225, 225)
    draw_outlined_text(frame, status_text, (22, 39), 0.65)
    draw_outlined_text(frame, instruction_text, (22, 68), 0.52, instruction_color)

    if not audio.enabled:
        audio_status = "Sound unavailable"
    elif audio.muted:
        audio_status = "Muted"
    else:
        audio_status = "Sound on"
    privacy_status = PRIVACY_LABELS[privacy_mode]
    footer_lines = (
        f"{mode_status} | {audio_status} | Privacy: {privacy_status}",
        "1 Challenge   2 Free play   P Privacy   M Mute   R Restart   Q Quit",
    )
    frame_height, frame_width = frame.shape[:2]
    for index, line in enumerate(footer_lines):
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
        scale = 0.5 * min(1.0, max(1, frame_width - 44) / max(1, text_width))
        draw_outlined_text(
            frame,
            line,
            (22, frame_height - 45 + index * 25),
            scale,
            thickness=2,
        )


def draw_countdown(frame, rhythm_round: RhythmRound, current_time: float) -> None:
    """Show 3, 2, 1 and GO without hiding the incoming first circle."""
    remaining = rhythm_round.countdown_remaining(current_time)
    if remaining > 0:
        message = str(max(1, math.ceil(remaining)))
    elif current_time - rhythm_round.song_start_time < 0.5:
        message = "GO!"
    else:
        return

    scale = 3.0 if message != "GO!" else 2.2
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 7)[0]
    position = (
        frame.shape[1] // 2 - text_size[0] // 2,
        frame.shape[0] // 2 + text_size[1] // 2,
    )
    draw_outlined_text(frame, message, position, scale, (80, 255, 110), 7)


def draw_results_screen(frame, rhythm_round: RhythmRound) -> None:
    """Give a demo round a clear ending with useful performance feedback."""
    score = rhythm_round.score
    height, width = frame.shape[:2]
    panel_left, panel_right = int(width * 0.19), int(width * 0.81)
    panel_top, panel_bottom = int(height * 0.17), int(height * 0.78)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (12, 14, 23),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (80, 220, 255),
        3,
        cv2.LINE_AA,
    )

    def centered(text: str, y: int, scale: float, color=(255, 255, 255), thickness=2):
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0]
        draw_outlined_text(
            frame,
            text,
            (width // 2 - text_width // 2, y),
            scale,
            color,
            thickness,
        )

    centered("ROUND COMPLETE", panel_top + 45, 0.85, (80, 220, 255), 2)
    centered(f"RANK {score.rank}", panel_top + 125, 2.0, (80, 255, 110), 5)
    centered(f"Score {score.score:,}   Accuracy {score.accuracy:.1f}%", panel_top + 172, 0.62)
    centered(
        f"Perfect {score.perfect}   Great {score.great}   "
        f"Good {score.good}   Miss {score.misses}",
        panel_top + 211,
        0.48,
    )
    centered(f"Best combo {score.max_combo}", panel_top + 246, 0.52)
    centered("Press R to replay", panel_bottom - 25, 0.55, (230, 230, 230))


def run_sound_test() -> bool:
    """Play the opening melody without opening the camera or hand model."""
    audio = AudioEngine(ALL_PITCHES)
    try:
        if not audio.start():
            print(f"Could not start sound: {audio.error_message}")
            return False
        print(f"Playing {MELODY_TITLE}. Press Ctrl+C to stop.")
        for pitch, step_seconds in zip(MELODY_NOTES, MELODY_STEP_SECONDS):
            if not audio.enabled:
                print(audio.error_message or "Sound output stopped.")
                return False
            audio.play_note(pitch, instrument="keys")
            time.sleep(step_seconds)
        time.sleep(0.8)
        if not audio.enabled:
            print(audio.error_message or "Sound output stopped.")
        return audio.enabled
    except KeyboardInterrupt:
        return True
    finally:
        audio.close()


def main() -> None:
    """Run the timed song challenge, free play, and privacy views."""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open the camera. Check that it is connected and that "
            "your terminal or editor has Camera permission in macOS Settings."
        )

    audio = AudioEngine(ALL_PITCHES)
    try:
        if not audio.start():
            print(audio.error_message or "Sound unavailable.")
            print("Air Rhythm will keep running. Check your sound output and restart.")
        reported_audio_error = audio.error_message
        with create_hand_landmarker() as hand_landmarker:
            previous_timestamp_ms = -1
            previous_frame_time = time.monotonic()
            last_spawn_time = previous_frame_time - NODE_SPAWN_INTERVAL_SECONDS
            active_nodes: list[FallingNode] = []
            hit_effects: list[HitEffect] = []
            fingertip_history: dict[tuple[str, int], FingertipHistory] = {}
            hit_count = 0
            miss_count = 0
            melody = MelodyPlayer()
            game_mode = GAME_MODE_CHALLENGE
            rhythm_round = RhythmRound(started_at=previous_frame_time)
            privacy_renderer = PrivacyRenderer(PrivacyMode.CAMERA)

            while True:
                frame_read_successfully, frame = camera.read()

                if not frame_read_successfully:
                    print("Could not read a camera frame. Closing Air Rhythm.")
                    break

                mirrored_frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = max(
                    int(time.monotonic() * 1000),
                    previous_timestamp_ms + 1,
                )
                previous_timestamp_ms = timestamp_ms
                detection_result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                frame_height, frame_width = mirrored_frame.shape[:2]
                current_time = time.monotonic()
                fingertip_motions, fingertip_history = collect_fingertip_motions(
                    detection_result,
                    frame_width,
                    frame_height,
                    current_time,
                    fingertip_history,
                )
                active_strikes = {
                    motion.strike_direction
                    for motion in fingertip_motions
                    if motion.strike_direction is not None
                }

                elapsed_seconds = min(
                    current_time - previous_frame_time,
                    MAX_FRAME_TIME_SECONDS,
                )
                previous_frame_time = current_time

                if game_mode == GAME_MODE_CHALLENGE:
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
                    active_nodes, node_hits, new_misses = update_challenge_nodes(
                        active_nodes,
                        current_time,
                        fingertip_motions,
                        rhythm_round,
                    )
                    hit_count = (
                        rhythm_round.score.perfect
                        + rhythm_round.score.great
                        + rhythm_round.score.good
                    )
                    miss_count = rhythm_round.score.misses
                else:
                    if (
                        current_time - last_spawn_time >= NODE_SPAWN_INTERVAL_SECONDS
                        and len(active_nodes) < MAX_ACTIVE_NODES
                    ):
                        active_nodes.append(
                            create_falling_node(
                                frame_width,
                                frame_height,
                                active_nodes,
                            )
                        )
                        last_spawn_time = current_time

                    active_nodes, node_hits, new_misses = update_falling_nodes(
                        active_nodes,
                        elapsed_seconds,
                        fingertip_motions,
                        frame_height,
                    )
                    hit_count += len(node_hits)
                    miss_count += new_misses

                play_node_hits(
                    node_hits,
                    audio,
                    melody,
                    game_mode == GAME_MODE_CHALLENGE,
                )
                if not audio.enabled and audio.error_message != reported_audio_error:
                    print(audio.error_message)
                    print("Check your sound output and restart Air Rhythm to reconnect.")
                    reported_audio_error = audio.error_message

                for node_hit in node_hits:
                    if node_hit.timing_grade is None:
                        effect_rating = {
                            "GOOD": "TOUCH",
                            "GREAT": "STRONG",
                            "PERFECT": "POWER",
                        }[node_hit.rating]
                        effect_detail = None
                    else:
                        effect_rating = node_hit.timing_grade.label
                        timing_milliseconds = round(abs(node_hit.timing_error or 0.0) * 1000)
                        if timing_milliseconds <= 15:
                            timing_description = "ON BEAT"
                        elif (node_hit.timing_error or 0.0) < 0:
                            timing_description = f"{timing_milliseconds} ms EARLY"
                        else:
                            timing_description = f"{timing_milliseconds} ms LATE"
                        motion_description = {
                            "GOOD": "TOUCH",
                            "GREAT": "STRONG",
                            "PERFECT": "POWER",
                        }[node_hit.rating]
                        effect_detail = f"{timing_description} | {motion_description}"

                    hit_effects.append(
                        HitEffect(
                            position=(
                                round(node_hit.node.x),
                                round(node_hit.node.y),
                            ),
                            color=node_hit.node.color,
                            started_at=current_time,
                            rating=effect_rating,
                            detail=effect_detail,
                        )
                    )
                hit_effects = [
                    effect
                    for effect in hit_effects
                    if current_time - effect.started_at < HIT_EFFECT_DURATION_SECONDS
                ]

                display_frame = privacy_renderer.apply(
                    mirrored_frame,
                    detection_result.hand_landmarks,
                )
                draw_falling_nodes(
                    display_frame,
                    active_nodes,
                    current_time if game_mode == GAME_MODE_CHALLENGE else None,
                )
                draw_hit_effects(display_frame, hit_effects, current_time)
                draw_status_overlay(
                    display_frame,
                    len(detection_result.hand_landmarks),
                    hit_count,
                    miss_count,
                    active_strikes,
                    audio,
                    privacy_renderer.mode,
                    game_mode,
                    rhythm_round,
                    current_time,
                )

                if game_mode == GAME_MODE_CHALLENGE:
                    if rhythm_round.phase_at(current_time) is RoundPhase.RESULTS:
                        draw_results_screen(display_frame, rhythm_round)
                    else:
                        draw_countdown(display_frame, rhythm_round, current_time)

                # Draw tracking last. Its 21 landmark points and connections
                # remain obvious in every privacy setting and on the results view.
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_skeleton(display_frame, hand_landmarks)
                draw_strike_feedback(display_frame, fingertip_motions)
                cv2.imshow(WINDOW_TITLE, display_frame)

                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord("q"):
                    break
                if pressed_key == ord("p"):
                    privacy_renderer.cycle()
                if pressed_key == ord("m"):
                    audio.set_muted(not audio.muted)
                if pressed_key in (ord("1"), ord("2"), ord("r")):
                    if pressed_key == ord("1"):
                        game_mode = GAME_MODE_CHALLENGE
                    elif pressed_key == ord("2"):
                        game_mode = GAME_MODE_FREE_PLAY

                    rhythm_round.reset(started_at=current_time)
                    melody.reset()
                    audio.stop_all()
                    active_nodes.clear()
                    hit_effects.clear()
                    fingertip_history.clear()
                    hit_count = 0
                    miss_count = 0
                    previous_frame_time = current_time
                    last_spawn_time = current_time - NODE_SPAWN_INTERVAL_SECONDS
    finally:
        camera.release()
        audio.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sound-test", action="store_true",
        help="play a short melody through your speakers without opening the camera",
    )
    arguments = parser.parse_args()
    if arguments.sound_test:
        raise SystemExit(0 if run_sound_test() else 1)
    main()
