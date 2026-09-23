"""Air Rhythm: catch falling circles to play a melody with your hands."""

import argparse
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import random
import time

import cv2
import mediapipe as mp
import numpy as np

from audio_engine import AudioEngine
from camera_capture import LatestFrameCamera
from hand_stabilizer import HandLandmarkStabilizer
from performance_benchmark import PerformanceBenchmark, save_benchmark_report
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
from performance_stage import (
    create_performance_stage,
    draw_camera_inset,
    draw_collision_points,
    draw_virtual_drumsticks,
)
from rhythm_game import (
    ChartEvent,
    RhythmRound,
    RoundPhase,
    TimingGrade,
)
import ui


CAMERA_INDEX = 0
CAMERA_REQUEST_WIDTH = 1280
CAMERA_REQUEST_HEIGHT = 720
CAMERA_REQUEST_FPS = 30.0
CAMERA_PROCESSING_MAX_WIDTH = 1280
CAMERA_PROCESSING_MAX_HEIGHT = 720
WINDOW_TITLE = "Air Rhythm | Play with your hands"
MAX_HANDS = 2
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
BENCHMARK_REPORT_DIRECTORY = (
    Path(__file__).resolve().parent / "benchmark_reports"
)
NODE_SPAWN_INTERVAL_SECONDS = 0.55
NODE_FALL_SPEED_PER_FRAME_HEIGHT = 0.28
MAX_ACTIVE_NODES = 6
MAX_FRAME_TIME_SECONDS = 0.1
FINGERTIP_TOUCH_RADIUS = 10
HIT_EFFECT_DURATION_SECONDS = 0.35
MIN_NODE_RADIUS = 34
NODE_RADIUS_PER_FRAME_MIN_DIMENSION = 0.0585
# The main stage uses the native shape of modern Mac displays.  This prevents
# the OpenCV window from placing a 16:9 camera image inside a much taller
# maximized client area.
STAGE_ASPECT_RATIO = 16 / 10
# A target becomes playable in the upper third of the stage, leaving the rest
# of the screen free for the falling-circle motion and virtual drumsticks.
CHALLENGE_TARGET_HEIGHT_RATIO = 0.30
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


class AppScreen(Enum):
    """The presentation states around the live camera pipeline."""

    TITLE = "TITLE"
    GAMEPLAY = "GAMEPLAY"
    RESULTS = "RESULTS"


def audio_status_label(audio: AudioEngine) -> str:
    """Return a short, truthful speaker status for the interface."""
    if not audio.enabled:
        return "Sound unavailable"
    if audio.muted:
        return "Muted"
    return "Sound on"


def handedness_confidences(detection_result) -> tuple[float, ...]:
    """Read MediaPipe's hand-classification confidence values when present.

    These scores describe the Left/Right handedness classification.  They are
    useful model telemetry, but they are not a general tracking-accuracy score.
    """
    scores = []
    for classifications in getattr(detection_result, "handedness", ()):
        if not classifications:
            continue
        score = getattr(classifications[0], "score", None)
        if isinstance(score, (int, float)) and math.isfinite(score):
            scores.append(max(0.0, min(1.0, float(score))))
    return tuple(scores)


def finish_benchmark_session(
    benchmark: PerformanceBenchmark,
) -> tuple[dict, tuple[Path, Path]]:
    """Stop an active benchmark, save both reports, and announce their paths."""
    report = benchmark.stop()
    paths = save_benchmark_report(report, BENCHMARK_REPORT_DIRECTORY)
    print(f"Benchmark JSON saved to: {paths[0]}")
    print(f"Benchmark Markdown saved to: {paths[1]}")
    return report, paths


def configure_camera_capture(camera) -> dict[str, float | None]:
    """Request a responsive camera mode and report what the backend accepted.

    Camera drivers may ignore OpenCV property requests, so the returned values
    are observations rather than promises. The frame is also size-limited
    later in the pipeline to guarantee bounded computer-vision work.
    """
    requests = (
        (cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_REQUEST_WIDTH)),
        (cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_REQUEST_HEIGHT)),
        (cv2.CAP_PROP_FPS, CAMERA_REQUEST_FPS),
    )
    for property_id, requested_value in requests:
        try:
            camera.set(property_id, requested_value)
        except (AttributeError, cv2.error):
            pass

    buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_property is not None:
        try:
            camera.set(buffer_property, 1)
        except (AttributeError, cv2.error):
            pass

    def reported_value(property_id: int) -> float | None:
        try:
            value = float(camera.get(property_id))
        except (AttributeError, TypeError, ValueError, cv2.error):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    configuration = {
        "width": reported_value(cv2.CAP_PROP_FRAME_WIDTH),
        "height": reported_value(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": reported_value(cv2.CAP_PROP_FPS),
    }
    reported_width = configuration["width"]
    reported_height = configuration["height"]
    reported_fps = configuration["fps"]
    reported_resolution = (
        f"{int(round(reported_width))}x{int(round(reported_height))}"
        if reported_width is not None and reported_height is not None
        else "unavailable"
    )
    fps_label = f"{reported_fps:.2f}" if reported_fps is not None else "unavailable"
    print(
        "Camera request: "
        f"{CAMERA_REQUEST_WIDTH}x{CAMERA_REQUEST_HEIGHT} at "
        f"{CAMERA_REQUEST_FPS:.0f} FPS; backend reports "
        f"{reported_resolution} at {fps_label} FPS."
    )
    return configuration


def resize_frame_for_processing(
    frame: np.ndarray,
    max_width: int = CAMERA_PROCESSING_MAX_WIDTH,
    max_height: int = CAMERA_PROCESSING_MAX_HEIGHT,
) -> np.ndarray:
    """Downscale a camera frame for CV work without changing its aspect ratio."""
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("camera frame must be an image array")
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("camera frame must have positive dimensions")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("processing limits must be positive")

    resize_scale = min(
        1.0,
        max_width / frame_width,
        max_height / frame_height,
    )
    if resize_scale >= 1.0:
        return frame

    resized_width = max(1, int(round(frame_width * resize_scale)))
    resized_height = max(1, int(round(frame_height * resize_scale)))
    return cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


@dataclass
class FallingNode:
    """A circular rhythm target moving down the generated stage."""

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
    in_order: bool | None = None


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
        hand_identity = getattr(
            hand_landmarks,
            "identity",
            f"hand-{hand_index}",
        )
        if (
            not getattr(hand_landmarks, "identity", None)
            and hand_index < len(detection_result.handedness)
        ):
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
    """Keep circles large enough to touch at every common stage resolution."""
    return max(
        MIN_NODE_RADIUS,
        int(min(frame_width, frame_height) * NODE_RADIUS_PER_FRAME_MIN_DIMENSION),
    )


def stage_dimensions_for_camera(
    camera_width: int,
    camera_height: int,
) -> tuple[int, int]:
    """Return a 16:10 stage that fully contains the live camera resolution.

    The performance view is generated independently from the camera preview.
    Rendering it in the same 16:10 shape as a maximized Mac client area avoids
    the large gray letterbox that a 16:9 camera frame would otherwise create.
    """
    if camera_width < 1 or camera_height < 1:
        raise ValueError("camera dimensions must be positive")

    stage_height = round(camera_width / STAGE_ASPECT_RATIO)
    if stage_height >= camera_height:
        return camera_width, stage_height

    stage_width = round(camera_height * STAGE_ASPECT_RATIO)
    return stage_width, camera_height


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
    """Spawn one free-play node at the top of the generated stage."""
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
    instrument = INSTRUMENT_BY_KEY[event.instrument]

    node = FallingNode(
        x=float(candidate_x),
        y=float(radius),
        radius=radius,
        speed=speed,
        color=instrument.color,
        instrument=event.instrument,
        chart_index=event.index,
        midi_note=event.pitch,
        target_time=target_time,
        spawn_time=spawn_time,
    )
    node.y = challenge_node_y(node, current_time)
    return node


def challenge_node_y(
    node: FallingNode,
    current_time: float,
) -> float:
    """Return a timed node's position while keeping one constant speed.

    A challenge circle reaches its playable point in the upper third at its
    scheduled beat, then continues at exactly that same speed until it leaves
    the bottom of the stage.
    """
    if node.spawn_time is None:
        raise ValueError("A challenge node is missing its spawn time.")

    elapsed_since_spawn = max(0.0, current_time - node.spawn_time)
    return node.radius + node.speed * elapsed_since_spawn


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
    frame_height: int,
) -> tuple[list[FallingNode], list[NodeHit], int]:
    """Keep each visible song circle hittable until it exits at the bottom."""
    remaining_nodes = []
    node_hits = []
    missed_node_count = 0
    rating_priority = {"GOOD": 1, "GREAT": 2, "PERFECT": 3}

    for node in active_nodes:
        if node.chart_index is None or node.target_time is None or node.spawn_time is None:
            raise ValueError("A challenge node is missing its chart timing.")

        # Absolute time prevents the music and circles from drifting if one
        # camera frame is slow. Both jump to the correct point on the same clock.
        node.y = challenge_node_y(node, current_time)
        timing_error = current_time - node.target_time
        collision_radius = node.radius + FINGERTIP_TOUCH_RADIUS
        node_center = (round(node.x), round(node.y))

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
            in_order = rhythm_round.timing_bonus_available(node.chart_index)
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
                    in_order=in_order,
                )
            )
        elif node.y - node.radius > frame_height:
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
    """Draw ordered music targets and make the next chart note brighter."""
    transparent_overlay = frame.copy()
    chart_indices = [
        node.chart_index
        for node in active_nodes
        if node.chart_index is not None
    ]
    next_chart_index = min(chart_indices) if chart_indices else None

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
        is_next = (
            next_chart_index is not None
            and node.chart_index == next_chart_index
        )
        border_color = node.color
        border_thickness = 4
        if is_next:
            # Brighten the existing target itself.  There is deliberately no
            # extra timing ring or fixed target marker.
            highlight_overlay = frame.copy()
            brighter_color = tuple(
                min(255, round(channel * 0.62 + 255 * 0.38))
                for channel in node.color
            )
            cv2.circle(
                highlight_overlay,
                center,
                node.radius,
                brighter_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.addWeighted(highlight_overlay, 0.24, frame, 0.76, 0, frame)
            border_color = brighter_color
            border_thickness = 6

        cv2.circle(
            frame,
            center,
            node.radius,
            border_color,
            border_thickness,
            cv2.LINE_AA,
        )
        note_label = (
            note_name(node.midi_note)
            if node.midi_note is not None
            else INSTRUMENT_BY_KEY[node.instrument].label
        )
        label = (
            f"{node.chart_index + 1:02d} / {note_label}"
            if node.chart_index is not None
            else note_label
        )
        label_scale = min(0.48, node.radius / max(90, len(label) * 15))
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
        "HIT": (210, 145, 255),
        "TOUCH": (255, 255, 255),
        "STRONG": (40, 210, 255),
        "POWER": (80, 255, 80),
    }

    for effect in hit_effects:
        progress = (current_time - effect.started_at) / HIT_EFFECT_DURATION_SECONDS
        ring_radius = int(25 + progress * 45)
        ui.draw_glow_circle(
            frame,
            effect.position,
            ring_radius,
            effect.color,
            1.0 - progress,
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


def configure_display_window() -> None:
    """Use a resizable, ratio-preserving window for the generated stage.

    The stage itself is rendered at 16:10, so this preserves circular targets
    while fitting a maximized modern Mac display without a large letterbox.
    """
    cv2.namedWindow(
        WINDOW_TITLE,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )
    aspect_property = getattr(cv2, "WND_PROP_ASPECT_RATIO", None)
    if aspect_property is None:
        return
    try:
        cv2.setWindowProperty(
            WINDOW_TITLE,
            aspect_property,
            cv2.WINDOW_KEEPRATIO,
        )
    except cv2.error:
        # Some HighGUI backends do not expose the aspect-ratio property.  The
        # named-window flag above remains the supported fallback.
        pass


def main() -> None:
    """Run the portfolio interface, timed challenge, and free-play mode."""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open the camera. Check that it is connected and that "
            "your terminal or editor has Camera permission in macOS Settings."
        )

    camera_configuration = configure_camera_capture(camera)
    configure_display_window()
    audio = AudioEngine(ALL_PITCHES)
    benchmark = PerformanceBenchmark()
    camera_stream = LatestFrameCamera(camera)
    try:
        camera_stream.start()
        if not audio.start():
            print(audio.error_message or "Sound unavailable.")
            print("Air Rhythm will keep running. Check your sound output and restart.")
        reported_audio_error = audio.error_message
        with create_hand_landmarker() as hand_landmarker:
            hand_stabilizer = HandLandmarkStabilizer()
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
            screen = AppScreen.TITLE
            show_help = False
            help_opened_at: float | None = None
            show_debug = False
            smoothed_fps: float | None = None
            stage_template: np.ndarray | None = None
            last_render_ms = 0.0
            last_complete_frame_ms = 0.0
            benchmark_notice: str | None = None
            benchmark_notice_until = 0.0
            last_camera_sequence = 0

            while True:
                frame_pipeline_started_at = time.perf_counter()
                captured_frame = camera_stream.read_latest(
                    after_sequence=last_camera_sequence,
                    timeout=1.0,
                )
                if captured_frame is None:
                    if camera_stream.failed:
                        print("The camera stopped returning frames. Closing Air Rhythm.")
                    else:
                        print("Timed out waiting for a camera frame. Closing Air Rhythm.")
                    break
                last_camera_sequence = captured_frame.sequence
                frame = captured_frame.image
                camera_capture_ms = captured_frame.camera_read_ms
                camera_wait_ms = captured_frame.main_thread_wait_ms
                camera_frames_skipped = captured_frame.skipped_since_previous

                captured_height, captured_width = frame.shape[:2]
                preprocessing_started_at = time.perf_counter()
                processing_frame = resize_frame_for_processing(frame)
                mirrored_frame = cv2.flip(processing_frame, 1)
                rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                camera_preprocessing_ms = (
                    time.perf_counter() - preprocessing_started_at
                ) * 1000

                timestamp_ms = max(
                    int(time.monotonic() * 1000),
                    previous_timestamp_ms + 1,
                )
                previous_timestamp_ms = timestamp_ms
                inference_started_at = time.perf_counter()
                raw_detection_result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
                inference_ms = (time.perf_counter() - inference_started_at) * 1000
                game_update_started_at = time.perf_counter()

                camera_height, camera_width = mirrored_frame.shape[:2]
                stage_width, stage_height = stage_dimensions_for_camera(
                    camera_width,
                    camera_height,
                )
                desired_stage_shape = (stage_height, stage_width, 3)
                if (
                    stage_template is None
                    or stage_template.shape != desired_stage_shape
                    or stage_template.dtype != mirrored_frame.dtype
                ):
                    stage_template = np.empty(
                        desired_stage_shape,
                        dtype=mirrored_frame.dtype,
                    )
                current_time = time.monotonic()
                stabilized_hands = hand_stabilizer.update(
                    raw_detection_result.hand_landmarks,
                    raw_detection_result.handedness,
                    current_time,
                )
                detection_result = stabilized_hands.active
                visible_hand_landmarks = (
                    stabilized_hands.visible.hand_landmarks
                )
                fingertip_motions, fingertip_history = collect_fingertip_motions(
                    detection_result,
                    stage_width,
                    stage_height,
                    current_time,
                    fingertip_history,
                )
                active_strikes = {
                    motion.strike_direction
                    for motion in fingertip_motions
                    if motion.strike_direction is not None
                }

                raw_frame_seconds = current_time - previous_frame_time
                elapsed_seconds = max(
                    0.0,
                    min(
                        raw_frame_seconds,
                        MAX_FRAME_TIME_SECONDS,
                    ),
                )
                if raw_frame_seconds > 0:
                    instantaneous_fps = 1.0 / raw_frame_seconds
                    smoothed_fps = (
                        instantaneous_fps
                        if smoothed_fps is None
                        else smoothed_fps * 0.85 + instantaneous_fps * 0.15
                    )
                previous_frame_time = current_time

                node_hits: list[NodeHit] = []
                if (
                    screen is AppScreen.GAMEPLAY
                    and not show_help
                    and game_mode == GAME_MODE_CHALLENGE
                ):
                    for event in rhythm_round.due_events(current_time):
                        active_nodes.append(
                            create_challenge_node(
                                event,
                                rhythm_round,
                                stage_width,
                                stage_height,
                                current_time,
                                active_nodes,
                            )
                        )
                    active_nodes, node_hits, _ = update_challenge_nodes(
                        active_nodes,
                        current_time,
                        fingertip_motions,
                        rhythm_round,
                        stage_height,
                    )
                    hit_count = rhythm_round.score.total_hits
                    miss_count = rhythm_round.score.misses
                    if (
                        rhythm_round.phase_at(current_time) is RoundPhase.RESULTS
                        and not active_nodes
                    ):
                        screen = AppScreen.RESULTS
                elif screen is AppScreen.GAMEPLAY and not show_help:
                    if (
                        current_time - last_spawn_time >= NODE_SPAWN_INTERVAL_SECONDS
                        and len(active_nodes) < MAX_ACTIVE_NODES
                    ):
                        active_nodes.append(
                            create_falling_node(
                                stage_width,
                                stage_height,
                                active_nodes,
                            )
                        )
                        last_spawn_time = current_time

                    active_nodes, node_hits, new_misses = update_falling_nodes(
                        active_nodes,
                        elapsed_seconds,
                        fingertip_motions,
                        stage_height,
                    )
                    hit_count += len(node_hits)
                    miss_count += new_misses

                if node_hits:
                    play_node_hits(
                        node_hits,
                        audio,
                        melody,
                        game_mode == GAME_MODE_CHALLENGE,
                    )
                    benchmark.record_audio_request(
                        (time.perf_counter() - frame_pipeline_started_at) * 1000
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
                        if node_hit.in_order is False:
                            timing_description = "OUT OF ORDER"
                        else:
                            timing_milliseconds = round(
                                abs(node_hit.timing_error or 0.0) * 1000
                            )
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
                game_update_ms = (
                    time.perf_counter() - game_update_started_at
                ) * 1000

                hand_count = len(detection_result.hand_landmarks)
                if hand_count == 0:
                    motion_status = "NO HAND"
                elif active_strikes:
                    motion_status = " + ".join(sorted(active_strikes))
                else:
                    motion_status = "NO DIRECTIONAL STRIKE"
                debug_info = {
                    "fps": "--" if smoothed_fps is None else f"{smoothed_fps:.1f}",
                    "hands": hand_count,
                    "landmarks": sum(
                        len(hand_landmarks)
                        for hand_landmarks in detection_result.hand_landmarks
                    ),
                    "confidence": handedness_confidences(detection_result),
                    "capture_ms": f"{camera_capture_ms:.1f}",
                    "camera_wait_ms": f"{camera_wait_ms:.1f}",
                    "preprocessing_ms": f"{camera_preprocessing_ms:.1f}",
                    "inference_ms": f"{inference_ms:.1f}",
                    "update_ms": f"{game_update_ms:.1f}",
                    "render_ms": f"{last_render_ms:.1f}",
                    "frame_ms": f"{last_complete_frame_ms:.1f}",
                    "gesture": motion_status,
                }
                privacy_label = PRIVACY_LABELS[privacy_renderer.mode]
                sound_label = audio_status_label(audio)

                # The only place where camera pixels are shown is the compact
                # live-input inset.  The main display below is generated from
                # scratch, so the performer remains off the music stage.
                render_started_at = time.perf_counter()
                input_preview = privacy_renderer.apply(
                    mirrored_frame,
                    detection_result.hand_landmarks,
                )
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_skeleton(input_preview, hand_landmarks)
                draw_strike_feedback(input_preview, fingertip_motions)

                display_frame = create_performance_stage(
                    stage_template,
                    current_time,
                )
                presentation_time = (
                    help_opened_at
                    if (
                        show_help
                        and screen is AppScreen.GAMEPLAY
                        and help_opened_at is not None
                    )
                    else current_time
                )
                if screen is not AppScreen.TITLE:
                    draw_falling_nodes(
                        display_frame,
                        active_nodes,
                        (
                            presentation_time
                            if game_mode == GAME_MODE_CHALLENGE
                            else None
                        ),
                    )
                    draw_hit_effects(display_frame, hit_effects, current_time)

                # Landmark 8 anchors each visible drumstick tip, while the
                # real hand and full skeleton remain visible in the inset.
                draw_virtual_drumsticks(
                    display_frame,
                    visible_hand_landmarks,
                )
                draw_collision_points(
                    display_frame,
                    detection_result.hand_landmarks,
                )

                if screen is AppScreen.TITLE:
                    ui.draw_title_screen(
                        display_frame,
                        hand_count=hand_count,
                        privacy_label=privacy_label,
                        sound_label=sound_label,
                        current_time=current_time,
                    )
                elif screen is AppScreen.GAMEPLAY:
                    challenge_mode = game_mode == GAME_MODE_CHALLENGE
                    score = rhythm_round.score
                    ui.draw_game_hud(
                        display_frame,
                        hands=hand_count,
                        score=score.score if challenge_mode else hit_count * 100,
                        combo=score.combo if challenge_mode else 0,
                        progress=(
                            rhythm_round.progress_at(presentation_time)
                            if challenge_mode
                            else 0.0
                        ),
                        hits=hit_count,
                        misses=miss_count,
                        mode_label="Challenge" if challenge_mode else "Free play",
                        song_label=(
                            MELODY_TITLE
                            if challenge_mode
                            else "Four-instrument free play"
                        ),
                        privacy_label=privacy_label,
                        sound_label=sound_label,
                    )
                    if challenge_mode:
                        remaining = rhythm_round.countdown_remaining(
                            presentation_time
                        )
                        if remaining > 0:
                            ui.draw_countdown(
                                display_frame,
                                str(max(1, math.ceil(remaining))),
                                progress=(
                                    1.0
                                    - remaining / rhythm_round.countdown_seconds
                                    if rhythm_round.countdown_seconds > 0
                                    else 1.0
                                ),
                                current_time=presentation_time,
                            )
                        elif presentation_time - rhythm_round.song_start_time < 0.5:
                            ui.draw_countdown(
                                display_frame,
                                "GO!",
                                progress=1.0,
                                current_time=presentation_time,
                            )
                else:
                    score = rhythm_round.score
                    ui.draw_results(
                        display_frame,
                        score=score.score,
                        completion=score.completion_accuracy,
                        rank=score.rank,
                        perfect=score.perfect,
                        great=score.great,
                        good=score.good,
                        misses=score.misses,
                        max_combo=score.max_combo,
                        basic_hits=score.basic_hits,
                        song_label=f"{MELODY_TITLE} challenge",
                        replay_hint="R / SPACE  Replay     T  Title",
                    )

                if show_debug:
                    ui.draw_debug_overlay(
                        display_frame,
                        debug_info,
                        top_offset=max(50, int(stage_height * 0.13)),
                    )

                draw_camera_inset(
                    display_frame,
                    input_preview,
                    mode_label=privacy_label,
                    hand_count=hand_count,
                )

                if show_help:
                    ui.draw_help_overlay(display_frame)

                benchmark_clock = time.perf_counter()
                if benchmark_notice_until <= benchmark_clock:
                    benchmark_notice = None
                if benchmark.active or benchmark_notice:
                    ui.draw_benchmark_status(
                        display_frame,
                        benchmark.live_summary(benchmark_clock),
                        notice=benchmark_notice,
                    )

                cv2.imshow(WINDOW_TITLE, display_frame)
                last_render_ms = (
                    time.perf_counter() - render_started_at
                ) * 1000
                last_complete_frame_ms = (
                    time.perf_counter() - frame_pipeline_started_at
                ) * 1000
                benchmark.record_frame(
                    camera_capture_ms=camera_capture_ms,
                    camera_wait_ms=camera_wait_ms,
                    camera_preprocessing_ms=camera_preprocessing_ms,
                    mediapipe_inference_ms=inference_ms,
                    game_update_ms=game_update_ms,
                    rendering_ms=last_render_ms,
                    complete_frame_ms=last_complete_frame_ms,
                    frame_interval_ms=raw_frame_seconds * 1000,
                    camera_frames_skipped=camera_frames_skipped,
                )

                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord("q"):
                    break
                if pressed_key == ord("p"):
                    privacy_renderer.cycle()
                elif pressed_key == ord("m"):
                    audio.set_muted(not audio.muted)
                elif pressed_key == ord("h"):
                    if show_help:
                        if (
                            screen is AppScreen.GAMEPLAY
                            and help_opened_at is not None
                        ):
                            pause_seconds = max(0.0, current_time - help_opened_at)
                            if game_mode == GAME_MODE_CHALLENGE:
                                rhythm_round.delay_timeline(pause_seconds)
                                for node in active_nodes:
                                    if node.target_time is not None:
                                        node.target_time += pause_seconds
                                    if node.spawn_time is not None:
                                        node.spawn_time += pause_seconds
                            else:
                                last_spawn_time += pause_seconds
                            previous_frame_time = current_time
                            fingertip_history.clear()
                        show_help = False
                        help_opened_at = None
                    else:
                        show_help = True
                        help_opened_at = current_time
                        audio.stop_all()
                elif pressed_key == ord("d"):
                    show_debug = not show_debug
                elif pressed_key == ord("b"):
                    if benchmark.active:
                        try:
                            finish_benchmark_session(benchmark)
                            benchmark_notice = "BENCHMARK SAVED"
                        except OSError as error:
                            print(f"Could not save benchmark report: {error}")
                            benchmark_notice = "BENCHMARK SAVE FAILED"
                        benchmark_notice_until = time.perf_counter() + 3.0
                    else:
                        benchmark.start(
                            metadata={
                                "requested_camera_resolution": (
                                    f"{CAMERA_REQUEST_WIDTH}x{CAMERA_REQUEST_HEIGHT}"
                                ),
                                "requested_camera_fps": CAMERA_REQUEST_FPS,
                                "reported_camera_resolution": (
                                    f"{int(round(camera_configuration['width']))}x"
                                    f"{int(round(camera_configuration['height']))}"
                                    if camera_configuration["width"] is not None
                                    and camera_configuration["height"] is not None
                                    else "Unavailable"
                                ),
                                "reported_camera_fps": (
                                    camera_configuration["fps"]
                                    if camera_configuration["fps"] is not None
                                    else "Unavailable"
                                ),
                                "camera_capture_mode": "Background latest-frame",
                                "captured_camera_resolution": (
                                    f"{captured_width}x{captured_height}"
                                ),
                                "processing_resolution": (
                                    f"{camera_width}x{camera_height}"
                                ),
                                "stage_resolution": f"{stage_width}x{stage_height}",
                                "screen": screen.value,
                                "game_mode": game_mode,
                                "input_view": privacy_label,
                                "sound_status": sound_label,
                                "maximum_hands": MAX_HANDS,
                                "hand_model": "MediaPipe Hand Landmarker",
                                "landmark_filter": "Adaptive One Euro-style",
                                "landmark_min_cutoff": hand_stabilizer.min_cutoff,
                                "landmark_speed_coefficient": (
                                    hand_stabilizer.speed_coefficient
                                ),
                                "visual_dropout_grace_ms": round(
                                    hand_stabilizer.dropout_grace_seconds * 1000
                                ),
                            }
                        )
                        benchmark_notice = "BENCHMARK RECORDING"
                        benchmark_notice_until = 0.0
                elif pressed_key in (ord("t"), 27):
                    screen = AppScreen.TITLE
                    game_mode = GAME_MODE_CHALLENGE
                    show_help = False
                    help_opened_at = None
                    melody.reset()
                    audio.stop_all()
                    active_nodes.clear()
                    hit_effects.clear()
                    fingertip_history.clear()
                    hit_count = 0
                    miss_count = 0
                else:
                    requested_mode = None
                    if pressed_key == ord("1"):
                        requested_mode = GAME_MODE_CHALLENGE
                    elif pressed_key == ord("2"):
                        requested_mode = GAME_MODE_FREE_PLAY
                    elif pressed_key == ord("r"):
                        requested_mode = game_mode
                    elif pressed_key == ord(" ") and screen in (
                        AppScreen.TITLE,
                        AppScreen.RESULTS,
                    ):
                        requested_mode = GAME_MODE_CHALLENGE

                    if requested_mode is not None:
                        game_mode = requested_mode
                        screen = AppScreen.GAMEPLAY
                        show_help = False
                        help_opened_at = None
                        rhythm_round.reset(started_at=current_time)
                        melody.reset()
                        audio.stop_all()
                        active_nodes.clear()
                        hit_effects.clear()
                        fingertip_history.clear()
                        hit_count = 0
                        miss_count = 0
                        previous_frame_time = current_time
                        last_spawn_time = (
                            current_time - NODE_SPAWN_INTERVAL_SECONDS
                        )
    finally:
        if benchmark.active:
            try:
                finish_benchmark_session(benchmark)
            except OSError as error:
                print(f"Could not save benchmark report: {error}")
        camera_stream.close()
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
