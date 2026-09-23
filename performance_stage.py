"""Person-free performance-stage rendering for Air Rhythm.

The main game view is generated from scratch; it never copies webcam pixels.
MediaPipe landmarks are used only as coordinates for virtual drumsticks.  The
real camera feed is deliberately contained in a separate live-input inset so a
viewer can still see the hand-tracking evidence without putting the player on
the main stage.
"""

from collections import OrderedDict
import math

import cv2
import numpy as np


HAND_LANDMARK_COUNT = 21
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
MAX_CACHED_BACKGROUNDS = 3

# OpenCV uses BGR.  The materials are deliberately quieter than the playable
# circles, so a target remains the most obvious object to reach for.
STAGE_INK = (0, 0, 0)
STAGE_PANEL = (20, 20, 20)
STAGE_BORDER = (92, 92, 92)
STICK_WOOD = (88, 154, 224)
STICK_HIGHLIGHT = (194, 226, 249)
STICK_SHADOW = (10, 13, 20)
STICK_ACCENTS = ((255, 221, 71), (224, 94, 255))

_background_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()


def _stick_accent(hand, fallback_index: int) -> tuple[int, int, int]:
    """Keep each hand's colour stable when detector result order changes."""
    identity = str(getattr(hand, "identity", "")).casefold()
    if identity == "left":
        return STICK_ACCENTS[0]
    if identity == "right":
        return STICK_ACCENTS[1]
    return STICK_ACCENTS[fallback_index % len(STICK_ACCENTS)]


def _frame_size(frame: np.ndarray) -> tuple[int, int]:
    """Return an image's width and height with clear stage-rendering errors."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy image")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a three-channel BGR image")
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("frame must contain at least one pixel")
    return width, height


def _filled_round_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw a filled rounded rectangle using only OpenCV primitives."""
    left, top, right, bottom = rect
    radius = max(0, min(radius, (right - left) // 2, (bottom - top) // 2))
    if radius <= 1:
        cv2.rectangle(image, (left, top), (right, bottom), color, -1)
        return
    cv2.rectangle(image, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(image, (left, top + radius), (right, bottom - radius), color, -1)
    for center in (
        (left + radius, top + radius),
        (right - radius, top + radius),
        (left + radius, bottom - radius),
        (right - radius, bottom - radius),
    ):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def _create_background(height: int, width: int) -> np.ndarray:
    """Cache a sparse star field over black, without using camera pixels."""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    random = np.random.default_rng(width * 100_003 + height)
    star_count = max(24, min(170, width * height // 6_500))
    for _ in range(star_count):
        point = (int(random.integers(width)), int(random.integers(height)))
        prominence = random.random()
        radius = 2 if prominence > 0.96 else 1 if prominence > 0.76 else 0
        brightness = int(random.integers(48, 112 if radius else 82))
        cv2.circle(
            background,
            point,
            radius,
            (brightness, brightness, brightness),
            -1,
            cv2.LINE_AA,
        )
    return background


def _background_for(height: int, width: int) -> np.ndarray:
    """Get a cached immutable base stage for the requested resolution."""
    key = (height, width)
    cached = _background_cache.get(key)
    if cached is not None:
        _background_cache.move_to_end(key)
        return cached
    background = _create_background(height, width)
    background.setflags(write=False)
    _background_cache[key] = background
    if len(_background_cache) > MAX_CACHED_BACKGROUNDS:
        _background_cache.popitem(last=False)
    return background


def create_performance_stage(frame: np.ndarray, current_time: float = 0.0) -> np.ndarray:
    """Return a generated performance scene matching ``frame``'s dimensions.

    The input frame supplies only shape information.  No pixel from it is
    copied into the returned scene, which keeps the main presentation private.
    """
    width, height = _frame_size(frame)
    stage = _background_for(height, width).copy()
    time_value = float(current_time) if isinstance(current_time, (int, float)) else 0.0
    if not math.isfinite(time_value):
        time_value = 0.0

    # Faint neutral outlines retain gentle motion without tinting the black
    # stage or competing with the playable circles and virtual drumsticks.
    phase = time_value * 0.55
    center = (
        round(width * (0.50 + 0.10 * math.sin(phase))),
        round(height * (0.48 + 0.05 * math.cos(phase * 0.83))),
    )
    for multiplier, color in ((0.18, (24, 24, 24)), (0.30, (16, 16, 16))):
        radius = max(25, round(min(width, height) * multiplier))
        cv2.circle(stage, center, radius, color, 1, cv2.LINE_AA)

    # A few tiny glints breathe slowly; the falling notes remain the only
    # bright moving objects on the stage.
    for x_fraction, y_fraction, offset in (
        (0.13, 0.26, 0.0),
        (0.83, 0.18, 1.7),
        (0.21, 0.73, 3.1),
        (0.76, 0.68, 4.8),
    ):
        point = (round(width * x_fraction), round(height * y_fraction))
        pulse = 0.5 + 0.5 * math.sin(time_value * 1.2 + offset)
        brightness = round(48 + pulse * 48)
        arm = max(1, round(min(width, height) * 0.003))
        ray_color = (brightness // 3,) * 3
        cv2.line(
            stage,
            (point[0] - arm, point[1]),
            (point[0] + arm, point[1]),
            ray_color,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            stage,
            (point[0], point[1] - arm),
            (point[0], point[1] + arm),
            ray_color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(stage, point, 1, (brightness,) * 3, -1, cv2.LINE_AA)
    return stage


def _hand_points(hand, width: int, height: int) -> np.ndarray | None:
    """Convert one MediaPipe-like hand into safe integer display points."""
    try:
        landmarks = list(hand)
    except TypeError:
        return None
    if len(landmarks) < HAND_LANDMARK_COUNT:
        return None

    points = []
    for landmark in landmarks[:HAND_LANDMARK_COUNT]:
        try:
            x = float(landmark.x)
            y = float(landmark.y)
        except (AttributeError, TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        points.append(
            (
                int(np.clip(x * width, 0, width - 1)),
                int(np.clip(y * height, 0, height - 1)),
            )
        )
    return np.asarray(points, dtype=np.float32)


def _draw_stick(
    frame: np.ndarray,
    handle: tuple[int, int],
    tip: tuple[int, int],
    body_radius: int,
    accent: tuple[int, int, int],
) -> None:
    """Draw one softly glowing digital drumstick inside a compact local area."""
    width, height = _frame_size(frame)
    glow_padding = body_radius * 5 + 8
    left = max(0, min(handle[0], tip[0]) - glow_padding)
    right = min(width, max(handle[0], tip[0]) + glow_padding + 1)
    top = max(0, min(handle[1], tip[1]) - glow_padding)
    bottom = min(height, max(handle[1], tip[1]) + glow_padding + 1)
    if left >= right or top >= bottom:
        return

    roi = frame[top:bottom, left:right]
    local_handle = (handle[0] - left, handle[1] - top)
    local_tip = (tip[0] - left, tip[1] - top)
    glow = roi.copy()
    cv2.line(
        glow,
        local_handle,
        local_tip,
        accent,
        body_radius * 5,
        cv2.LINE_AA,
    )
    cv2.circle(glow, local_tip, body_radius * 3, accent, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.16, roi, 0.84, 0, roi)

    cv2.line(
        frame,
        handle,
        tip,
        STICK_SHADOW,
        body_radius * 2 + 5,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        handle,
        tip,
        STICK_WOOD,
        body_radius * 2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        handle,
        tip,
        STICK_HIGHLIGHT,
        max(2, body_radius // 2),
        cv2.LINE_AA,
    )
    cv2.circle(frame, handle, body_radius, STICK_WOOD, -1, cv2.LINE_AA)
    cv2.circle(frame, tip, body_radius + 3, STICK_SHADOW, -1, cv2.LINE_AA)
    cv2.circle(frame, tip, body_radius + 1, accent, -1, cv2.LINE_AA)
    cv2.circle(frame, tip, max(2, body_radius // 2), STICK_HIGHLIGHT, -1, cv2.LINE_AA)


def draw_virtual_drumsticks(frame: np.ndarray, hand_landmarks) -> np.ndarray:
    """Replace each detected hand with a screen-aligned virtual drumstick.

    A stick's tip is pinned to landmark 8 (the index fingertip), exactly where
    the rhythm collision system already measures contact.  Its body points
    back through the index finger toward landmark 5, so it rotates naturally
    as the player turns their hand.  This is a 2D visual proxy, not a claim to
    reconstruct a physical drumstick in three dimensions.
    """
    width, height = _frame_size(frame)
    min_dimension = min(width, height)
    if hand_landmarks is None:
        return frame

    for hand_index, hand in enumerate(hand_landmarks):
        points = _hand_points(hand, width, height)
        if points is None:
            continue
        index_base = points[5]
        index_tip = points[8]
        direction = index_tip - index_base
        direction_length = float(np.linalg.norm(direction))
        if direction_length < 4.0:
            direction = points[12] - points[9]
            direction_length = float(np.linalg.norm(direction))
        if direction_length < 1.0:
            continue

        unit = direction / direction_length
        stick_length = float(
            np.clip(direction_length * 1.85, min_dimension * 0.16, min_dimension * 0.38)
        )
        tip = tuple(np.rint(index_tip).astype(int))
        handle = tuple(np.rint(index_tip - unit * stick_length).astype(int))
        body_radius = max(4, round(min_dimension * 0.0105))
        accent = _stick_accent(hand, hand_index)
        _draw_stick(frame, handle, tip, body_radius, accent)
    return frame


def draw_collision_points(frame: np.ndarray, hand_landmarks) -> np.ndarray:
    """Show every fingertip position used by the collision system.

    The index-finger point sits directly on the virtual drumstick tip.  Four
    smaller points make the other active fingertips visible as well, so the
    generated stage never hides where a collision can actually happen.
    """
    width, height = _frame_size(frame)
    min_dimension = min(width, height)
    if hand_landmarks is None:
        return frame

    core_radius = max(2, round(min_dimension * 0.0042))
    glow_radius = core_radius * 3
    for hand_index, hand in enumerate(hand_landmarks):
        points = _hand_points(hand, width, height)
        if points is None:
            continue
        accent = _stick_accent(hand, hand_index)
        for fingertip_index in FINGERTIP_INDICES:
            point = tuple(np.rint(points[fingertip_index]).astype(int))
            left = max(0, point[0] - glow_radius)
            right = min(width, point[0] + glow_radius + 1)
            top = max(0, point[1] - glow_radius)
            bottom = min(height, point[1] + glow_radius + 1)
            if left >= right or top >= bottom:
                continue

            roi = frame[top:bottom, left:right]
            glow = roi.copy()
            local_point = (point[0] - left, point[1] - top)
            cv2.circle(
                glow,
                local_point,
                glow_radius,
                accent,
                -1,
                cv2.LINE_AA,
            )
            cv2.addWeighted(glow, 0.18, roi, 0.82, 0, roi)
            cv2.circle(frame, point, core_radius + 1, (18, 22, 34), -1, cv2.LINE_AA)
            cv2.circle(frame, point, core_radius, accent, -1, cv2.LINE_AA)
            cv2.circle(
                frame,
                point,
                max(1, core_radius // 2),
                (248, 251, 255),
                -1,
                cv2.LINE_AA,
            )
    return frame


def _fit_camera_inset(
    frame_width: int,
    frame_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int, int, int]:
    """Return outer and inner bounds for a bottom-right camera inset."""
    minimum = min(frame_width, frame_height)
    margin = max(8, round(minimum * 0.022))
    padding = max(3, round(minimum * 0.010))
    header_height = max(14, round(minimum * 0.032))
    available_width = max(1, frame_width - margin * 2)
    available_height = max(1, frame_height - margin * 2 - header_height - padding * 2)
    inner_width = min(max(72, round(frame_width * 0.29)), available_width - padding * 2)
    aspect_ratio = source_width / max(1, source_height)
    inner_height = max(1, round(inner_width / max(0.1, aspect_ratio)))
    max_inner_height = max(1, min(round(frame_height * 0.31), available_height))
    if inner_height > max_inner_height:
        inner_height = max_inner_height
        inner_width = max(1, round(inner_height * aspect_ratio))

    outer_width = inner_width + padding * 2
    outer_height = inner_height + header_height + padding * 2
    right = frame_width - margin
    bottom = frame_height - margin
    left = max(0, right - outer_width)
    top = max(0, bottom - outer_height)
    return left, top, right, bottom, padding, header_height


def draw_camera_inset(
    frame: np.ndarray,
    camera_frame: np.ndarray,
    *,
    mode_label: str,
    hand_count: int,
) -> tuple[int, int, int, int]:
    """Draw the real camera plus skeleton evidence in a compact inset.

    ``camera_frame`` is expected to have already received the selected privacy
    treatment and landmark skeleton.  The returned bounds make the placement
    easy to inspect in camera-free tests.
    """
    frame_width, frame_height = _frame_size(frame)
    source_width, source_height = _frame_size(camera_frame)
    left, top, right, bottom, padding, header_height = _fit_camera_inset(
        frame_width,
        frame_height,
        source_width,
        source_height,
    )
    corner_radius = max(5, round(min(frame_width, frame_height) * 0.018))

    # Soft shadow first, then a calm glass-like outer shell.  This keeps the
    # live evidence visually separate without looking like a heavy old panel.
    shadow_left = min(frame_width - 1, left + 3)
    shadow_top = min(frame_height - 1, top + 4)
    shadow_right = min(frame_width - 1, right + 3)
    shadow_bottom = min(frame_height - 1, bottom + 4)
    _filled_round_rect(
        frame,
        (shadow_left, shadow_top, shadow_right, shadow_bottom),
        (5, 7, 12),
        corner_radius,
    )
    _filled_round_rect(frame, (left, top, right, bottom), STAGE_PANEL, corner_radius)
    cv2.rectangle(frame, (left, top), (right, bottom), STAGE_BORDER, 1, cv2.LINE_AA)

    image_left = left + padding
    image_top = top + header_height + padding
    image_right = max(image_left + 1, right - padding)
    image_bottom = max(image_top + 1, bottom - padding)
    image_width = image_right - image_left
    image_height = image_bottom - image_top
    interpolation = cv2.INTER_AREA if camera_frame.shape[0] >= image_height else cv2.INTER_LINEAR
    preview = cv2.resize(camera_frame, (image_width, image_height), interpolation=interpolation)
    frame[image_top:image_bottom, image_left:image_right] = preview
    cv2.rectangle(
        frame,
        (image_left, image_top),
        (image_right - 1, image_bottom - 1),
        (158, 166, 197),
        1,
        cv2.LINE_AA,
    )

    scale = max(0.28, min(0.52, min(frame_width / 1280.0, frame_height / 720.0) * 0.48))
    text_y = top + max(11, header_height - 4)
    cv2.circle(frame, (left + padding + 3, text_y - 4), 3, (103, 239, 112), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "LIVE INPUT",
        (left + padding + 10, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (238, 244, 255),
        1,
        cv2.LINE_AA,
    )
    status = f"{str(mode_label).upper()} | {max(0, int(hand_count))}/2"
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.78, 1)[0]
    cv2.putText(
        frame,
        status,
        (max(left + padding + 10, right - padding - status_size[0]), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.78,
        (182, 193, 215),
        1,
        cv2.LINE_AA,
    )
    return left, top, right, bottom


__all__ = [
    "create_performance_stage",
    "draw_camera_inset",
    "draw_collision_points",
    "draw_virtual_drumsticks",
]
