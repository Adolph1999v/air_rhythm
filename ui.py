"""Reusable OpenCV interface components for Air Rhythm.

The functions in this file only draw on a frame.  They do not open a camera,
run a hand model, or play audio, which keeps the visual layer easy to reuse
and test.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any

import cv2


# OpenCV colours use blue, green, red (BGR) order.
INK = (10, 12, 22)
TEXT_SHADOW = (20, 23, 34)
PANEL = (24, 27, 42)
PANEL_LIGHT = (48, 55, 76)
WHITE = (244, 248, 255)
MUTED = (176, 187, 211)
CYAN = (255, 222, 67)
BLUE = (255, 137, 44)
MAGENTA = (229, 72, 255)
GREEN = (112, 246, 104)
GOLD = (75, 213, 255)
RED = (90, 95, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_CONTROLS = (
    ("SPACE", "Start / replay"),
    ("1 / 2", "Challenge / free play"),
    ("R", "Restart current mode"),
    ("B", "Start / save benchmark"),
    ("P", "Input inset privacy"),
    ("M", "Mute"),
    ("D", "CV / ML details"),
    ("T / ESC", "Return to title"),
    ("Q", "Quit"),
)


def _frame_size(frame) -> tuple[int, int]:
    """Return width and height, with a clear error for an unusable frame."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be an OpenCV image")
    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("frame must contain at least one pixel")
    return width, height


def _ui_scale(frame) -> float:
    """Scale spacing and text for both 4:3 and 16:9 camera frames."""
    width, height = _frame_size(frame)
    return max(0.32, min(1.6, min(width / 1280.0, height / 720.0)))


def _clamp01(value: float) -> float:
    """Keep a progress value inside 0 to 1, including invalid numbers."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _safe_int(value: Any, default: int = 0) -> int:
    """Turn a display value into an integer without breaking the UI."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return int(number) if math.isfinite(number) else default


def _clip_rect(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clip a rectangle to the visible image and preserve point order."""
    width, height = _frame_size(frame)
    left, right = sorted((int(top_left[0]), int(bottom_right[0])))
    top, bottom = sorted((int(top_left[1]), int(bottom_right[1])))
    left = max(0, min(width - 1, left))
    right = max(left, min(width - 1, right))
    top = max(0, min(height - 1, top))
    bottom = max(top, min(height - 1, bottom))
    return left, top, right, bottom


def _filled_round_rect(
    image,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw a filled rounded rectangle using basic OpenCV shapes."""
    left, top, right, bottom = rect
    radius = max(0, min(int(radius), (right - left) // 2, (bottom - top) // 2))
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


def fit_text_scale(
    text: str,
    max_width: int,
    preferred_scale: float,
    min_scale: float = 0.2,
    thickness: int = 1,
    font: int = FONT,
) -> float:
    """Return the largest useful text scale that fits a given width."""
    preferred = max(float(min_scale), float(preferred_scale))
    available = max(1, int(max_width))
    measured = cv2.getTextSize(str(text), font, preferred, max(1, thickness))[0][0]
    if measured <= available:
        return preferred
    return max(float(min_scale), preferred * available / max(1, measured))


def _ellipsize(
    text: str,
    max_width: int,
    scale: float,
    thickness: int,
    font: int,
) -> str:
    """Shorten text only when scaling alone cannot fit it."""
    result = str(text)
    if cv2.getTextSize(result, font, scale, thickness)[0][0] <= max_width:
        return result
    suffix = "..."
    while result and cv2.getTextSize(
        result + suffix, font, scale, thickness
    )[0][0] > max_width:
        result = result[:-1]
    return result + suffix if result else ""


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
    *,
    align: str = "left",
    max_width: int | None = None,
    min_scale: float = 0.2,
    outline: bool = True,
    font: int = FONT,
) -> tuple[int, int, int, int]:
    """Draw fitted, outlined text and return its visible bounding box.

    ``position`` is the text baseline.  Alignment may be ``left``, ``center``,
    or ``right``.  The dark outline keeps words readable over camera footage.
    """
    width, height = _frame_size(frame)
    if align not in {"left", "center", "right"}:
        raise ValueError("align must be left, center, or right")

    value = str(text)
    chosen_scale = max(float(min_scale), float(scale))
    if max_width is not None:
        chosen_scale = fit_text_scale(
            value, max_width, chosen_scale, min_scale, thickness, font
        )
        value = _ellipsize(value, max(1, max_width), chosen_scale, thickness, font)
    (text_width, text_height), baseline = cv2.getTextSize(
        value, font, chosen_scale, max(1, thickness)
    )
    x, y = int(position[0]), int(position[1])
    if align == "center":
        x -= text_width // 2
    elif align == "right":
        x -= text_width
    x = max(0, min(max(0, width - text_width - 1), x))
    y = max(text_height + 1, min(height - max(1, baseline) - 1, y))

    # A border around tiny one-pixel glyphs looks like a duplicate character.
    # Small labels already sit on glass panels, so reserve the border for
    # prominent text that is thick enough to carry it cleanly.
    use_outline = outline and thickness >= 2
    if use_outline:
        cv2.putText(
            frame,
            value,
            (x, y),
            font,
            chosen_scale,
            TEXT_SHADOW,
            max(2, thickness + 1),
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        value,
        (x, y),
        font,
        chosen_scale,
        color,
        max(1, thickness),
        cv2.LINE_AA,
    )
    return x, y - text_height, text_width, text_height + baseline


def draw_glass_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    accent: tuple[int, int, int] | None = CYAN,
    alpha: float = 0.78,
    radius: int | None = None,
) -> Any:
    """Draw a restrained glass surface for the generated performance stage."""
    rect = _clip_rect(frame, top_left, bottom_right)
    scale = _ui_scale(frame)
    corner_radius = max(2, int(14 * scale)) if radius is None else max(0, radius)
    overlay = frame.copy()
    _filled_round_rect(overlay, rect, PANEL, corner_radius)
    opacity = _clamp01(alpha)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)
    left, top, right, bottom = rect
    border_color = accent if accent is not None else PANEL_LIGHT
    # A thin top signal keeps hierarchy without the heavy, brightly bordered
    # card look of the earlier interface.
    if accent is not None and bottom > top and right > left:
        cv2.line(
            frame,
            (left + corner_radius, top + 1),
            (right - corner_radius, top + 1),
            accent,
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
    if border_color is not None:
        cv2.line(
            frame,
            (left + corner_radius, bottom - 1),
            (right - corner_radius, bottom - 1),
            border_color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_glow_circle(
    frame,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int] = CYAN,
    pulse: float = 0.0,
) -> Any:
    """Draw a soft neon ring for countdowns and interactive highlights."""
    width, height = _frame_size(frame)
    radius = max(1, int(radius))
    pulse = _clamp01(pulse)
    center_x, center_y = int(center[0]), int(center[1])
    largest_extra = int(14 * (0.7 + pulse))
    outer_radius = radius + largest_extra + 8
    left = max(0, center_x - outer_radius)
    right = min(width, center_x + outer_radius + 1)
    top = max(0, center_y - outer_radius)
    bottom = min(height, center_y + outer_radius + 1)
    if left >= right or top >= bottom:
        return frame

    roi = frame[top:bottom, left:right]
    local_center = (center_x - left, center_y - top)
    for extra, opacity in ((14, 0.08), (8, 0.13), (3, 0.22)):
        overlay = roi.copy()
        cv2.circle(
            overlay,
            local_center,
            radius + int(extra * (0.7 + pulse)),
            color,
            max(2, extra // 2),
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, opacity, roi, 1.0 - opacity, 0, roi)
    cv2.circle(frame, (center_x, center_y), radius, color, 2, cv2.LINE_AA)
    return frame


def draw_brand_badge(
    frame,
    subtitle: str = "AI HAND-TRACKED MUSIC",
    tech: str = "OpenCV + MediaPipe",
) -> Any:
    """Draw a compact project badge with its computer-vision technology."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    margin = max(5, int(18 * scale))
    panel_width = min(width - margin * 2, max(int(245 * scale), int(width * 0.25)))
    panel_height = min(height - margin * 2, max(int(56 * scale), 34))
    right = margin + max(1, panel_width)
    bottom = margin + max(1, panel_height)
    draw_glass_panel(frame, (margin, margin), (right, bottom), accent=CYAN, alpha=0.72)
    text_x = margin + max(7, int(13 * scale))
    first_y = margin + max(14, int(22 * scale))
    available = max(20, panel_width - max(14, int(25 * scale)))
    draw_text(
        frame, subtitle, (text_x, first_y), max(0.39, 0.58 * scale),
        CYAN, 1, max_width=available, min_scale=0.18,
    )
    second_y = min(bottom - 5, first_y + max(13, int(21 * scale)))
    draw_text(
        frame, tech, (text_x, second_y), max(0.32, 0.46 * scale),
        MUTED, 1, max_width=available, min_scale=0.16,
    )
    return frame


def _draw_background_tint(frame, opacity: float = 0.62) -> None:
    """Dim a camera frame while keeping a subtle cyan and purple stage glow."""
    width, height = _frame_size(frame)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), INK, -1)
    cv2.circle(overlay, (0, height // 2), max(width, height) // 2, (50, 29, 64), -1)
    cv2.circle(
        overlay, (width - 1, height // 3), max(width, height) // 3,
        (68, 48, 22), -1,
    )
    cv2.addWeighted(overlay, _clamp01(opacity), frame, 1.0 - _clamp01(opacity), 0, frame)


def draw_title_screen(
    frame,
    hand_count: int,
    privacy_label: str,
    sound_label: str,
    current_time: float,
    ready_progress: float | None = None,
) -> Any:
    """Draw an airy landing screen over the person-free performance stage."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    _draw_background_tint(frame, 0.18)
    center_x = width // 2
    margin = max(6, int(18 * scale))
    pulse_time = float(current_time) if isinstance(current_time, (int, float)) else 0.0
    pulse = 0.5 + 0.5 * math.sin(pulse_time * 2.4) if math.isfinite(pulse_time) else 0.5

    badge_right = min(width - margin, margin + max(112, int(210 * scale)))
    badge_bottom = min(height - margin, margin + max(23, int(38 * scale)))
    draw_glass_panel(
        frame,
        (margin, margin),
        (badge_right, badge_bottom),
        accent=CYAN,
        alpha=0.48,
        radius=max(8, int(22 * scale)),
    )
    draw_text(
        frame,
        "LIVE RHYTHM STAGE",
        (margin + max(8, int(14 * scale)), badge_bottom - max(7, int(12 * scale))),
        max(0.22, 0.39 * scale),
        CYAN,
        1,
        max_width=max(30, badge_right - margin - 18),
        min_scale=0.15,
    )

    title_y = max(42, int(height * 0.31))
    title_scale = fit_text_scale(
        "AIR RHYTHM",
        max(40, int(width * 0.74)),
        max(0.78, 2.05 * scale),
        0.42,
        max(2, int(3 * scale)),
    )
    draw_text(
        frame,
        "AIR RHYTHM",
        (center_x, title_y),
        title_scale,
        WHITE,
        max(2, int(3 * scale)),
        align="center",
        max_width=max(40, int(width * 0.74)),
    )
    signal_y = min(height - 4, title_y + max(12, int(25 * scale)))
    signal_half = max(20, int(width * (0.08 + 0.018 * pulse)))
    cv2.line(
        frame,
        (center_x - signal_half, signal_y),
        (center_x + signal_half, signal_y),
        CYAN,
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )

    subtitle_y = signal_y + max(18, int(36 * scale))
    draw_text(
        frame,
        "A PRIVATE PERFORMANCE STAGE FOR HAND-TRACKED MUSIC",
        (center_x, subtitle_y),
        max(0.29, 0.57 * scale),
        CYAN,
        1,
        align="center",
        max_width=max(40, int(width * 0.76)),
        min_scale=0.17,
    )
    detail_y = subtitle_y + max(16, int(30 * scale))
    draw_text(
        frame,
        "MediaPipe pretrained landmarks + custom gesture / collision logic",
        (center_x, detail_y),
        max(0.26, 0.45 * scale),
        MUTED,
        1,
        align="center",
        max_width=max(40, int(width * 0.76)),
        min_scale=0.16,
    )

    progress = None if ready_progress is None else _clamp01(ready_progress)
    ready = progress is None or progress >= 1.0
    action = "SPACE  START CHALLENGE" if ready else "PREPARING LANDMARK INPUT"
    action_color = GREEN if ready else GOLD
    action_width = min(max(150, int(320 * scale)), max(80, int(width * 0.60)))
    action_height = max(28, int(54 * scale))
    action_top = min(
        height - action_height - margin,
        detail_y + max(18, int(38 * scale)),
    )
    action_left = max(margin, center_x - action_width // 2)
    action_right = min(width - margin, action_left + action_width)
    draw_glass_panel(
        frame,
        (action_left, action_top),
        (action_right, action_top + action_height),
        accent=action_color,
        alpha=0.62,
        radius=action_height // 2,
    )
    draw_text(
        frame,
        action,
        (center_x, action_top + action_height - max(8, int(15 * scale))),
        max(0.27, 0.48 * scale),
        action_color,
        1,
        align="center",
        max_width=max(35, action_right - action_left - 20),
        min_scale=0.16,
    )
    if progress is not None and not ready:
        _draw_progress_bar(
            frame,
            action_left + max(10, int(18 * scale)),
            action_top + action_height - max(7, int(12 * scale)),
            action_right - max(10, int(18 * scale)),
            action_top + action_height - max(4, int(7 * scale)),
            progress,
        )

    status = (
        f"INPUT {str(privacy_label).upper()}  |  "
        f"HANDS {max(0, _safe_int(hand_count))}/2  |  {str(sound_label).upper()}"
    )
    status_y = min(height - max(20, int(48 * scale)), action_top + action_height + max(21, int(39 * scale)))
    draw_text(
        frame,
        status,
        (center_x, status_y),
        max(0.22, 0.37 * scale),
        MUTED,
        1,
        align="center",
        max_width=max(40, int(width * 0.72)),
        min_scale=0.15,
    )
    controls = "1 Challenge   2 Free play   B Benchmark   D Technical view   Q Quit"
    draw_text(
        frame,
        controls,
        (margin, height - margin),
        max(0.20, 0.34 * scale),
        MUTED,
        1,
        max_width=max(45, int(width * 0.54)),
        min_scale=0.14,
    )
    return frame


def _draw_progress_bar(
    frame,
    left: int,
    top: int,
    right: int,
    bottom: int,
    progress: float,
) -> None:
    """Draw a clamped song progress line."""
    progress = _clamp01(progress)
    cv2.rectangle(frame, (left, top), (right, bottom), PANEL_LIGHT, -1, cv2.LINE_AA)
    filled_right = left + round((right - left) * progress)
    if filled_right > left:
        cv2.rectangle(frame, (left, top), (filled_right, bottom), CYAN, -1, cv2.LINE_AA)
    marker_x = max(left, min(right, filled_right))
    cv2.circle(frame, (marker_x, (top + bottom) // 2), max(2, bottom - top), WHITE, -1, cv2.LINE_AA)


def draw_game_hud(
    frame,
    hands: int,
    score: int,
    combo: int,
    progress: float,
    hits: int,
    misses: int,
    mode_label: str,
    song_label: str,
    privacy_label: str,
    sound_label: str,
    debug_info: Mapping[str, Any] | None = None,
) -> Any:
    """Draw a quiet top rail, leaving most of the stage open for movement."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    margin = max(5, int(14 * scale))
    rail_top = margin
    rail_bottom = min(height - margin - 1, rail_top + max(42, int(66 * scale)))
    draw_glass_panel(
        frame,
        (margin, rail_top),
        (width - margin, rail_bottom),
        accent=CYAN,
        alpha=0.48,
        radius=max(10, int(20 * scale)),
    )

    pad = max(7, int(14 * scale))
    first_line = rail_top + max(13, int(22 * scale))
    second_line = rail_bottom - max(7, int(13 * scale))
    left_right = min(width - margin, margin + max(135, int(width * 0.27)))
    right_left = max(left_right + max(8, int(16 * scale)), width - margin - max(135, int(width * 0.23)))
    center_left = left_right + max(7, int(14 * scale))
    center_right = max(center_left + 1, right_left - max(7, int(14 * scale)))

    draw_text(
        frame,
        "SCORE",
        (margin + pad, first_line),
        max(0.22, 0.34 * scale),
        MUTED,
        1,
    )
    draw_text(
        frame,
        f"{max(0, _safe_int(score)):,}",
        (margin + pad, second_line),
        max(0.37, 0.67 * scale),
        WHITE,
        max(1, int(2 * scale)),
        max_width=max(30, left_right - margin - pad * 2),
        min_scale=0.20,
    )
    draw_text(
        frame,
        f"x{max(0, _safe_int(combo))}",
        (left_right - pad, second_line),
        max(0.28, 0.45 * scale),
        GOLD if _safe_int(combo) else MUTED,
        1,
        align="right",
        max_width=max(25, int(width * 0.09)),
        min_scale=0.16,
    )

    center_x = (center_left + center_right) // 2
    center_width = max(30, center_right - center_left)
    draw_text(
        frame,
        f"{str(mode_label).upper()}  /  {song_label}",
        (center_x, first_line),
        max(0.22, 0.38 * scale),
        WHITE,
        1,
        align="center",
        max_width=center_width,
        min_scale=0.15,
    )
    _draw_progress_bar(
        frame,
        center_left,
        max(first_line + 4, second_line - max(5, int(8 * scale))),
        center_right,
        max(first_line + 7, second_line - max(2, int(4 * scale))),
        progress,
    )

    right_width = max(30, width - margin - right_left - pad)
    draw_text(
        frame,
        f"HANDS {max(0, _safe_int(hands))}/2",
        (right_left, first_line),
        max(0.22, 0.34 * scale),
        GREEN if _safe_int(hands) else MUTED,
        1,
        max_width=right_width,
        min_scale=0.15,
    )
    draw_text(
        frame,
        f"{max(0, _safe_int(hits))} HIT  |  {max(0, _safe_int(misses))} MISS",
        (right_left, second_line),
        max(0.22, 0.36 * scale),
        WHITE,
        1,
        max_width=right_width,
        min_scale=0.15,
    )

    footer_height = max(24, int(36 * scale))
    footer_top = max(rail_bottom + 3, height - margin - footer_height)
    footer_right = min(width - margin, margin + max(155, int(width * 0.54)))
    draw_glass_panel(
        frame,
        (margin, footer_top),
        (footer_right, height - margin),
        accent=None,
        alpha=0.42,
        radius=footer_height // 2,
    )
    footer_text = f"{mode_label} | Input {privacy_label} | {sound_label} | H Help"
    draw_text(
        frame,
        footer_text,
        (margin + pad, height - margin - max(6, int(10 * scale))),
        max(0.19, 0.31 * scale),
        MUTED,
        1,
        max_width=max(50, footer_right - margin - pad * 2),
        min_scale=0.14,
    )

    if debug_info is not None:
        combined_debug = dict(debug_info)
        combined_debug.setdefault("hands", hands)
        draw_debug_overlay(
            frame,
            combined_debug,
            top_offset=rail_bottom + max(4, int(10 * scale)),
        )
    return frame


def draw_countdown(
    frame,
    value: str | int,
    progress: float = 0.0,
    current_time: float = 0.0,
) -> Any:
    """Draw a glowing countdown number or GO message in the centre."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    center = (width // 2, height // 2)
    text = str(value)
    if text.strip() == "":
        return frame
    is_go = text.upper().startswith("GO")
    color = GREEN if is_go else CYAN
    time_value = float(current_time) if isinstance(current_time, (int, float)) else 0.0
    pulse = 0.5 + 0.5 * math.sin(time_value * 7.0) if math.isfinite(time_value) else 0.5
    ring_radius = max(25, int(min(width, height) * (0.11 + pulse * 0.01)))
    draw_glow_circle(frame, center, ring_radius, color, pulse)
    ring_progress = _clamp01(progress)
    if ring_progress > 0:
        cv2.ellipse(
            frame, center, (ring_radius + 8, ring_radius + 8), -90,
            0, 360 * ring_progress, WHITE, max(2, int(4 * scale)), cv2.LINE_AA,
        )
    text_scale = fit_text_scale(
        text, ring_radius * 2 - 12, max(0.9, (1.8 if is_go else 2.4) * scale),
        0.45, max(2, int(4 * scale)),
    )
    text_height = cv2.getTextSize(text, FONT, text_scale, max(2, int(4 * scale)))[0][1]
    draw_text(
        frame, text, (center[0], center[1] + text_height // 2), text_scale,
        color, max(2, int(4 * scale)), align="center",
        max_width=max(15, ring_radius * 2 - 12), min_scale=0.4,
    )
    return frame


def draw_benchmark_status(
    frame,
    summary: Mapping[str, Any] | None = None,
    *,
    notice: str | None = None,
) -> Any:
    """Draw a compact recording badge without exposing camera information."""
    info = dict(summary or {})
    active = bool(info.get("active"))
    if not active and not notice:
        return frame

    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    margin = max(5, int(14 * scale))
    panel_width = min(
        width - margin * 2,
        max(160, int(300 * scale)),
    )
    panel_height = max(28, int((78 if active else 42) * scale))
    top = max(margin, int(height * 0.12))
    bottom = min(height - margin, top + panel_height)
    right = min(width - margin, margin + panel_width)
    accent = RED if active else GREEN
    draw_glass_panel(
        frame,
        (margin, top),
        (right, bottom),
        accent=accent,
        alpha=0.68,
        radius=max(8, int(18 * scale)),
    )

    text_left = margin + max(9, int(16 * scale))
    text_width = max(30, right - text_left - max(7, int(12 * scale)))
    first_y = top + max(14, int(24 * scale))
    if not active:
        draw_text(
            frame,
            str(notice),
            (text_left, first_y),
            max(0.23, 0.39 * scale),
            accent,
            1,
            max_width=text_width,
            min_scale=0.15,
        )
        return frame

    elapsed = max(0.0, float(info.get("elapsed_seconds", 0.0) or 0.0))
    frames = max(0, _safe_int(info.get("frames", 0)))
    draw_text(
        frame,
        f"BENCHMARK REC  {elapsed:05.1f}s  |  {frames} FRAMES",
        (text_left, first_y),
        max(0.22, 0.38 * scale),
        RED,
        1,
        max_width=text_width,
        min_scale=0.14,
    )
    average_fps = info.get("average_fps")
    fps_text = (
        "--"
        if not isinstance(average_fps, (int, float)) or not math.isfinite(average_fps)
        else f"{average_fps:.1f}"
    )
    second_y = min(bottom - 5, first_y + max(13, int(22 * scale)))
    draw_text(
        frame,
        f"AVG {fps_text} FPS  |  SLOW {max(0, _safe_int(info.get('slow_frames')))}  |  "
        f"DROP~ {max(0, _safe_int(info.get('estimated_dropped_frames')))}  |  "
        f"AUDIO {max(0, _safe_int(info.get('audio_samples')))}",
        (text_left, second_y),
        max(0.18, 0.31 * scale),
        MUTED,
        1,
        max_width=text_width,
        min_scale=0.12,
    )
    return frame


def draw_results(
    frame,
    *,
    score: int,
    completion: float,
    rank: str,
    perfect: int,
    great: int,
    good: int,
    misses: int,
    max_combo: int,
    basic_hits: int = 0,
    song_label: str = "Song challenge",
    replay_hint: str = "Press R to replay",
) -> Any:
    """Draw a compact end-of-song summary without covering the stage."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    _draw_background_tint(frame, 0.22)
    margin = max(6, int(18 * scale))
    panel_left = max(margin, int(width * 0.10))
    panel_right_ratio = 0.64 if width >= 800 else 0.88
    panel_right = min(width - margin, int(width * panel_right_ratio))
    panel_top = max(margin, int(height * 0.16))
    panel_bottom = min(height - margin, int(height * 0.82))
    draw_glass_panel(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        accent=GREEN,
        alpha=0.63,
        radius=max(12, int(26 * scale)),
    )

    center_x = (panel_left + panel_right) // 2
    available = max(40, panel_right - panel_left - max(22, int(48 * scale)))
    title_y = panel_top + max(21, int(39 * scale))
    draw_text(
        frame,
        "PERFORMANCE COMPLETE",
        (center_x, title_y),
        max(0.31, 0.58 * scale),
        CYAN,
        max(1, int(2 * scale)),
        align="center",
        max_width=available,
        min_scale=0.19,
    )
    song_y = title_y + max(18, int(31 * scale))
    draw_text(
        frame,
        song_label,
        (center_x, song_y),
        max(0.23, 0.38 * scale),
        MUTED,
        1,
        align="center",
        max_width=available,
        min_scale=0.15,
    )

    completion_value = 0.0
    try:
        completion_value = float(completion)
    except (TypeError, ValueError):
        pass
    if not math.isfinite(completion_value):
        completion_value = 0.0
    completion_value = max(0.0, min(100.0, completion_value))
    rank_text = str(rank).upper()[:3] or "-"
    rank_y = song_y + max(39, int(84 * scale))
    draw_text(
        frame,
        rank_text,
        (center_x, rank_y),
        fit_text_scale(rank_text, max(35, int(available * 0.30)), max(1.2, 2.55 * scale), 0.7, max(3, int(5 * scale))),
        GREEN,
        max(3, int(5 * scale)),
        align="center",
        max_width=available,
    )
    draw_text(
        frame,
        "FINAL RANK",
        (center_x, rank_y + max(17, int(28 * scale))),
        max(0.21, 0.32 * scale),
        MUTED,
        1,
        align="center",
        max_width=available,
        min_scale=0.14,
    )

    summary_y = rank_y + max(47, int(83 * scale))
    draw_text(
        frame,
        f"{max(0, _safe_int(score)):,}",
        (center_x, summary_y),
        max(0.46, 0.82 * scale),
        WHITE,
        max(1, int(2 * scale)),
        align="center",
        max_width=available,
        min_scale=0.24,
    )
    draw_text(
        frame,
        f"SCORE  |  COMPLETION {completion_value:.1f}%  |  BEST COMBO x{max(0, _safe_int(max_combo))}",
        (center_x, summary_y + max(18, int(32 * scale))),
        max(0.20, 0.33 * scale),
        MUTED,
        1,
        align="center",
        max_width=available,
        min_scale=0.14,
    )

    stats_y = summary_y + max(43, int(74 * scale))
    stats = (
        f"PERFECT {max(0, _safe_int(perfect))}   |   "
        f"GREAT {max(0, _safe_int(great))}   |   "
        f"GOOD {max(0, _safe_int(good))}   |   "
        f"HIT {max(0, _safe_int(basic_hits))}   |   "
        f"MISS {max(0, _safe_int(misses))}"
    )
    draw_text(
        frame,
        stats,
        (center_x, stats_y),
        max(0.20, 0.34 * scale),
        GOLD,
        1,
        align="center",
        max_width=available,
        min_scale=0.14,
    )
    meter_left = panel_left + max(18, int(42 * scale))
    meter_right = panel_right - max(18, int(42 * scale))
    meter_top = min(panel_bottom - max(27, int(52 * scale)), stats_y + max(9, int(17 * scale)))
    _draw_progress_bar(
        frame,
        meter_left,
        meter_top,
        meter_right,
        meter_top + max(3, int(5 * scale)),
        completion_value / 100.0,
    )
    draw_text(
        frame,
        replay_hint,
        (center_x, panel_bottom - max(9, int(18 * scale))),
        max(0.21, 0.36 * scale),
        GREEN,
        1,
        align="center",
        max_width=available,
        min_scale=0.15,
    )
    return frame


def draw_help_overlay(
    frame,
    controls: Sequence[tuple[str, str]] = DEFAULT_CONTROLS,
) -> Any:
    """Draw a compact control guide that pauses the active round safely."""
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    _draw_background_tint(frame, 0.57)
    left = max(5, int(width * 0.19))
    right = min(width - 6, int(width * 0.81))
    top = max(5, int(height * 0.15))
    bottom = min(height - 6, int(height * 0.85))
    draw_glass_panel(
        frame,
        (left, top),
        (right, bottom),
        accent=CYAN,
        alpha=0.78,
        radius=max(12, int(26 * scale)),
    )
    center_x = width // 2
    available = max(25, right - left - max(20, int(50 * scale)))
    heading_y = top + max(20, int(40 * scale))
    draw_text(
        frame, "QUICK GUIDE", (center_x, heading_y), max(0.37, 0.66 * scale),
        CYAN, max(1, int(2 * scale)), align="center", max_width=available, min_scale=0.22,
    )
    guide_y = heading_y + max(18, int(32 * scale))
    draw_text(
        frame, "Move a fingertip through a falling circle to play its note.",
        (center_x, guide_y), max(0.29, 0.48 * scale), WHITE, 1,
        align="center", max_width=available, min_scale=0.18,
    )

    rows = list(controls)[:9]
    if rows:
        content_top = guide_y + max(18, int(48 * scale))
        content_bottom = bottom - max(21, int(42 * scale))
        use_two_columns = right - left >= 500 and bottom - top >= 350
        column_count = 2 if use_two_columns else 1
        rows_per_column = math.ceil(len(rows) / column_count)
        inner_left = left + max(12, int(40 * scale))
        inner_right = right - max(12, int(40 * scale))
        column_gap = max(8, int(24 * scale)) if use_two_columns else 0
        column_width = max(
            1,
            (inner_right - inner_left - column_gap * (column_count - 1))
            // column_count,
        )
        row_gap = max(18, min(max(23, int(58 * scale)), max(18, (content_bottom - content_top) // rows_per_column)))
        row_scale = max(0.24, 0.40 * scale)
        for index, item in enumerate(rows):
            try:
                key, action = item
            except (TypeError, ValueError):
                continue
            column = index // rows_per_column
            row = index % rows_per_column
            column_left = inner_left + column * (column_width + column_gap)
            column_right = min(inner_right, column_left + column_width)
            y = min(content_bottom, content_top + row * row_gap)
            key_x = column_left + max(7, int(13 * scale))
            action_x = column_left + max(58, int(100 * scale))
            draw_text(
                frame, str(key), (key_x, y), row_scale, GOLD, 1,
                max_width=max(25, action_x - key_x - 8), min_scale=0.14,
            )
            draw_text(
                frame, str(action), (action_x, y), row_scale, WHITE, 1,
                max_width=max(30, column_right - action_x - 10), min_scale=0.16,
            )
            if row < rows_per_column - 1:
                cv2.line(
                    frame,
                    (column_left, min(content_bottom, y + max(7, int(13 * scale)))),
                    (column_right, min(content_bottom, y + max(7, int(13 * scale)))),
                    PANEL_LIGHT,
                    1,
                    cv2.LINE_AA,
                )
    draw_text(
        frame, "H  RESUME PERFORMANCE", (center_x, bottom - max(7, int(18 * scale))),
        max(0.27, 0.44 * scale), GREEN, 1, align="center",
        max_width=available, min_scale=0.17,
    )
    return frame


def _format_confidence(value: Any) -> str | None:
    """Format a model confidence value when one is available."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(number)
        if not numbers:
            return None
        value = sum(numbers) / len(numbers)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    percentage = number * 100.0 if 0.0 <= number <= 1.0 else number
    return f"{max(0.0, min(100.0, percentage)):.0f}%"


def draw_debug_overlay(
    frame,
    debug_info: Mapping[str, Any] | None = None,
    *,
    top_offset: int | None = None,
) -> Any:
    """Show truthful computer-vision details for a portfolio recording.

    MediaPipe supplies a pretrained hand-landmark model.  OpenCV handles the
    camera image and rendering.  Air Rhythm's own rules measure landmark
    motion and check a fingertip path against each circle.
    """
    info = dict(debug_info or {})
    width, height = _frame_size(frame)
    scale = _ui_scale(frame)
    margin = max(5, int(14 * scale))
    panel_width = min(width - margin * 2, max(145, int(280 * scale)))
    left = width - margin - panel_width
    top = max(margin, int(top_offset) if top_offset is not None else margin)
    line_gap = max(12, int(26 * scale))
    confidence = _format_confidence(info.get("confidence"))
    detected_hands = max(
        0, _safe_int(info.get("hands", info.get("hand_count", 0)))
    )
    landmark_count = max(
        0,
        _safe_int(
            info.get(
                "landmarks",
                info.get("landmark_count", detected_hands * 21),
            )
        ),
    )
    lines = [
        ("CV / ML PIPELINE", CYAN),
        (f"FPS  {info.get('fps', '--')}", WHITE),
        (f"Hands  {detected_hands}/2", WHITE),
        (f"Landmarks  {landmark_count}", WHITE),
    ]
    if confidence is not None:
        lines.append((f"Mean hand-class confidence  {confidence}", GREEN))
    if info.get("capture_ms") is not None:
        lines.append((f"Background camera read  {info['capture_ms']} ms", WHITE))
    if info.get("camera_wait_ms") is not None:
        lines.append((f"Main-loop camera wait  {info['camera_wait_ms']} ms", WHITE))
    if info.get("preprocessing_ms") is not None:
        lines.append((f"Camera preparation  {info['preprocessing_ms']} ms", WHITE))
    if info.get("inference_ms") is not None:
        lines.append((f"Inference  {info['inference_ms']} ms", WHITE))
    if info.get("update_ms") is not None:
        lines.append((f"Game update  {info['update_ms']} ms", WHITE))
    if info.get("render_ms") is not None:
        lines.append((f"Rendering  {info['render_ms']} ms", WHITE))
    if info.get("frame_ms") is not None:
        lines.append((f"Frame pipeline  {info['frame_ms']} ms", WHITE))
    if info.get("gesture"):
        lines.append((f"Motion rule  {info['gesture']}", GOLD))
    lines.extend(
        (
            ("MediaPipe: pretrained landmark model", MUTED),
            ("OpenCV: camera capture + rendering", MUTED),
            ("App logic: fingertip path collision", MUTED),
        )
    )
    panel_height = min(
        height - top - margin,
        max(30, max(37, int(24 * scale)) + line_gap * len(lines)),
    )
    if panel_height <= 3:
        return frame
    bottom = top + panel_height
    draw_glass_panel(frame, (left, top), (width - margin, bottom), accent=CYAN, alpha=0.72)
    text_x = left + max(8, int(14 * scale))
    available = max(20, width - margin - text_x - max(6, int(10 * scale)))
    first_y = top + max(13, int(22 * scale))
    for index, (line, color) in enumerate(lines):
        y = first_y + index * line_gap
        if y > bottom - 5:
            break
        draw_text(
            frame, line, (text_x, y), max(0.24, 0.42 * scale), color, 1,
            max_width=available, min_scale=0.15,
        )
    return frame


__all__ = [
    "BLUE",
    "CYAN",
    "DEFAULT_CONTROLS",
    "FONT",
    "GOLD",
    "GREEN",
    "INK",
    "MAGENTA",
    "MUTED",
    "PANEL",
    "PANEL_LIGHT",
    "RED",
    "TEXT_SHADOW",
    "WHITE",
    "draw_brand_badge",
    "draw_benchmark_status",
    "draw_countdown",
    "draw_debug_overlay",
    "draw_game_hud",
    "draw_glass_panel",
    "draw_glow_circle",
    "draw_help_overlay",
    "draw_results",
    "draw_text",
    "draw_title_screen",
    "fit_text_scale",
]
