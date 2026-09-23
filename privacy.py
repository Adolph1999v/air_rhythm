"""Privacy-safe live-input rendering for Air Rhythm.

The hands-only view reveals real camera pixels only inside masks inferred from the
21 hand landmarks.  Pixels outside those masks are replaced with an opaque stage.
If a hand is held over a face or another private object, pixels inside the hand
mask can still include that object because landmarks do not provide full image
segmentation.
"""

from collections import OrderedDict
from enum import Enum
import math

import cv2
import numpy as np


class PrivacyMode(Enum):
    """The three camera visibility levels available to the player."""

    HANDS_ONLY = "hands_only"
    SKELETON_ONLY = "skeleton_only"
    CAMERA = "camera"


PRIVACY_MODE_ORDER = (
    PrivacyMode.HANDS_ONLY,
    PrivacyMode.SKELETON_ONLY,
    PrivacyMode.CAMERA,
)

# The palm is filled as one solid region.  Fingers are added as thick paths so
# spaces between spread fingers remain private.
PALM_INDICES = (0, 1, 2, 5, 9, 13, 17)
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
LANDMARK_COUNT = 21
MAX_CACHED_BACKGROUNDS = 3


def next_privacy_mode(mode: PrivacyMode) -> PrivacyMode:
    """Return the next mode in the hands, skeleton, camera cycle."""
    try:
        position = PRIVACY_MODE_ORDER.index(mode)
    except ValueError as error:
        raise ValueError(f"Unknown privacy mode: {mode!r}") from error
    return PRIVACY_MODE_ORDER[(position + 1) % len(PRIVACY_MODE_ORDER)]


class PrivacyRenderer:
    """Replace private camera areas before the live-input inset is drawn.

    ``apply`` always returns a new array with the same shape and dtype as the
    input frame.  In ``CAMERA`` mode that array is an unchanged copy.  Drawing
    the MediaPipe skeleton after this method keeps its lines bright in every
    mode.
    """

    def __init__(self, mode: PrivacyMode = PrivacyMode.HANDS_ONLY):
        if not isinstance(mode, PrivacyMode):
            raise TypeError("mode must be a PrivacyMode")
        self.mode = mode
        self._background_cache = OrderedDict()

    def cycle(self) -> PrivacyMode:
        """Select and return the next privacy mode."""
        self.mode = next_privacy_mode(self.mode)
        return self.mode

    def apply(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """Render ``frame`` according to the active privacy mode.

        ``hand_landmarks`` is an iterable of hands, where each hand contains the
        usual 21 MediaPipe-like objects with normalized ``x`` and ``y`` fields.
        The module deliberately does not import MediaPipe, which also makes the
        renderer easy to test independently.

        A landmark mask cannot distinguish a hand from something directly behind
        it.  A hand held over a face may therefore reveal a few face pixels inside
        the hand-shaped cutout; everything outside the cutout is fully replaced.
        """
        self._validate_frame(frame)

        if self.mode is PrivacyMode.CAMERA:
            return frame.copy()

        stage = self._stage_for(frame)
        if self.mode is PrivacyMode.SKELETON_ONLY:
            return stage.copy()

        mask = self._build_hand_mask(frame.shape[:2], hand_landmarks)
        if not np.any(mask):
            return stage.copy()

        alpha = self._inside_feather(mask)
        return self._blend(frame, stage, alpha)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.ndim not in (2, 3) or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("frame must be a non-empty image")
        if frame.ndim == 3 and frame.shape[2] == 0:
            raise ValueError("frame must contain at least one channel")
        if not (np.issubdtype(frame.dtype, np.integer) or np.issubdtype(frame.dtype, np.floating)):
            raise TypeError("frame must use an integer or floating-point dtype")

    def _stage_for(self, frame: np.ndarray) -> np.ndarray:
        key = (frame.shape, frame.dtype.str)
        cached = self._background_cache.get(key)
        if cached is not None:
            self._background_cache.move_to_end(key)
            return cached

        stage = self._create_stage(frame.shape, frame.dtype)
        stage.setflags(write=False)
        self._background_cache[key] = stage
        if len(self._background_cache) > MAX_CACHED_BACKGROUNDS:
            self._background_cache.popitem(last=False)
        return stage

    @staticmethod
    def _create_stage(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create a low-contrast stage that keeps bright game graphics legible."""
        height, width = shape[:2]
        channels = 1 if len(shape) == 2 else shape[2]
        y_mix = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

        if channels == 1:
            top = np.array([0.035], dtype=np.float32)
            bottom = np.array([0.075], dtype=np.float32)
        else:
            # OpenCV uses BGR.  Extra channels receive an opaque neutral value.
            top = np.full(channels, 0.06, dtype=np.float32)
            bottom = np.full(channels, 0.09, dtype=np.float32)
            top[:3] = (0.075, 0.045, 0.035)
            bottom[:3] = (0.12, 0.065, 0.07)
            if channels >= 4:
                top[3] = bottom[3] = 1.0

        stage = (
            top[None, None, :] * (1.0 - y_mix[:, :, None])
            + bottom[None, None, :] * y_mix[:, :, None]
        )
        stage = np.broadcast_to(stage, (height, width, channels)).copy()

        # A sparse grid gives the hidden-camera view some depth without competing
        # with falling circles or the hand skeleton.
        grid_spacing = max(48, min(height, width) // 9)
        grid_value = np.minimum(stage + 0.012, 1.0)
        stage[::grid_spacing, :, :] = grid_value[::grid_spacing, :, :]
        stage[:, ::grid_spacing, :] = grid_value[:, ::grid_spacing, :]

        if np.issubdtype(dtype, np.integer):
            maximum = np.iinfo(dtype).max
            stage = np.rint(stage * maximum).astype(dtype)
        else:
            stage = stage.astype(dtype)

        return stage[:, :, 0] if len(shape) == 2 else stage

    @classmethod
    def _build_hand_mask(cls, image_shape, hands) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if hands is None:
            return mask

        for hand in hands:
            points = cls._hand_points(hand, width, height)
            if points is None:
                continue
            cls._add_hand_to_mask(mask, points)
        return mask

    @staticmethod
    def _hand_points(hand, width: int, height: int) -> np.ndarray | None:
        try:
            landmarks = list(hand)
        except TypeError:
            return None
        if len(landmarks) < LANDMARK_COUNT:
            return None

        points = []
        for landmark in landmarks[:LANDMARK_COUNT]:
            try:
                x = float(landmark.x)
                y = float(landmark.y)
            except (AttributeError, TypeError, ValueError):
                return None
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            pixel_x = int(round(np.clip(x, 0.0, 1.0) * (width - 1)))
            pixel_y = int(round(np.clip(y, 0.0, 1.0) * (height - 1)))
            points.append((pixel_x, pixel_y))
        return np.asarray(points, dtype=np.int32)

    @staticmethod
    def _add_hand_to_mask(mask: np.ndarray, points: np.ndarray) -> None:
        height, width = mask.shape
        palm_width = float(np.linalg.norm(points[5] - points[17]))
        palm_length = float(np.linalg.norm(points[0] - points[9]))
        hand_scale = max(palm_width, palm_length, 1.0)

        # A minimum width helps distant hands remain visible.  The upper bound
        # prevents one uncertain landmark from exposing a large part of a frame.
        max_thickness = max(6, int(round(min(height, width) * 0.09)))
        finger_thickness = int(
            np.clip(round(hand_scale * 0.30), 7, max_thickness)
        )

        palm = cv2.convexHull(points[list(PALM_INDICES)])
        cv2.fillConvexPoly(mask, palm, 255, cv2.LINE_AA)

        for chain in FINGER_CHAINS:
            finger = points[list(chain)].reshape((-1, 1, 2))
            cv2.polylines(
                mask,
                [finger],
                False,
                255,
                finger_thickness,
                cv2.LINE_AA,
            )
            for point in finger[:, 0, :]:
                cv2.circle(
                    mask,
                    tuple(point),
                    finger_thickness // 2,
                    255,
                    -1,
                    cv2.LINE_AA,
                )

        wrist_radius = int(
            np.clip(round(palm_width * 0.27), finger_thickness // 2, max_thickness)
        )
        cv2.circle(
            mask,
            tuple(points[0]),
            wrist_radius,
            255,
            -1,
            cv2.LINE_AA,
        )

        dilation_radius = int(np.clip(round(hand_scale * 0.055), 2, 8))
        kernel_size = dilation_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        # Threshold first so anti-aliased drawing cannot produce a faint camera
        # leak beyond the explicit, dilated silhouette.
        binary_hand = np.where(mask > 0, 255, 0).astype(np.uint8)
        dilated = cv2.dilate(binary_hand, kernel)
        np.maximum(mask, dilated, out=mask)

    @staticmethod
    def _inside_feather(mask: np.ndarray) -> np.ndarray:
        """Soften inward from the mask edge while keeping its exterior at zero."""
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        distance_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        visible_area = int(np.count_nonzero(binary_mask))
        approximate_scale = math.sqrt(max(visible_area, 1))
        feather_width = float(np.clip(round(approximate_scale * 0.025), 2, 8))
        return np.clip(distance_inside / feather_width, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _blend(frame: np.ndarray, stage: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        expanded_alpha = alpha if frame.ndim == 2 else alpha[:, :, None]
        result = (
            frame.astype(np.float32) * expanded_alpha
            + stage.astype(np.float32) * (1.0 - expanded_alpha)
        )
        if np.issubdtype(frame.dtype, np.integer):
            limits = np.iinfo(frame.dtype)
            result = np.clip(np.rint(result), limits.min, limits.max)
        return result.astype(frame.dtype)
