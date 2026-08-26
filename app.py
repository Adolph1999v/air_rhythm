"""Phase 2 of Air Rhythm: a mirrored camera preview with hand landmarks."""

from pathlib import Path
import time

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
WINDOW_TITLE = "Air Rhythm | Phase 2 - Hand Tracking"
MAX_HANDS = 2
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

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


def draw_hand_skeleton(frame, hand_landmarks) -> None:
    """Draw one hand's landmark points, connecting lines, and fingertips."""
    frame_height, frame_width = frame.shape[:2]
    pixel_points = []

    for landmark in hand_landmarks:
        pixel_x = max(0, min(int(landmark.x * frame_width), frame_width - 1))
        pixel_y = max(0, min(int(landmark.y * frame_height), frame_height - 1))
        pixel_points.append((pixel_x, pixel_y))

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


def draw_status_overlay(frame, hand_count: int) -> None:
    """Show how many hands MediaPipe found in the current frame."""
    status_text = f"Hands detected: {hand_count}/{MAX_HANDS}"
    cv2.rectangle(frame, (12, 12), (260, 52), (25, 25, 25), -1)
    cv2.putText(
        frame,
        status_text,
        (22, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    """Open the camera, track hands, display landmarks, and clean up safely."""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check that it is connected and that "
            "your terminal or editor has Camera permission in macOS Settings."
        )

    try:
        with create_hand_landmarker() as hand_landmarker:
            previous_timestamp_ms = -1

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

                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_skeleton(mirrored_frame, hand_landmarks)

                draw_status_overlay(
                    mirrored_frame,
                    len(detection_result.hand_landmarks),
                )
                cv2.imshow(WINDOW_TITLE, mirrored_frame)

                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
