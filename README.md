# Air Rhythm

Air Rhythm is a camera-controlled rhythm game built with Python, OpenCV, and MediaPipe. Notes will travel toward a hit zone while the player uses timed hand gestures to play them.

The project is being developed phase by phase as a practical computer-vision learning project.

## Project status

Phases 1 and 2 are complete. The application currently opens a mirrored webcam preview, detects up to two hands, and draws each hand's 21 landmarks, skeleton lines, and highlighted fingertips in real time.

### Current capabilities

- Safe webcam startup and shutdown
- Mirrored camera preview for natural hand movement
- MediaPipe hand tracking in video mode
- Two-hand landmark skeletons and fingertip highlights
- Live detected-hand counter

### Next milestone

Phase 3 will add the first rhythm-game interaction:

- Four note lanes
- A visible hit zone
- A static practice target
- Downward index-finger tap detection
- Clear hit and miss feedback

## How the current foundation works

```text
Webcam frame
    -> OpenCV mirrors the frame
    -> the frame is converted from BGR to RGB
    -> MediaPipe detects hand landmarks
    -> normalized landmarks are converted to pixels
    -> OpenCV draws the result
```

## Run locally

Create and prepare the project-specific virtual environment:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
```

Run Air Rhythm:

```bash
.venv/bin/python app.py
```

Press `Q` while the camera window is selected to quit.

## Project structure

```text
app.py                       Main camera and hand-tracking application
models/hand_landmarker.task  Local MediaPipe hand model
requirements.txt             Python dependencies
```

## Roadmap

- [x] Phase 1 — Camera foundation
- [x] Phase 2 — Two-hand landmark tracking
- [ ] Phase 3 — Note lanes, hit zone, and gesture detection
- [ ] Phase 4 — Falling notes, timing windows, scoring, and sound
- [ ] Phase 5 — Music charts, calibration, visual polish, and release
- [ ] Future — AI-generated rhythm challenges
