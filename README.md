# AirBeat

AirBeat is a hand-tracked virtual drum studio built with Python and OpenCV.

## Current milestone

**Phase 2 — Hand tracking**: AirBeat opens a mirrored webcam preview, detects up to two hands, and draws each hand's 21 landmarks, skeleton lines, and highlighted fingertips in real time.

## Current capabilities

- Safe webcam startup and shutdown
- Mirrored camera preview for natural hand movement
- MediaPipe hand tracking in video mode
- Two-hand landmark skeletons and fingertip highlights
- Live detected-hand counter

## Run it locally

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py
```

Press `Q` while the camera window is selected to quit.
