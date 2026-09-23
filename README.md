# Air Rhythm — Computer Vision Music Interface

Air Rhythm lets you play music by moving your hands in front of a webcam. A live hand-tracking model locates your fingertips; my computer-vision and timing code turns their movement into hits on falling notes. The main screen is a generated performance stage, while a small camera inset shows the real hand skeleton driving it.

## Computer vision in action

```text
Webcam → OpenCV frame processing → MediaPipe hand landmarks
       → hand stabilisation and fingertip paths → note collision → timed sound
```

- **Two-hand tracking:** the pretrained model finds 21 landmarks on each hand. The live inset draws their skeletons so viewers can see the vision input behind each virtual drumstick.
- **Motion-aware interaction:** custom filtering reduces landmark jitter, keeps left and right hand identities stable, and checks fingertip paths between frames so quick movements can still register.
- **Real-time pipeline:** a background camera reader supplies the newest frame while inference and rendering continue. A separate beat clock keeps notes moving at a steady speed across varying frame rates.
- **Visible, private stage:** the main performance view is drawn from scratch without copying camera pixels; only the small inset shows live camera evidence.

## Tech stack and ML model

| Technology | Role |
| --- | --- |
| Python + OpenCV | Camera input, image processing, rendering, and the live interface |
| **MediaPipe Hand Landmarker** (`models/hand_landmarker.task`) | **Pretrained machine-learning model** estimating 21 landmarks for each of up to two hands in video mode |
| NumPy + sounddevice | Generated tones and output-only audio playback |

MediaPipe supplies the hand model; I did **not** train it. The camera pipeline, landmark filtering, fingertip collision, rhythm logic, audio engine, and interface are my application code. Training and evaluating an original gesture classifier is a future milestone.

## How the music works

The challenge encodes a simplified 35-note opening of Beethoven's *Für Elise*, based on a [public-domain score](https://www.mutopiaproject.org/ftp/BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59-let.pdf). Each falling circle carries a scheduled pitch; fingertip contact plays that pitch through the app's own audio synthesizer. There is no bundled song recording. Free play maps four coloured circles to complementary pitches.

## Measured performance

![Bar chart comparing four live-camera runs: 15.03 FPS at 1080p with serial capture, 16.04 and 16.59 FPS at 720p with serial capture, and 30.39 FPS at 720p with background capture.](docs/performance_progress.svg)

On the test Mac, average FPS rose from **15.03 to 30.39** and average frame processing fell from **50.41 ms to 20.12 ms**. The final 64.67-second run recorded **zero estimated dropped frames**. Resolution changed from 1080p to 720p during this work; moving camera reads to a background worker produced the largest improvement. These are measured results for this setup, not a guaranteed rate on every computer. [Read the full performance study](docs/PERFORMANCE_STUDY.md).

## Explore the project

| Guide | What it covers |
| --- | --- |
| [Getting Started](docs/GETTING_STARTED.md) | Download ZIP or clone, install, run, and troubleshoot |
| [Play Guide](docs/PLAY_GUIDE.md) | Controls, challenge rules, scoring, music, and privacy views |
| [Technical Overview](docs/TECHNICAL_OVERVIEW.md) | Model boundaries, camera-to-audio pipeline, and code structure |
| [Performance Study](docs/PERFORMANCE_STUDY.md) | Benchmark method, before-and-after data, and limitations |
| [Future Plan](FUTURE_PLAN.md) | Demo, robustness tests, and future ML work |

The live prototype and performance benchmark are complete. Final demo recording and broader tracking checks are still in progress.
