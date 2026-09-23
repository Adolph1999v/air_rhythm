# Air Rhythm Performance Study

## Purpose

This study explains how Air Rhythm's real-time computer-vision pipeline was measured, where its original bottleneck was, what changed, and how those changes affected responsiveness. The goal is reproducible engineering evidence rather than an unsupported claim that the application "feels faster."

The benchmark stores timing numbers, environment details, and counters only. It does not save camera images or hand-landmark coordinates.

## Result at a glance

The final 64.67-second run processed 1,940 frames at the 30 FPS camera target. It reached 30.39 reported average FPS with zero estimated dropped frames. The original 77.20-second baseline reached 15.03 average FPS and estimated 593 dropped frames.

Compared with the baseline:

- Reported average FPS increased by 102.20%.
- Average complete-frame processing time fell from 50.41 ms to 20.12 ms, a 60.08% reduction.
- The 95th-percentile complete-frame time fell from 53.42 ms to 31.28 ms, placing 95% of measured processing samples below the 33.33 ms target-frame budget.
- Average rendering time fell by 28.11%.
- Average MediaPipe inference time fell by 18.13%.
- Average frame-start-to-audio-request time on hit frames fell from 34.96 ms to 9.48 ms, a 72.89% reduction.

These figures show that moving camera reads off the main loop produced the decisive improvement. Reducing the working resolution helped rendering, but did not remove the serial camera wait by itself.

## What was measured

One displayed frame travels through several stages:

1. The camera produces a frame.
2. OpenCV mirrors and prepares it.
3. MediaPipe estimates 21 landmarks for each detected hand.
4. Adaptive filtering stabilises the landmarks and preserves hand identity.
5. The game updates notes, collisions, timing, and score.
6. OpenCV draws the generated stage, virtual drumsticks, interface, and live-input inset.
7. A hit requests an already-prepared sound from the audio engine.

The benchmark records:

- frame intervals and derived FPS;
- background camera-read time;
- main-loop wait for a fresh camera frame;
- camera-preparation time;
- MediaPipe inference time;
- game-update time;
- rendering and display-submission time;
- complete main-loop frame time; and
- frame-start-to-audio-request time for hit frames.

Background camera-read time overlaps main-loop work. It must not be added to complete-frame time. The audio measurement ends when software requests a sound; it does not measure physical speaker output, operating-system audio buffering, or Bluetooth delay.

## Test environment and protocol

All comparison runs used the same environment:

| Setting | Value |
| --- | --- |
| Operating system | Darwin 27.0.0 |
| Machine architecture | arm64 |
| Python | 3.14.6 |
| Target rate | 30 FPS |
| Game mode | Challenge |
| Input view | Camera |
| Sound | On |
| Maximum hands | 2 |
| Landmark model | MediaPipe Hand Landmarker |

The baseline used 1920x1080 camera input and a 1920x1200 stage. Later runs requested, captured, and processed 1280x720 camera frames with a 1280x800 stage. The final validation run lasted longer than 60 seconds and included 70 hit-frame audio samples.

The reported average FPS is the mean of per-frame FPS values derived from frame intervals. It can differ slightly from frame count divided by total session duration. The final run's direct throughput was 1,940 / 64.67, or approximately 30.0 frames per second.

## Architecture versions

### 1. Full-resolution serial pipeline

The first version read a 1920x1080 camera frame inside the same loop that performed inference, game logic, and rendering. Every frame therefore paid the camera wait before other work could continue.

### 2. Resolution-bounded serial pipeline

The second version requested 1280x720 at 30 FPS and prevented oversized frames from entering inference and rendering. This reduced rendering work, but the camera read still blocked the main loop. Two runs were retained to show that the result was repeatable rather than based on one favourable sample.

### 3. Background latest-frame pipeline

The final version reads camera frames continuously on a background worker. The main loop consumes only the newest complete frame, so it does not build a queue of stale input. Camera waiting can overlap inference, game logic, and rendering.

This version also includes adaptive landmark smoothing, persistent Left/Right hand identities, and a 140 ms visual-only dropout grace period. A stale hand pose may remain visible briefly to avoid flicker, but it is immediately excluded from collision detection.

## Benchmark comparison

| Run | Duration | Frames | Camera / stage | Average FPS | Minimum FPS | Slow frames | Estimated dropped frames |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Full-HD serial baseline | 77.20 s | 1,159 | 1920x1080 / 1920x1200 | 15.03 | 13.17 | 100.00% | 593 |
| 720p serial A | 58.48 s | 908 | 1280x720 / 1280x800 | 16.04 | 13.52 | 96.48% | 448 |
| 720p serial B | 56.82 s | 901 | 1280x720 / 1280x800 | 16.59 | 13.78 | 95.56% | 429 |
| 720p background final | 64.67 s | 1,940 | 1280x720 / 1280x800 | 30.39 | 20.56 | 30.21% | 0 |

| Run | Complete average | Complete P95 | Inference average | Rendering average | Hit-to-audio request average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-HD serial baseline | 50.41 ms | 53.42 ms | 8.01 ms | 14.67 ms | 34.96 ms |
| 720p serial A | 50.94 ms | 55.49 ms | 8.41 ms | 12.67 ms | 39.05 ms |
| 720p serial B | 49.64 ms | 56.24 ms | 7.87 ms | 12.47 ms | 38.09 ms |
| 720p background final | 20.12 ms | 31.28 ms | 6.56 ms | 10.55 ms | 9.48 ms |

The short 2.84-second background-capture check performed before the final run is intentionally excluded from the comparison tables. It confirmed that the new instrumentation worked, but it was too short to serve as the final performance result.

## Reading the final result correctly

The final background camera read averaged 33.32 ms, which matches a camera producing approximately 30 frames per second. That work happened concurrently with the main loop. The main loop waited only 2.57 ms on average for a fresh frame, with a median wait of 0.11 ms.

The final complete-frame pipeline averaged 20.12 ms, leaving 13.21 ms of average processing headroom inside a 33.33 ms frame budget. Its 95th-percentile time was 31.28 ms. No camera frames were skipped for freshness, which indicates that processing kept pace with this camera during the run.

The final report still marks 30.21% of intervals as "slow" because any interval even slightly above 33.33 ms receives that label. A slow interval is not automatically a dropped frame. The dropped-frame estimate increases only when an interval spans enough time to miss a complete additional 30 FPS frame slot. The final run estimated zero such drops.

## Hand-stability work

Virtual drumsticks amplify small landmark-angle changes because a long rendered stick extends beyond the detected fingertip. The stabilisation layer addresses this in four ways:

- a movement-adaptive low-pass filter suppresses small stationary changes;
- deliberate planar and forward movement automatically receives less smoothing;
- stable hand identities stop detector list-order changes from swapping stick history and colour; and
- short visual dropout grace prevents flicker without allowing stale poses to trigger hits.

Automated tests cover stationary jitter reduction, stick-angle jitter, fast deliberate movement, forward depth movement, crossed hands, reversed detector order, invalid landmarks, and temporary detection loss. Manual testing found the movement smoother after this change. Some vibration remains when a palm or wrist is partly outside the lower edge of the camera view because the landmark model has less visible hand context. Stronger smoothing was not selected because it would add input lag and could delay legitimate strikes.

The live reports deliberately store no landmark coordinates, so the visual improvement is not presented as a quantitative tracking-accuracy result. A future robustness study can measure detection loss, recovery time, false hits, and hit success across controlled hand positions and lighting conditions without retaining camera images.

## Conclusions

Three engineering lessons came from the measurements:

1. Measure the whole pipeline. Inference was not the main bottleneck; serial camera waiting was.
2. Lowering resolution helps only the pixel-heavy stages. It improved rendering but could not solve a blocking architecture.
3. Real-time input should favour fresh data over queued data. Background latest-frame capture removed the serial wait while avoiding stale-camera latency.

The current pipeline meets its 30 FPS showcase target on the documented machine and camera configuration. The next evidence-focused task is a repeatable computer-vision robustness matrix covering lighting, backgrounds, one and two hands, crossed hands, partial occlusion, camera-edge positions, and recovery after detection loss.
