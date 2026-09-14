# Air Rhythm

Air Rhythm is an interactive computer-vision portfolio project built with Python, OpenCV, and MediaPipe. It turns live hand landmarks into musical input: move your fingertips through falling notes to play a recognisable melody, receive timing feedback, or experiment in free play.

The rhythm-game format makes the vision pipeline easy to see, but the main goal is practical OpenCV and ML/AI learning. The project is developed phase by phase to demonstrate real-time camera processing, landmark tracking, motion analysis, collision detection, audio timing, measurement, and honest technical communication to employers and recruiters.

MediaPipe supplies the **pretrained** Hand Landmarker used to find hands and their 21 landmarks. The surrounding pipeline—including camera handling, landmark history, fingertip paths, interaction logic, rhythm timing, audio, privacy rendering, and interface—is project engineering built around that model. MediaPipe was not trained by this project. A future milestone will add and evaluate an original temporal gesture classifier trained on recorded landmark sequences.

## Project status

Phases 1–4 are complete, and the Phase 5 portfolio interface is implemented and covered by camera-free rendering tests. Final live-camera and recording validation is still pending. Air Rhythm now presents the game on a generated, person-free performance stage: coloured circles fall across the full stage, virtual drumsticks mirror the tracked hands, and a compact lower-right live-input inset shows the real camera plus its landmark skeleton. Fingertip contact drives the music and timing grades appear after a hit.

### Current capabilities

- Safe webcam startup and shutdown
- Mirrored live-input preview for natural hand movement
- MediaPipe hand tracking in video mode
- Two-hand landmark skeletons and fingertip highlights in the live-input inset
- Live detected-hand counter
- Invisible full-frame play area with no fixed lanes
- Generated main performance stage with no copied webcam pixels
- Virtual drumsticks that use the index-finger landmark as each stick tip
- Time-based, constant-speed falling-node movement that stays consistent across frame rates
- Contact detection using all fingertips on both hands
- Path-based collision that catches fast movements between camera frames
- Separate musical-timing grades and movement-strength bonuses
- Downward and forward movement bonuses without requiring those gestures
- Hit and miss counters with short visual rating effects
- Immediate, overlapping synthesised sounds through a persistent audio output stream
- A simplified 35-note opening melody from Beethoven's *Für Elise*
- An absolute beat clock that stays aligned after a slow camera frame
- First challenge circle enters from the top exactly when the countdown reaches `GO`
- Reliable `HIT` feedback with optional `PERFECT`, `GREAT`, and `GOOD` timing bonuses
- Score, combo, completion percentage, final rank, countdown, and results screen
- A polished title screen and recording-friendly visual style
- A compact gameplay HUD with score, combo, hit/miss totals, and song progress
- Toggleable help and technical overlays
- Live FPS, MediaPipe inference time, detected-hand count, landmark count, and mean handedness-classification confidence in the technical view
- An optional OpenCV + MediaPipe technical overlay
- Four colour-coded instruments and a free-play mode
- Hands-only privacy, skeleton-only privacy, and normal-camera modes for the live-input inset
- Mute, restart, and a camera-free sound check

### Next milestone

The next work will strengthen the evidence behind the showcase:

- Measure camera capture, MediaPipe inference, game update, rendering, and audio latency.
- Add repeatable FPS, frame-time, and robustness benchmarks.
- Calibrate camera and audio delay from measured results.
- Collect labelled landmark sequences for a custom temporal gesture classifier.
- Compare the trained classifier with the current rule-based movement logic using precision, recall, F1 score, and a confusion matrix.

The complete career-focused roadmap, measurable completion checks, and deferred game ideas are in [FUTURE_PLAN.md](FUTURE_PLAN.md).

## Play music with your hands

The game starts in **Challenge** mode. The first circle enters from the top at the exact moment the three-second countdown reaches `GO`, then takes two seconds to reach its first beat in the upper third of the screen. Each later circle carries one scheduled note from the opening of *Für Elise*. A clean falling circle is the only target marker; after a touch, the score system reports how close it was to the musical beat. If it is never touched, it keeps the same speed, continues down, and becomes a miss only after leaving the bottom of the stage.

The timing window is intentionally forgiving for camera play:

| Distance from the beat | Timing grade |
| --- | --- |
| Up to 0.10 seconds | `PERFECT` |
| Up to 0.22 seconds | `GREAT` |
| Up to 0.42 seconds | `GOOD` |
| Outside 0.42 seconds while still visible | `HIT` |

Every visible circle remains physically hittable from the moment it enters at the top until it leaves through the bottom. Any contact removes the circle, plays its attached sound, and continues the hit combo. Contacts near the ideal song time can earn a `PERFECT`, `GREAT`, or `GOOD` timing bonus; contacts elsewhere receive the neutral `HIT` result. This avoids penalising the player for a hidden timing point after the fixed target and timing halo were removed from the interface.

Every caught circle is worth 1,000 base points. Musical timing adds 500 for `PERFECT`, 250 for `GREAT`, or 100 for `GOOD`; a normal `HIT` keeps the full base value. Completion is calculated only from caught circles versus misses, so catching every circle produces 100% completion regardless of the optional timing bonuses.

Timing and hand movement are separate. `PERFECT`, `GREAT`, and `GOOD` describe closeness to the musical beat. A normal touch is enough to hit; a stronger or deliberate downward/forward movement adds a small loudness and score bonus. The contact path between camera frames is still checked, so a quick sweep can count even when no single frame captures the fingertip inside the circle.

In **Free play**, each instrument has a fixed pitch. These four pitches fit together, so you can experiment with your own patterns.

| Circle colour | Label | Sound | Free-play pitch |
| --- | --- | --- | --- |
| Blue | KEYS | Soft keyboard | C5 |
| Gold | BELL | Bright bell | E5 |
| Green | PLUCK | Plucked string-like tone | G5 |
| Pink | MALLET | Warm marimba-like tone | A5 |

Challenge notes use the keyboard sound so the melody remains consistent and recognisable. In free play, colours identify sounds; they never require a particular gesture. Free-play feedback says `TOUCH`, `STRONG`, or `POWER` to keep movement strength distinct from musical timing.

### Controls

Select the camera window before pressing a key.

| Key | Action |
| --- | --- |
| `Space` | Start from the title screen or replay after Results |
| `1` | Restart the timed *Für Elise* challenge |
| `2` | Free play: invent your own tune |
| `P` | Cycle the live-input inset: hands only → skeleton only → normal camera |
| `M` | Mute/unmute audio |
| `R` | Restart the current mode, melody, and counters |
| `H` | Show or hide Help; an active round pauses safely |
| `D` | Show or hide the technical overlay |
| `T` or `Esc` | Return to the title screen |
| `Q` | Quit |

Changing mode starts a fresh round. Muting clears ringing notes, but the game and score continue; press `R` to return to the beginning. Opening Help safely pauses the active round and moves its beat clock forward by the paused time. The optional technical overlay exposes live computer-vision information for demonstrations and debugging.

## Privacy views

The large performance stage is always person-free: it is drawn from scratch and does not copy camera pixels. It uses the tracked landmarks only to place and rotate the virtual drumsticks.

The lower-right **LIVE INPUT** inset starts in **normal camera** mode. It shows the mirrored camera plus the coloured hand skeleton, making the live OpenCV and MediaPipe pipeline visible in a demo without placing the player on the main stage.

Press `P` once for **Hands only** in that inset. The app replaces the face, body, room, and other camera pixels with an opaque stage, while revealing small hand shapes estimated from MediaPipe's 21 landmarks. Press `P` again for **Skeleton only**, which reveals no original camera pixels. Press it once more to return the inset to the normal camera.

MediaPipe still receives the normal mirrored camera frame in every input mode. Privacy is applied only to the displayed inset, so hiding camera pixels does not weaken hand detection.

The hands-only mask is estimated from landmarks rather than pixel-perfect hand segmentation. If a hand passes directly across a face, a few pixels behind the hand can fall inside its cut-out. Use skeleton-only mode when complete visual privacy is required. This feature changes the app display; it does not control separate screen-recording or camera software.

## How the current foundation works

```text
Webcam frame
    -> OpenCV mirrors the frame
    -> the frame is converted from BGR to RGB
    -> MediaPipe detects hand landmarks
    -> normalized landmarks are converted to pixels
    -> the index-fingertip landmarks place virtual drumstick tips on a generated stage
    -> the selected privacy view prepares only the lower-right live-input inset
    -> fingertip paths are checked against falling circles
    -> the beat clock compares contact time with each node's target time
    -> successful hits trigger prepared sounds in the background
    -> OpenCV draws the performance stage, interface, drumsticks, and input skeleton
```

MediaPipe performs pretrained landmark inference; OpenCV owns the surrounding live image pipeline and display. The application converts the model's normalized output into pixel positions, keeps short movement histories, maps each index fingertip to a 2D virtual drumstick, checks fingertip paths between frames, and combines contact time with the song clock. This distinction matters when describing the project: the current AI component is an integrated pretrained model, while the motion, interaction, timing, and presentation systems are original application engineering.

Press `D` during the demonstration to reveal the technical overlay. It makes the active pipeline visible through live FPS, MediaPipe inference time, hand count, processed landmark count, and the mean Left/Right handedness-classification confidence when available. That value is not presented as overall tracking accuracy. Press `D` again for the cleaner recording view.

The sound engine prepares short waveforms once at startup. A waveform is a list of numbers telling the speaker how to move. On a hit, the game requests a prepared sound instead of loading a file or generating a new tone in the camera loop. A separate audio callback mixes ringing notes together, allowing quick consecutive hits without cutting the previous sound off.

NumPy creates the waveforms, and sounddevice sends them to your selected speakers. The audio stream is output-only and does not use the microphone. If sound output cannot start, the camera game continues and shows a message; the terminal prints the reason.

## Run locally

Create and prepare the project-specific virtual environment:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
```

If the environment already exists, run only the install command to check its dependencies. NumPy and sounddevice may already be installed through MediaPipe; they are also listed explicitly because the game now uses them directly.

Run Air Rhythm:

```bash
.venv/bin/python app.py
```

To hear the melody without opening the camera:

```bash
.venv/bin/python app.py --sound-test
```

This plays the keyboard melody with a simple reference rhythm, then exits. Press `Ctrl+C` to stop early. For your first test, use the Mac's speakers or wired headphones; Bluetooth can add noticeable delay. If sound is unavailable, check the selected macOS output device and restart the app.

### Music source

The bundled tune is a simplified, single-note arrangement of the opening of Beethoven's *Für Elise*. Its pitches come from the [Mutopia public-domain score](https://www.mutopiaproject.org/ftp/BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59-let.pdf). The app synthesises its own audio; it does not include a recording or the full piano accompaniment. The note sequence lives in `music.py`, ready for future melody additions.

## Project structure

```text
app.py                       Main camera and hand-tracking application
audio_engine.py              Prepared tones and background audio mixing
music.py                     Instruments, melody notes, and note progression
privacy.py                   Hands-only and skeleton-only display rendering
rhythm_game.py               Song clock, timing grades, score, and round state
performance_stage.py         Person-free stage, virtual drumsticks, and input inset
ui.py                        Reusable title, HUD, help, debug, and results drawing
models/hand_landmarker.task  Local MediaPipe hand model
requirements.txt             Python dependencies
tests/                       Camera-free game, privacy, music, and audio checks
FUTURE_PLAN.md               Career-focused milestones and deferred ideas
```

Run the automated checks without opening the camera or speakers:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Roadmap

- [x] Phase 1 — Camera foundation
- [x] Phase 2 — Two-hand landmark tracking
- [x] Phase 3 — Free-falling nodes and hybrid fingertip collision
- [x] Phase 4 — Sound, beat scheduling, scoring, and first music chart
- [x] Phase 5 implementation — Portfolio UI, compact HUD, help, progress, and technical overlay
- [ ] Phase 5 validation — Live 720p/1080p camera check and a clear 30-second recording
- [ ] Next — Measured computer-vision performance, latency, calibration, and robustness
- [ ] Later — Custom temporal gesture dataset, model training, and evaluation

See [FUTURE_PLAN.md](FUTURE_PLAN.md) for definitions of done, learning outcomes, and optional product ideas kept outside the current portfolio scope.
