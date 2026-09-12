# Air Rhythm

Air Rhythm is a camera-controlled musical game built with Python, OpenCV, and MediaPipe. Catch falling circles with your fingertips to play a recognisable melody, or use free play to make your own tune.

The project is being developed phase by phase as a practical computer-vision learning project.

## Project status

Phases 1–4 are complete. Air Rhythm now has a timed song challenge, free play, scoring, and privacy views. Colored circles fall from the top of the full camera frame at varied horizontal positions. Fingertip contact drives the music, while a shrinking halo teaches the player when to hit.

### Current capabilities

- Safe webcam startup and shutdown
- Mirrored camera preview for natural hand movement
- MediaPipe hand tracking in video mode
- Two-hand landmark skeletons and fingertip highlights
- Live detected-hand counter
- Invisible full-frame play area with no fixed lanes
- Time-based falling-node movement that stays consistent across frame rates
- Contact detection using all fingertips on both hands
- Path-based collision that catches fast movements between camera frames
- Separate musical-timing grades and movement-strength bonuses
- Downward and forward movement bonuses without requiring those gestures
- Hit and miss counters with short visual rating effects
- Immediate, overlapping synthesised sounds through a persistent audio output stream
- A simplified 35-note opening melody from Beethoven's *Für Elise*
- An absolute beat clock that stays aligned after a slow camera frame
- Shrinking timing halos with `PERFECT`, `GREAT`, and `GOOD` accuracy grades
- Score, combo, accuracy, final rank, countdown, and results screen
- Four colour-coded instruments and a free-play mode
- Hands-only privacy, skeleton-only privacy, and normal-camera views
- Mute, restart, and a camera-free sound check

### Next milestone

Phase 5 will focus on making the project ready to showcase:

- Camera and audio-delay calibration
- Adjustable difficulty and more song charts
- A polished title/tutorial screen
- Recording-friendly layout and performance tuning

## Play music with your hands

The game starts in **Challenge** mode. After a three-second countdown, each circle carries one scheduled note from the opening of *Für Elise*. The circle's white halo becomes smaller as its beat approaches. The ideal moment is when the halo meets the circle and turns green. A late halo turns warm orange.

The timing window is intentionally forgiving for camera play:

| Distance from the beat | Timing grade |
| --- | --- |
| Up to 0.10 seconds | `PERFECT` |
| Up to 0.22 seconds | `GREAT` |
| Up to 0.42 seconds | `GOOD` |

A very early contact waits until the playable window opens. If a circle passes the late edge of that window, it becomes a miss and breaks the combo. This keeps accidental contact near the top of the screen from playing a note far too early.

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
| `1` | Restart the timed *Für Elise* challenge |
| `2` | Free play: invent your own tune |
| `P` | Cycle hands only → skeleton only → normal camera |
| `M` | Mute/unmute audio |
| `R` | Restart the current mode, melody, and counters |
| `Q` | Quit |

Changing mode starts a fresh round. Muting clears ringing notes, but the game and score continue; press `R` to return to the beginning.

## Privacy views

Air Rhythm starts with the **normal camera** view. The hand skeleton remains visible over the full mirrored image.

Press `P` once for **Hands only**. The app replaces the face, body, room, and other camera pixels with an opaque dark stage, while revealing small hand shapes estimated from MediaPipe's 21 landmarks. Press `P` again for **Skeleton only**, which reveals no original camera pixels. Press it once more to return to the normal camera.

MediaPipe still receives the normal mirrored camera frame in every view. Privacy is applied only to the image shown in the game, so hiding the camera does not weaken hand detection.

The hands-only mask is estimated from landmarks rather than pixel-perfect hand segmentation. If a hand passes directly across a face, a few pixels behind the hand can fall inside its cut-out. Use skeleton-only mode when complete visual privacy is required. This privacy feature changes the live game display; it does not control separate screen-recording or camera software.

## How the current foundation works

```text
Webcam frame
    -> OpenCV mirrors the frame
    -> the frame is converted from BGR to RGB
    -> MediaPipe detects hand landmarks
    -> normalized landmarks are converted to pixels
    -> the selected privacy view replaces hidden camera pixels
    -> fingertip paths are checked against falling circles
    -> the beat clock compares contact time with each node's target time
    -> successful hits trigger prepared sounds in the background
    -> OpenCV draws the game and hand tracking
```

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
models/hand_landmarker.task  Local MediaPipe hand model
requirements.txt             Python dependencies
tests/                       Camera-free game, privacy, music, and audio checks
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
- [ ] Phase 5 — Calibration, visual polish, packaging, and release
- [ ] Future — AI-generated rhythm challenges
