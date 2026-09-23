# Play Guide

Air Rhythm has a short timed challenge and a free-play mode. The main performance stage shows falling circles and virtual drumsticks. A smaller **LIVE INPUT** inset shows the real camera and the tracked hand skeleton.

## Challenge mode

Press `Space` on the title screen. A three-second countdown runs; the first circle enters from the top when it reaches `GO`. That circle takes two seconds to reach its beat in the upper third of the stage. Later circles carry the scheduled notes of a simplified melody.

There are no visible lanes or fixed targets. A circle can be hit anywhere while it remains on screen, from the top until it leaves through the bottom. The next unhit circle is a little brighter and each circle shows its order number. Missing one does not make it accelerate: its speed stays constant until it leaves the stage.

Touch a circle with any fingertip on either tracked hand to hit it. The index fingertip sits at the tip of its virtual drumstick, and the other four fingertips have smaller visible collision markers. The game also checks the fingertip path between camera frames, helping it catch quick sweeps. A normal touch is enough; a deliberate downward or forward motion can add a small movement bonus.

### Timing and score

| Distance from the scheduled beat | Result | Timing bonus |
| --- | --- | ---: |
| Up to 0.10 seconds | `PERFECT` | 500 points |
| Up to 0.22 seconds | `GREAT` | 250 points |
| Up to 0.42 seconds | `GOOD` | 100 points |
| Farther away, while the circle is visible | `HIT` | 0 points |

Every caught circle earns 1,000 base points. Combo and deliberate movement can add smaller bonuses. Completion depends on how many circles you catch, not on the timing labels.

Playing in order matters for timing bonuses. Touching a later circle first gives it a normal `HIT` and marks earlier skipped circles as out of order. You can still catch those earlier circles for sound and base points, but their timing bonuses cannot be restored. Once that skipped section is resolved, the next untouched circle can earn timing bonuses again.

An untouched circle counts as a miss only after it exits the bottom of the stage. A miss breaks the combo. The absolute song clock keeps future notes scheduled after a slow camera frame or a pause in Help.

## Free play

Press `2` to play four musical pitches in any order. The colours indicate sounds, not required hand gestures. These pitches belong to a C-major pentatonic set, so simple combinations tend to work together.

| Circle colour | Label | Sound | Pitch |
| --- | --- | --- | --- |
| Blue | KEYS | Soft keyboard | C5 |
| Gold | BELL | Bright bell | E5 |
| Green | PLUCK | Plucked string-like tone | G5 |
| Pink | MALLET | Warm marimba-like tone | A5 |

Free-play feedback says `TOUCH`, `STRONG`, or `POWER` for movement strength. Those are separate from the challenge's musical-timing grades.

## Controls

Select the Air Rhythm camera window before pressing a key.

| Key | Action |
| --- | --- |
| `Space` | Start the challenge from the title screen or replay after Results |
| `1` | Restart the timed melody challenge |
| `2` | Start free play |
| `P` | Cycle the input inset: hands only → skeleton only → normal camera |
| `M` | Mute or unmute; muting clears sounds already ringing |
| `R` | Restart the current mode, melody, and counters |
| `H` | Show or hide Help; an active round pauses safely |
| `D` | Show or hide the technical overlay |
| `B` | Start or stop a performance benchmark and save its reports |
| `T` or `Esc` | Return to the title screen |
| `Q` | Quit |

Changing mode starts a fresh round. Muting does not stop the game or its score.

## Camera visibility

The main stage is generated without copying camera pixels, so your face, body, and room are not shown there. The small **LIVE INPUT** inset starts in normal-camera mode to show the real image and hand skeleton together. Press `P` for a hands-only inset, again for a skeleton-only inset, and again to return to normal camera.

The hands-only shape is estimated from the 21 hand landmarks rather than pixel-perfect image segmentation. If a hand passes in front of a face or private object, some background pixels inside that shape can remain visible. Choose skeleton-only mode when you need the inset to show no original camera pixels. These modes affect the app display; they do not control separate screen-recording software.

MediaPipe still analyses the mirrored camera image in every display mode. Hiding camera pixels in the inset does not change its hand-detection input.

## The music

The challenge uses a simplified, 35-note version of the opening right-hand melody from Beethoven's *Für Elise*. Its pitches were transcribed from the [Mutopia public-domain score](https://www.mutopiaproject.org/ftp/BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59-let.pdf). The app stores note pitches and preview timings in `music.py`, schedules circles from that chart, and synthesises its own sounds when you hit them. No recording or full piano accompaniment is bundled.

For installation and camera permissions, see [Getting Started](GETTING_STARTED.md). For the implementation, see the [Technical Overview](TECHNICAL_OVERVIEW.md).
