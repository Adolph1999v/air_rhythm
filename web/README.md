# Air Rhythm Web

This folder holds the browser version alongside the finished Python desktop app. The web version is still local-only on its development branch.

## Current milestone: playable browser challenge

The browser now plays the desktop app's 35-note song challenge and four-sound free-play mode. After enabling the camera, choose a mode. Circles enter from the top, fall at a constant speed, and stay hittable until they leave the bottom. The index fingertips steer the virtual sticks; any detected fingertip can touch a circle. A quick downward or forward movement improves the movement rating, but plain contact still counts. The browser synthesizes each hit locally with Web Audio—there is no song recording, Python server, or camera upload.

The song clock, order penalty, timing windows, score, and melody match the desktop logic. The checked-in browser melody data is compared against `../music.py` by a Python parity test. Camera landmarks are smoothed before rendering and fingertip-path collision checks. The main starfield is generated graphics; camera pixels appear only in the live-input inset with the 21-point hand skeleton.

Motion parity: both versions request 1280×720 at 30 FPS and cap larger camera frames at 1280×720 for tracking. The web version processes presented camera frames rather than repeatedly tracking the same frame at the screen's refresh rate. Challenge notes keep the desktop two-second trip to the 30%-height beat point and continue at that exact speed; free-play notes use the desktop 28%-of-stage-height-per-second fall with its 100 ms slow-frame cap. The same adaptive hand filter and movement thresholds remain in use.

## Run locally

Use Node.js 20.19+ or 22.12+. From the repository root:

```sh
cd web
npm ci
npm run dev
```

Open the localhost address shown in the terminal, click **Enable camera & tracking**, and allow camera access. Choose **Start song challenge** or **Free play**. Bring both hands into view and touch falling circles. Click **Stop camera** to release the camera. Camera access in a browser requires localhost or HTTPS.

Keyboard shortcuts: `1` or Space starts the song challenge; `2` starts free play; `R` restarts; `T` or Escape returns to the menu; `H` opens the paused help screen; `M` mutes. The same mode, help, mute, restart, and menu actions have on-screen buttons for pointer use.

To check the production build:

```sh
npm run build
npm test
```

From the repository root, `.venv/bin/python -m unittest discover -s tests` also checks that the web melody and instrument data match the Python source.

## What is reused

- `../models/hand_landmarker.task` is the same pretrained model file used by the desktop app. Vite includes that file in the web build; there is no second checked-in model copy.
- MediaPipe's browser WebAssembly files come from the pinned npm package. `npm run dev` and `npm run build` prepare the required SIMD and non-SIMD files under ignored `public/wasm/`.
- The generated stage, virtual-stick shape, and mirrored camera/skeleton inset follow the desktop design. The browser implementation uses Canvas instead of OpenCV, TypeScript for the game rules, and Web Audio for the desktop-style synthesized sounds.
- Video frames are processed on the visitor's device. This app does not upload or save camera frames. The MediaPipe runtime may handle usage metrics according to its own privacy notice.

This is the first playable browser milestone, not yet a final public demo. Manual camera-and-audio testing in Chrome and Safari is still needed for this new gameplay. Phone layouts, camera orientation, and real-device performance tuning are next; the desktop app's diagnostic benchmark has not yet been ported to the web.
