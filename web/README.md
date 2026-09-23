# Air Rhythm Web

This folder holds the browser version alongside the finished Python desktop app. The web version is still local-only on its development branch.

## Current milestone: tracking preview

The page now requests camera permission, loads MediaPipe Hand Landmarker in the browser, and tracks up to two hands. A generated black-sky stage shows virtual sticks anchored to the index fingertips. The only place camera pixels appear is the lower-right live-input inset, where the 21-point hand skeleton is drawn. An audio context is unlocked by the start button for later sound work; this milestone does **not** include falling notes, scoring, or hit sounds.

## Run locally

Use Node.js 20.19+ or 22.12+. From the repository root:

```sh
cd web
npm ci
npm run dev
```

Open the localhost address shown in the terminal, click **Enable camera & tracking**, and allow camera access. Move one or both hands into view. Click **Stop camera** to release the camera. Camera access in a browser requires localhost or HTTPS.

To check the production build:

```sh
npm run build
```

## What is reused

- `../models/hand_landmarker.task` is the same pretrained model file used by the desktop app. Vite includes that file in the web build; there is no second checked-in model copy.
- MediaPipe's browser WebAssembly files come from the pinned npm package. `npm run dev` and `npm run build` prepare the required SIMD and non-SIMD files under ignored `public/wasm/`.
- The generated stage, virtual-stick shape, and mirrored camera/skeleton inset follow the desktop design. The browser implementation uses Canvas instead of OpenCV.
- Video frames are processed on the visitor's device. This app does not upload or save camera frames. The MediaPipe runtime may handle usage metrics according to its own privacy notice.

Next, the desktop music, timing, smoothing, collision, scoring, and sound behaviour will be brought over and tested for parity. Phone layouts and performance will be refined after desktop gameplay works.
