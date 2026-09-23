# Air Rhythm Web

This folder is the browser version of Air Rhythm. It lives beside the finished Python desktop app so both versions can share project history without changing the desktop application.

Phase 1 is a runnable TypeScript/Vite foundation only. It does **not** request camera access, run hand tracking, or play music yet.

## Run locally

Use Node.js 20.19+ or 22.12+ (the current development machine uses Node.js 24).

```sh
cd web
npm ci
npm run dev
```

To check the production build:

```sh
npm run build
```

## Boundaries for the next phases

- Camera frames and hand tracking will stay on the visitor's device; the playable site needs no Python server.
- The existing `../models/hand_landmarker.task` asset and `../music.py` chart data are the sources to evaluate for reuse. Model compatibility with MediaPipe's browser runtime will be checked before adding it to the web build.
- The scoring, timing, smoothing, and collision rules will be translated from the Python modules and checked against the desktop behaviour.
- Browser-native Canvas and Web Audio will replace OpenCV window drawing and `sounddevice` output, while preserving the same interaction and appearance where practical.
- Mobile layout and performance will be tested on real devices before publication.

This branch is local-only until the web version has been tested and approved for a push.
