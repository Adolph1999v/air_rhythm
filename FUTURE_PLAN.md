# Air Rhythm — Future Plan

## Purpose

Air Rhythm is a learning and portfolio project that demonstrates practical computer vision and machine-learning skills.
The current goal is to help employers and recruiters quickly understand what was built, why it works, and how its quality was measured.
Releasing a complete commercial game is not the current goal.

## Current technical baseline

The project already uses:

- OpenCV for webcam capture, frame processing, drawing, and the live interface.
- MediaPipe's pretrained Hand Landmarker for 21 landmarks on each detected hand.
- Landmark history for fingertip movement and direction estimates.
- Path-based collision so fast motion between frames can still hit a node.
- A timed rhythm challenge, scoring, synthesised audio, and privacy views.
- A portfolio interface with title, gameplay HUD, help, results, and live technical telemetry.

MediaPipe is a pretrained model used by this project.
It must not be described as a model trained by us.
A future custom gesture classifier will be the project's own trained ML component.

## Priority order

Plans are ranked by career and learning value:

1. Make the existing computer-vision pipeline easy to see and explain.
2. Measure speed, latency, and reliability with repeatable evidence.
3. Build and evaluate a custom temporal gesture classifier.
4. Test robustness across realistic camera conditions.
5. Produce a short demo and a clear technical case study.
6. Consider optional game and release features only after the portfolio evidence is complete.

## Milestone 1 — Portfolio-ready interface and technical visibility

**Status:** Implemented with camera-free tests; live-camera and recording validation is still pending.

### Work

- Create a polished title and tutorial screen.
- Keep the gameplay screen clean for screen recording.
- Show score, combo, timing feedback, song progress, and hand-detection status clearly.
- Keep the MediaPipe hand skeleton visible during the demonstration.
- Add a debug view for FPS, inference time, detected hands, landmark count, and handedness-classification confidence when available.
- Add a small OpenCV + MediaPipe label without making the interface look like a technical dashboard.
- Move drawing responsibilities out of the main camera loop where practical.

### Definition of Done

- A new viewer understands the interaction within five seconds.
- The title, gameplay, and results screens fit correctly at 1280×720 and 1920×1080.
- Debug information can be turned on and off with one key.
- Normal play remains readable when the debug view is off.
- A 30-second recording clearly shows hand tracking, node contact, timing feedback, and score changes.

### Learning outcomes

- Separate computer-vision processing from interface rendering.
- Design live overlays that stay readable over changing camera frames.
- Present technical information without distracting from the interaction.

## Milestone 2 — Latency calibration and performance measurement

### Work

- Measure camera capture, MediaPipe inference, game update, rendering, and audio-trigger time separately.
- Add a rolling FPS and frame-time measurement.
- Record average, median, 95th-percentile, and worst frame times.
- Create a simple audio-delay calibration flow for the local computer.
- Compare speakers, wired headphones, and Bluetooth only when those devices are available.
- Profile the main loop and optimise the slowest measured stage.
- Document the test machine, camera resolution, and test settings.

### Definition of Done

- A benchmark can run for at least 60 seconds without opening private recordings or saving camera images.
- The benchmark table reports FPS and timing for each major pipeline stage.
- The game stays responsive at the chosen showcase resolution.
- Calibration produces a saved timing offset that the beat clock can use.
- Before-and-after measurements support every performance optimisation claimed in the README.

### Learning outcomes

- Understand latency across a real-time computer-vision pipeline.
- Use profiling evidence to choose optimisations.
- Explain why average speed alone can hide dropped or delayed frames.

## Milestone 3 — Custom temporal gesture classifier

### Goal

Train a small model that recognises movement from a sequence of hand landmarks over time.
This will add original ML work while MediaPipe continues to provide the raw hand landmarks.

### Dataset

- Record short landmark sequences rather than camera images by default.
- Store normalized x, y, and z values plus timestamps and handedness.
- Include labels such as `idle`, `touch`, `swipe_down`, `swipe_left`, `swipe_right`, and `push_forward`.
- Collect multiple speeds, hand sizes, positions, distances, and lighting conditions.
- Record several sessions so train and test data do not come from the same continuous capture.
- Add examples where a movement starts but does not finish.
- Keep dataset consent and privacy notes with the collection script.

### Model work

- Start with the current rule-based movement logic as the baseline.
- Train a simple classical baseline using features such as velocity, direction, distance, and duration.
- Compare it with one temporal model, such as a small 1D CNN or LSTM, only if the dataset is large enough.
- Keep model input, preprocessing, and prediction code reproducible.
- Measure prediction time inside the live application.

### Evaluation

- Split data by recording session, and by person when multiple volunteers are available.
- Report accuracy, per-class precision, recall, and F1 score.
- Produce a confusion matrix to show which gestures the model mixes up.
- Measure false activations during `idle` sequences.
- Compare the trained model with the rule-based baseline under the same test data.
- Create a model card covering data, intended use, limits, privacy, metrics, and known failure cases.

### Definition of Done

- The dataset format and label guide are documented.
- A repeatable command trains the model from saved landmark sequences.
- A held-out test set is never used during training or tuning.
- The report contains precision, recall, F1, a confusion matrix, and live inference time.
- The live app can switch between rule-based and trained gesture recognition for comparison.
- The README states exactly which part was trained by us and which part comes from MediaPipe.

### Learning outcomes

- Design and label a small temporal dataset.
- Prevent data leakage between training and evaluation.
- Compare a learned model with a meaningful baseline.
- Interpret class-level errors rather than relying on one accuracy number.
- Integrate a trained model into a real-time OpenCV application.

## Milestone 4 — Computer-vision robustness

### Work

- Build a repeatable test checklist for bright, dim, and uneven lighting.
- Test plain and cluttered backgrounds.
- Test one hand, two hands, crossed hands, partial hands, and brief occlusion.
- Give each detected hand a stable identity across frames so crossed hands cannot share movement history accidentally.
- Test slow movement and fast movement at several distances from the camera.
- Record detection loss, false hits, missed hits, and recovery time.
- Tune smoothing and collision settings from measurements rather than one recording.
- Add graceful feedback when hand landmarks become unstable or disappear.

### Definition of Done

- At least three lighting conditions and three background conditions are documented.
- A fixed interaction script is repeated for each condition.
- The results table reports hit success, false-hit count, tracking loss, and recovery time.
- Known failure cases are shown honestly in the case study.
- Selected thresholds and smoothing values include a short evidence-based explanation.

### Learning outcomes

- Test a vision system beyond the ideal development setup.
- Understand occlusion, lighting, motion blur, and landmark jitter.
- Turn observed failures into measurable engineering decisions.

## Milestone 5 — Demo evidence and case study

### Portfolio assets

- Create an architecture diagram from webcam input to audio output.
- Record a 30–60 second demo with normal gameplay and a short debug-view segment.
- Add a benchmark table for FPS, pipeline latency, and gesture metrics.
- Add a concise README case study covering the problem, design, challenges, results, and lessons.
- Include one short example of a failure and the improvement made from it.
- Add screenshots of the title screen, gameplay, debug view, and results screen.
- Prepare a short LinkedIn caption focused on the computer-vision and ML learning.

### Definition of Done

- The repository can be understood without running the program.
- Every performance or ML claim links to a measurement or test result.
- The architecture diagram matches the current code.
- The demo shows the hand skeleton and real interaction without edits that hide failures.
- Setup instructions work from a clean local environment.
- The case study clearly distinguishes existing libraries from original engineering work.

### Learning outcomes

- Communicate technical work to both engineers and recruiters.
- Support portfolio claims with evidence.
- Explain tradeoffs, limitations, and lessons with confidence.

## Optional later product and game ideas

These ideas can make Air Rhythm more game-like, but they come after the portfolio milestones:

- Easy, Normal, and Hard difficulties.
- More public-domain or original song charts.
- A song-selection library.
- User-created rhythm charts.
- More instruments and sound packs.
- Calibration profiles for different cameras and audio devices.
- Improved privacy using hand segmentation rather than a landmark-shaped mask.
- Packaging as a standalone desktop application.
- Installer, release builds, automatic updates, and broader operating-system support.
- Accessibility settings and remappable controls.
- AI-assisted rhythm-chart generation, evaluated separately from the gesture classifier.

## Not in the current scope

- Commercial game release or monetisation.
- Large content libraries or licensed popular-song recordings.
- Online multiplayer, accounts, leaderboards, or cloud services.
- Mobile and console versions.
- Complex menus, cosmetics, achievements, or progression systems.
- Anti-cheat systems or competitive balancing.
- Production support for every camera, speaker, and operating system.
- Claiming that MediaPipe's Hand Landmarker was trained by this project.

New ideas should enter the current roadmap only when they improve one of these goals:

- Demonstrate OpenCV or real-time computer-vision engineering.
- Add honest, measurable ML work.
- Improve the quality of the technical evidence.
- Teach a skill that can be clearly explained in an interview.

Otherwise, they belong in the optional list until the portfolio version is complete.
