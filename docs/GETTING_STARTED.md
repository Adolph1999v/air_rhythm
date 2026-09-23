# Getting Started

Air Rhythm has been tested on macOS with an Apple Silicon Mac, a webcam, Python 3.14, and `uv`. The camera game uses the webcam and speakers; the automated tests can run without either one. Other operating systems have not yet been validated.

## Get the project

To download without Git:

1. Open the [Air Rhythm repository](https://github.com/Adolph1999v/air_rhythm).
2. Select **Code → Download ZIP**.
3. Unzip the download. Open Terminal and move into the extracted `air_rhythm-main` folder. If you left it in Downloads, use:

   ```bash
   cd ~/Downloads/air_rhythm-main
   ```

If you use Git instead, clone the same repository and move into it:

```bash
git clone https://github.com/Adolph1999v/air_rhythm.git
cd air_rhythm
```

All following commands run from that project folder.

## Install dependencies

Check that `uv` is available with `uv --version`. Then create the project environment and install the dependencies:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
```

If `.venv` already exists, run only the `uv pip install` command to check its dependencies. The local MediaPipe model is stored at `models/hand_landmarker.task`; the app loads it from the project folder.

If you already have Python 3.14 but not `uv`, the standard-library environment and `pip` also work:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the app

```bash
.venv/bin/python app.py
```

Allow Camera access for the app that launched Python, such as Terminal or your editor, when macOS asks. Select the Air Rhythm window before using its keyboard controls. Press `Space` to start the challenge or `2` for free play, and `Q` to quit. The full controls and scoring rules are in the [Play Guide](PLAY_GUIDE.md).

The app requests 1280×720 at 30 FPS. A camera may deliver a different mode: larger frames are scaled down for hand tracking, and smaller frames keep their native size. Frame rate is determined by the camera and computer. The performance stage adapts to the processed camera size and the window can be resized.

## Check sound without opening the camera

```bash
.venv/bin/python app.py --sound-test
```

This plays the simplified melody and exits. Press `Ctrl+C` to stop early. For the first check, use the Mac speakers or wired headphones; Bluetooth devices may add noticeable delay. The microphone is not used.

## Run automated checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These checks do not require a live camera or speaker.

## Common problems

- **Camera does not open:** Check that a camera is connected and that Terminal or your editor has permission under **System Settings → Privacy & Security → Camera**. Close other apps that may be holding the camera, then restart Air Rhythm.
- **No sound:** Check the selected macOS output device. Restart the app after changing devices. The terminal prints the audio error if output cannot start; the camera game can still run.
- **Hand tracking is unstable near an edge:** Keep the whole palm and wrist in view and improve lighting. Partial hands and motion blur can make the pretrained landmark estimates less stable.
- **The display looks softer or runs slower:** A lower-resolution camera can still be used, but enlarging its image does not add detail. Actual FPS also depends on the camera, driver, and computer.

For the implementation and measured performance, see the [Technical Overview](TECHNICAL_OVERVIEW.md) and [Performance Study](PERFORMANCE_STUDY.md).
