"""Small, responsive instruments made locally from sound waves.

Importing this module never opens an audio device. Call ``start()`` once, then
``play_note()`` for each successful hit; the camera loop never waits for a note
to finish. Only the output stream is used, so no microphone access is needed.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import importlib
import math

import numpy as np


INSTRUMENTS = ("keys", "bell", "pluck", "marimba")
MAX_VOICES = 16
MAX_PENDING_NOTES = 64
MIX_CHUNK_FRAMES = 2048

# Each partial is (pitch multiplier, loudness, decay time in seconds).
# Different blends make the same musical note sound like different instruments.
_TIMBRES = {
    "keys": (1.5, ((1.0, 1.0, 0.55), (2.0, 0.30, 0.28), (3.0, 0.10, 0.16))),
    "bell": (1.8, ((1.0, 1.0, 0.70), (2.0, 0.35, 0.45), (3.0, 0.14, 0.23), (4.2, 0.06, 0.15))),
    "pluck": (1.0, ((1.0, 1.0, 0.29), (2.0, 0.40, 0.15), (3.0, 0.17, 0.09), (4.0, 0.08, 0.06))),
    "marimba": (0.9, ((1.0, 1.0, 0.25), (4.0, 0.28, 0.08), (10.0, 0.05, 0.03))),
}


def synthesize_note(midi_note: int, instrument: str, sample_rate: float) -> np.ndarray:
    """Build one reusable mono note; MIDI 69 is the familiar A at 440 Hz."""
    if not 0 <= midi_note <= 127:
        raise ValueError("MIDI notes must be between 0 and 127.")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("The sample rate must be positive.")
    duration, partials = _TIMBRES[instrument]
    frequency = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)
    times = np.arange(max(2, round(duration * sample_rate))) / sample_rate
    samples = np.zeros(len(times), dtype=np.float64)
    for multiplier, amplitude, decay in partials:
        # Frequencies above the device's limit would fold into unwanted pitches.
        if frequency * multiplier < sample_rate * 0.48:
            samples += (
                amplitude
                * np.sin(2.0 * np.pi * frequency * multiplier * times)
                * np.exp(-times / decay)
            )

    # A gentle start and finish prevent clicks when a note begins or ends.
    attack_frames = min(len(samples), max(2, round(sample_rate * 0.004)))
    release_frames = min(len(samples), max(2, round(sample_rate * 0.03)))
    samples[:attack_frames] *= np.linspace(0.0, 1.0, attack_frames)
    samples[-release_frames:] *= np.linspace(1.0, 0.0, release_frames)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples /= peak
    return samples.astype(np.float32)


@dataclass
class _Voice:
    """One note being played; only the audio callback changes these slots."""

    samples: np.ndarray | None = None
    position: int = 0
    gain: float = 0.0


class AudioEngine:
    """Keep one audio stream open and mix a bounded number of note tails."""

    def __init__(self, pitches: Iterable[int], volume: float = 0.45) -> None:
        self._pitches = tuple(sorted(set(pitches)))
        if any(not isinstance(note, int) or not 0 <= note <= 127 for note in self._pitches):
            raise ValueError("Pitches must be integer MIDI notes between 0 and 127.")
        if not math.isfinite(volume):
            raise ValueError("Volume must be a finite number.")
        self._volume = max(0.0, min(1.0, volume))
        self._samples: dict[tuple[int, str], np.ndarray] = {}
        self._pending: deque[tuple[int, np.ndarray, float]] = deque(maxlen=MAX_PENDING_NOTES)
        self._voices = [_Voice() for _ in range(MAX_VOICES)]
        self._scratch = np.empty(MIX_CHUNK_FRAMES, dtype=np.float32)
        self._next_voice = 0
        self._generation = 0
        self._callback_generation = 0
        self._stream = None
        self._callback_abort = RuntimeError
        self._enabled = False
        self._muted = False
        self.error_message: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether audio started successfully and is still available."""
        return self._enabled

    @property
    def muted(self) -> bool:
        """Whether the user has temporarily silenced note playback."""
        return self._muted

    def start(self) -> bool:
        """Prepare the notes and open speakers; failure leaves the game usable."""
        if self._enabled:
            return True
        self.close()
        self.error_message = None
        try:
            # sounddevice initializes PortAudio on import, so load it only here.
            sounddevice = importlib.import_module("sounddevice")
            device = sounddevice.query_devices(kind="output")
            channels = min(2, int(device["max_output_channels"]))
            sample_rate = float(device["default_samplerate"])
            if channels < 1:
                raise RuntimeError("No audio output device is available.")
            if not math.isfinite(sample_rate) or sample_rate <= 0:
                raise RuntimeError("The audio output device has no usable sample rate.")

            self._samples = {
                (note, instrument): synthesize_note(note, instrument, sample_rate)
                for note in self._pitches
                for instrument in INSTRUMENTS
            }
            self._callback_abort = sounddevice.CallbackAbort
            self._stream = sounddevice.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                latency="low",
                blocksize=0,
                callback=self._audio_callback,
                finished_callback=self._on_stream_finished,
            )
            self._enabled = True
            self._stream.start()
            if not self._stream.active:
                raise RuntimeError("The audio output stream did not start.")
            return self._enabled
        except Exception as error:
            self._enabled = False
            self.error_message = f"Sound unavailable: {error}"
            self.close()
            return False

    def play_note(self, midi_note: int, instrument: str = "keys", velocity: float = 1.0) -> None:
        """Queue a prepared note immediately; unknown notes are safely ignored."""
        if not self._enabled or self._muted or not math.isfinite(velocity):
            return
        samples = self._samples.get((midi_note, instrument))
        if samples is None or velocity <= 0:
            return
        gain = self._volume * min(1.0, velocity)
        # There is one producer (the camera loop) and one consumer (audio).
        # Bounded deque append/popleft operations keep hits from building a backlog.
        self._pending.append((self._generation, samples, gain))

    def set_muted(self, muted: bool) -> None:
        """Silence old notes so unmuting only plays fresh hits."""
        self._muted = bool(muted)
        self.stop_all()

    def stop_all(self) -> None:
        """Discard queued hits and clear playing notes at the next audio buffer."""
        self._generation += 1
        self._pending.clear()

    def close(self) -> None:
        """Release speakers safely, including after a partial startup failure."""
        self._enabled = False
        self.stop_all()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _on_stream_finished(self) -> None:
        if self._enabled:
            self.error_message = "Sound unavailable: the audio output stopped."
        self._enabled = False

    def _clear_voices(self) -> None:
        for voice in self._voices:
            voice.samples = None
            voice.position = 0
            voice.gain = 0.0
        self._next_voice = 0

    def _audio_callback(self, outdata, frames, _time, _status) -> None:
        """Fill a speaker buffer using existing samples; never wait or print here."""
        outdata.fill(0)
        try:
            generation = self._generation
            if generation != self._callback_generation:
                self._clear_voices()
                self._callback_generation = generation
            if self._muted or not self._enabled:
                return

            for _ in range(min(len(self._pending), MAX_PENDING_NOTES)):
                try:
                    command_generation, samples, gain = self._pending.popleft()
                except IndexError:
                    break  # The main thread may have cleared the queue.
                if generation != self._generation:
                    # Keep a fresh hit if reset/unmute happened while we read it.
                    generation = self._generation
                    self._clear_voices()
                    self._callback_generation = generation
                    if self._muted or not self._enabled:
                        return
                if command_generation != generation:
                    continue
                voice = self._voices[self._next_voice]
                voice.samples, voice.position, voice.gain = samples, 0, gain
                self._next_voice = (self._next_voice + 1) % MAX_VOICES

            # The host may ask for different buffer sizes. Fixed small pieces
            # let us reuse scratch memory without allocating an audio buffer here.
            for offset in range(0, frames, MIX_CHUNK_FRAMES):
                chunk_frames = min(MIX_CHUNK_FRAMES, frames - offset)
                output = outdata[offset : offset + chunk_frames, 0]
                for voice in self._voices:
                    if voice.samples is None:
                        continue
                    count = min(chunk_frames, len(voice.samples) - voice.position)
                    np.multiply(
                        voice.samples[voice.position : voice.position + count],
                        voice.gain,
                        out=self._scratch[:count],
                    )
                    np.add(output[:count], self._scratch[:count], out=output[:count])
                    voice.position += count
                    if voice.position >= len(voice.samples):
                        voice.samples = None

            # Smoothly tame loud chords before they reach the speaker's limits.
            np.tanh(outdata[:, 0], out=outdata[:, 0])
            outdata[:, 0] *= 0.95
            if outdata.shape[1] == 2:
                outdata[:, 1] = outdata[:, 0]

            # A mute/reset arriving during the mix should also silence this buffer.
            if self._muted or generation != self._generation or not self._enabled:
                outdata.fill(0)
                self._clear_voices()
        except Exception:
            outdata.fill(0)
            self._enabled = False
            self.error_message = "Sound unavailable: audio playback stopped unexpectedly."
            raise self._callback_abort
