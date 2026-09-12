"""Exercise sound generation and playback without using real speakers."""

from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audio_engine import AudioEngine, INSTRUMENTS, MAX_PENDING_NOTES, MAX_VOICES, synthesize_note


class FakeStream:
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def abort(self):
        self.active = False
        self.settings["finished_callback"]()

    def close(self):
        self.closed = True


class AudioEngineTests(unittest.TestCase):
    def make_engine(self, channels=2):
        fake_device = SimpleNamespace(
            query_devices=lambda **kwargs: {
                "max_output_channels": channels,
                "default_samplerate": 24000.0,
            },
            OutputStream=FakeStream,
            CallbackAbort=RuntimeError,
        )
        engine = AudioEngine([60, 64, 67, 69])
        with patch("audio_engine.importlib.import_module", return_value=fake_device):
            self.assertTrue(engine.start())
        self.addCleanup(engine.close)
        return engine

    def render(self, engine, frames=1024):
        channels = engine._stream.settings["channels"]
        output = np.full((frames, channels), np.nan, dtype=np.float32)
        engine._audio_callback(output, frames, None, None)
        self.assertTrue(np.isfinite(output).all())
        return output

    def test_instruments_have_expected_pitch_and_smooth_endpoints(self):
        for instrument in INSTRUMENTS:
            with self.subTest(instrument=instrument):
                samples = synthesize_note(69, instrument, 24000)
                self.assertEqual(samples.dtype, np.float32)
                self.assertEqual(samples[0], 0)
                self.assertEqual(samples[-1], 0)
                self.assertLessEqual(np.max(np.abs(samples)), 1.0)
                spectrum = np.abs(np.fft.rfft(samples))
                pitch = np.fft.rfftfreq(len(samples), 1.0 / 24000)[np.argmax(spectrum)]
                self.assertAlmostEqual(pitch, 440.0, delta=2.0)

    def test_constructor_does_not_initialize_hardware(self):
        with patch("audio_engine.importlib.import_module") as importer:
            engine = AudioEngine([60])
            self.assertFalse(engine.enabled)
            importer.assert_not_called()

    def test_output_stream_uses_device_rate_and_supports_mono(self):
        engine = self.make_engine(channels=1)
        self.assertEqual(engine._stream.settings["samplerate"], 24000)
        self.assertEqual(engine._stream.settings["latency"], "low")
        self.assertEqual(engine._stream.settings["blocksize"], 0)
        engine.play_note(60)
        self.assertGreater(np.max(np.abs(self.render(engine))), 0.0)

    def test_note_queue_and_polyphony_are_bounded_without_clipping(self):
        engine = self.make_engine()
        for _ in range(MAX_PENDING_NOTES * 4):
            engine.play_note(60)
        self.assertEqual(len(engine._pending), MAX_PENDING_NOTES)
        output = self.render(engine, frames=5000)
        self.assertEqual(sum(voice.samples is not None for voice in engine._voices), MAX_VOICES)
        self.assertLess(np.max(np.abs(output)), 1.0)
        np.testing.assert_array_equal(output[:, 0], output[:, 1])

    def test_chord_mixes_more_than_one_pitch(self):
        engine = self.make_engine()
        for pitch in (60, 64, 67):
            engine.play_note(pitch)
        output = self.render(engine, frames=8000)[:, 0]
        spectrum = np.abs(np.fft.rfft(output))
        frequencies = np.fft.rfftfreq(len(output), 1.0 / 24000)
        for frequency in (261.63, 329.63, 392.0):
            near_pitch = spectrum[np.abs(frequencies - frequency) < 4]
            self.assertGreater(near_pitch.max(), spectrum.max() * 0.3)

    def test_mute_discards_active_and_queued_notes_even_if_unmuted_between_callbacks(self):
        engine = self.make_engine()
        engine.play_note(60)
        self.assertTrue(self.render(engine).any())
        engine.play_note(64)
        engine.set_muted(True)
        engine.play_note(67)
        engine.set_muted(False)
        self.assertFalse(self.render(engine).any())
        engine.play_note(69)
        self.assertTrue(self.render(engine).any())

    def test_stop_all_silences_queued_and_active_notes(self):
        engine = self.make_engine()
        engine.play_note(60)
        self.render(engine)
        engine.play_note(64)
        engine.stop_all()
        self.assertFalse(self.render(engine).any())

    def test_reset_during_queue_read_keeps_the_fresh_hit(self):
        engine = self.make_engine()

        class ResetOnRead(deque):
            def popleft(self):
                engine.stop_all()
                engine.play_note(69)
                return super().popleft()

        engine._pending = ResetOnRead(maxlen=MAX_PENDING_NOTES)
        engine.play_note(60)
        self.assertTrue(self.render(engine).any())
        voice = engine._voices[0]
        self.assertIs(voice.samples, engine._samples[(69, "keys")])

    def test_notes_finish_and_unknown_notes_do_not_play(self):
        engine = self.make_engine()
        engine.play_note(1)
        engine.play_note(60, instrument="missing")
        engine.play_note(60, velocity=float("nan"))
        self.assertFalse(self.render(engine).any())
        engine.play_note(60, "marimba")
        self.assertTrue(self.render(engine, frames=24000).any())
        self.assertFalse(self.render(engine).any())

    def test_start_failure_keeps_game_usable(self):
        engine = AudioEngine([60])
        with patch("audio_engine.importlib.import_module", side_effect=ImportError("unavailable")):
            self.assertFalse(engine.start())
        self.assertFalse(engine.enabled)
        self.assertIn("unavailable", engine.error_message)
        engine.play_note(60)
        engine.close()
        engine.close()

    def test_no_output_channels_fails_gracefully(self):
        fake_device = SimpleNamespace(query_devices=lambda **kwargs: {
            "max_output_channels": 0, "default_samplerate": 24000,
        })
        engine = AudioEngine([60])
        with patch("audio_engine.importlib.import_module", return_value=fake_device):
            self.assertFalse(engine.start())
        self.assertIn("No audio output", engine.error_message)

    def test_stream_is_closed_after_startup_failure(self):
        stream = FakeStream(finished_callback=lambda: None)
        fake_device = SimpleNamespace(
            query_devices=lambda **kwargs: {"max_output_channels": 2, "default_samplerate": 24000},
            OutputStream=lambda **kwargs: stream,
            CallbackAbort=RuntimeError,
        )
        engine = AudioEngine([60])
        with patch("audio_engine.importlib.import_module", return_value=fake_device):
            with patch.object(stream, "start", side_effect=RuntimeError("device busy")):
                self.assertFalse(engine.start())
        self.assertTrue(stream.closed)
        self.assertFalse(engine.enabled)
        self.assertIn("device busy", engine.error_message)

    def test_unexpected_stop_disables_audio_and_close_is_repeatable(self):
        engine = self.make_engine()
        stream = engine._stream
        stream.abort()
        self.assertFalse(engine.enabled)
        self.assertIn("stopped", engine.error_message)
        engine.close()
        engine.close()
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
