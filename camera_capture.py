"""Non-blocking latest-frame camera capture for Air Rhythm."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    """One immutable snapshot handed from the camera worker to the app."""

    image: np.ndarray
    sequence: int
    camera_read_ms: float
    main_thread_wait_ms: float
    skipped_since_previous: int


class LatestFrameCamera:
    """Read frames continuously and expose only the freshest complete frame.

    OpenCV's camera read normally blocks the same thread that performs hand
    inference and rendering.  This worker lets the camera wait in parallel.
    Older unread frames are intentionally replaced so input stays current.
    """

    def __init__(self, camera) -> None:
        self._camera = camera
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._failed = False
        self._latest_image: np.ndarray | None = None
        self._latest_sequence = 0
        self._latest_read_ms = 0.0

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failed

    def start(self) -> "LatestFrameCamera":
        """Start the single camera-reading worker."""
        with self._condition:
            if self._closed:
                raise RuntimeError("camera stream is closed")
            if self._running:
                return self
            self._running = True
            self._failed = False
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="air-rhythm-camera",
                daemon=True,
            )
            self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                if not self._running:
                    return

            read_started_at = time.perf_counter()
            try:
                success, image = self._camera.read()
            except Exception:
                with self._condition:
                    if self._running:
                        self._failed = True
                        self._running = False
                        self._condition.notify_all()
                return
            camera_read_ms = (time.perf_counter() - read_started_at) * 1000.0

            with self._condition:
                if not self._running:
                    return
                if not success or image is None:
                    self._failed = True
                    self._running = False
                    self._condition.notify_all()
                    return
                self._latest_image = image
                self._latest_sequence += 1
                self._latest_read_ms = max(0.0, camera_read_ms)
                self._condition.notify_all()

    def read_latest(
        self,
        *,
        after_sequence: int = 0,
        timeout: float = 1.0,
    ) -> CapturedFrame | None:
        """Wait for a newer frame and return the freshest one available."""
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        try:
            wait_timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a non-negative finite number") from error
        if not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError("timeout must be a non-negative finite number")

        wait_started_at = time.perf_counter()
        deadline = wait_started_at + wait_timeout
        with self._condition:
            while (
                self._latest_sequence <= after_sequence
                and self._running
                and not self._failed
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

            main_thread_wait_ms = (
                time.perf_counter() - wait_started_at
            ) * 1000.0
            if (
                self._latest_image is None
                or self._latest_sequence <= after_sequence
            ):
                return None
            sequence = self._latest_sequence
            return CapturedFrame(
                image=self._latest_image,
                sequence=sequence,
                camera_read_ms=self._latest_read_ms,
                main_thread_wait_ms=max(0.0, main_thread_wait_ms),
                skipped_since_previous=max(0, sequence - after_sequence - 1),
            )

    def close(self) -> None:
        """Stop capture, release the device, and wait briefly for the worker."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._running = False
            self._condition.notify_all()

        self._camera.release()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


__all__ = ["CapturedFrame", "LatestFrameCamera"]
