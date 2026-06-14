"""Capture system audio from BlackHole and emit overlapping mono windows.

macOS hides system audio from apps, so you route playback through a BlackHole
loopback device (see README). This module reads that device and yields fixed
windows (default 10s) that overlap by `overlap_sec` (default 1s). The overlap
gives Whisper a little context at each boundary; the consumer de-duplicates the
shared region using word timestamps.
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

TARGET_SR = 16_000  # Whisper expects 16 kHz mono


def list_devices() -> str:
    return str(sd.query_devices())


def find_blackhole() -> int | None:
    """Return the input-device index whose name contains 'blackhole'."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return idx
    return None


class AudioWindower:
    def __init__(
        self,
        device: int | str | None = None,
        window_sec: float = 10.0,
        overlap_sec: float = 1.0,
        target_sr: int = TARGET_SR,
    ):
        if device is None:
            device = find_blackhole()
            if device is None:
                raise RuntimeError(
                    "No BlackHole input device found. Run with --list-devices "
                    "to see options, or pass --device. See README for setup."
                )
        self.device = device
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.target_sr = target_sr

        info = sd.query_devices(device)
        self.src_sr = int(info["default_samplerate"])
        self.channels = min(2, int(info["max_input_channels"])) or 1

        self.win_samples = int(self.window_sec * self.src_sr)
        self.step_samples = int((self.window_sec - self.overlap_sec) * self.src_sr)

        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            print(f"[audio] {status}")
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        self._q.put(mono.copy())

    def _to_16k(self, chunk: np.ndarray) -> np.ndarray:
        if self.src_sr == self.target_sr:
            return chunk.astype(np.float32)
        out = resample_poly(chunk, self.target_sr, self.src_sr)
        return out.astype(np.float32)

    def windows(self):
        """Yield (audio_16k: np.ndarray, is_first: bool) windows forever."""
        buf = np.zeros(0, dtype=np.float32)
        first = True
        stream = sd.InputStream(
            samplerate=self.src_sr,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )
        with stream:
            print(
                f"[audio] capturing from device {self.device} "
                f"@ {self.src_sr} Hz, {self.window_sec:.0f}s windows / "
                f"{self.overlap_sec:.0f}s overlap"
            )
            while not self._stop.is_set():
                try:
                    buf = np.concatenate([buf, self._q.get(timeout=1.0)])
                except queue.Empty:
                    continue
                while len(buf) >= self.win_samples:
                    window = buf[: self.win_samples]
                    yield self._to_16k(window), first
                    first = False
                    buf = buf[self.step_samples :]

    def stop(self) -> None:
        self._stop.set()
