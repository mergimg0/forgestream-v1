"""MicrophoneSource -- capture live audio from system microphone."""

from __future__ import annotations

import asyncio
import queue
from typing import Any, AsyncIterator

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

from .source import AudioSource


class MicrophoneSource(AudioSource):
    """Captures audio from the system microphone via sounddevice.

    Uses a callback-based InputStream that fills a queue.
    The chunks() async generator reads from the queue.
    """

    def __init__(self, device: int | None = None) -> None:
        super().__init__()
        self.device = device
        self._stream = None
        self._queue: queue.Queue[bytes] = queue.Queue()

    @staticmethod
    def list_input_devices() -> list[dict[str, Any]]:
        """List available audio input devices."""
        if not HAS_SOUNDDEVICE:
            return []
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Called by sounddevice when audio data is available."""
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        self._queue.put(audio_int16.tobytes())

    async def start(self) -> None:
        """Open the audio input stream."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not installed. pip install sounddevice")

        chunk_samples = int(self.sample_rate * self.chunk_duration)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_samples,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._active = True

    async def stop(self) -> None:
        """Close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks from the microphone."""
        while self._active:
            try:
                chunk = self._queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                await asyncio.sleep(0.05)
