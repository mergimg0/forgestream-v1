"""FileReplaySource -- replay pre-recorded audio files as PCM chunks."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import soundfile as sf

from .source import AudioSource


def _natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


class FileReplaySource(AudioSource):
    """Replays audio files as PCM 16kHz mono chunks.

    Accepts a single file or a folder of audio files (natural sort order).
    speed=1.0 is real-time, speed=0 is as fast as possible.
    """

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

    def __init__(self, path: str, speed: float = 1.0) -> None:
        super().__init__()
        self.path = Path(path)
        self.speed = speed
        self._files: list[Path] = []
        self._resolve_files()

    def _resolve_files(self) -> None:
        if self.path.is_dir():
            self._files = sorted(
                [f for f in self.path.iterdir() if f.suffix.lower() in self.AUDIO_EXTENSIONS],
                key=_natural_sort_key,
            )
        elif self.path.is_file():
            self._files = [self.path]

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks from all audio files in sequence."""
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        for audio_file in self._files:
            try:
                data, sr = sf.read(audio_file, dtype="int16", always_2d=True)
            except Exception:
                # Use pydub for formats soundfile can't handle (m4a, mp3)
                from pydub import AudioSegment
                seg = AudioSegment.from_file(str(audio_file))
                seg = seg.set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(2)
                data = np.frombuffer(seg.raw_data, dtype=np.int16).reshape(-1, 1)
                sr = self.sample_rate

            # Resample if needed
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                indices = np.arange(0, len(data), 1 / ratio).astype(int)
                indices = indices[indices < len(data)]
                data = data[indices]

            # Convert to mono if stereo
            if data.ndim > 1 and data.shape[1] > 1:
                data = data.mean(axis=1).astype(np.int16)
            elif data.ndim > 1:
                data = data[:, 0]

            # Yield in chunks
            for i in range(0, len(data), chunk_samples):
                chunk_data = data[i : i + chunk_samples]
                if len(chunk_data) < chunk_samples:
                    chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)))
                yield chunk_data.astype(np.int16).tobytes()

                if self.speed > 0:
                    await asyncio.sleep(self.chunk_duration / self.speed)

        self._active = False
