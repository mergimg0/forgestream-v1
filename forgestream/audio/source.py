"""AudioSource abstract base class -- unified interface for all audio inputs."""

from __future__ import annotations

from typing import AsyncIterator


class AudioSource:
    """Abstract base for audio input sources.

    All sources yield PCM 16kHz mono int16 audio chunks.
    Chunk size: 16000 samples/sec * 2 bytes * 0.5s = 16000 bytes per chunk.
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5  # seconds

    def __init__(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def chunk_size_bytes(self) -> int:
        return int(self.sample_rate * 2 * self.channels * self.chunk_duration)

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def chunks(self) -> AsyncIterator[bytes]:
        raise NotImplementedError("Subclasses must implement chunks()")
        yield  # make it a generator
