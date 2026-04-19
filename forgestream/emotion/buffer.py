"""AudioRingBuffer — stores audio chunks for parallel emotion extraction.

Lock-free ring buffer. The Gemini send loop writes chunks; the EmotionExtractor
reads them. Old chunks are evicted when capacity is exceeded. Chunk indices
are monotonically increasing and never reused.
"""

from __future__ import annotations

from collections import deque


class AudioRingBuffer:
    """Ring buffer for PCM 16kHz mono int16 audio chunks.

    Parameters:
        capacity_seconds: Maximum audio duration to retain.
        sample_rate: Audio sample rate in Hz.
        chunk_duration: Duration of each chunk in seconds.
    """

    def __init__(
        self,
        capacity_seconds: float = 30.0,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._bytes_per_chunk = int(sample_rate * 2 * chunk_duration)
        max_chunks = int(capacity_seconds / chunk_duration)
        self._max_chunks = max(1, max_chunks)

        self._chunks: deque[tuple[int, bytes]] = deque(
            maxlen=self._max_chunks
        )
        self._next_index: int = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently stored."""
        return len(self._chunks)

    def write_chunk(self, chunk: bytes) -> int:
        """Write a chunk and return its index.

        If the buffer is full, the oldest chunk is evicted.
        """
        idx = self._next_index
        self._next_index += 1
        self._chunks.append((idx, chunk))
        return idx

    def read_chunk(self, chunk_index: int) -> bytes | None:
        """Read a chunk by its index. Returns None if evicted or not found."""
        for idx, data in self._chunks:
            if idx == chunk_index:
                return data
        return None

    def read_window(self, duration_seconds: float) -> bytes:
        """Read the most recent N seconds of audio as contiguous bytes."""
        n_chunks = int(duration_seconds / self._chunk_duration)
        n_chunks = min(n_chunks, len(self._chunks))
        if n_chunks == 0:
            return b""
        recent = list(self._chunks)[-n_chunks:]
        return b"".join(data for _, data in recent)

    def read_window_numpy(self, duration_seconds: float) -> "numpy.ndarray":
        """Read the most recent N seconds as a numpy int16 array."""
        import numpy

        raw = self.read_window(duration_seconds)
        if not raw:
            return numpy.array([], dtype=numpy.int16)
        return numpy.frombuffer(raw, dtype=numpy.int16)

    def chunk_timestamp_ms(self, chunk_index: int) -> int:
        """Convert a chunk index to its start timestamp in milliseconds."""
        return int(chunk_index * self._chunk_duration * 1000)

    def latest_chunk_index(self) -> int | None:
        """Return the index of the most recently written chunk, or None."""
        if not self._chunks:
            return None
        return self._chunks[-1][0]
