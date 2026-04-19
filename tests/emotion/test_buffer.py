"""Tests for AudioRingBuffer — the tee between AudioSource and EmotionExtractor."""

import pytest

from forgestream.emotion.buffer import AudioRingBuffer


class TestAudioRingBuffer:
    def test_write_and_read_single_chunk(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk = b"\x00\x01" * 8000  # 0.5s at 16kHz 16-bit mono = 16000 bytes
        idx = buf.write_chunk(chunk)
        assert idx == 0
        assert buf.read_chunk(0) == chunk

    def test_sequential_writes_increment_index(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk = b"\x00" * 16000
        idx0 = buf.write_chunk(chunk)
        idx1 = buf.write_chunk(chunk)
        idx2 = buf.write_chunk(chunk)
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_read_nonexistent_chunk_returns_none(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        assert buf.read_chunk(0) is None
        assert buf.read_chunk(99) is None

    def test_old_chunks_evicted_after_capacity(self):
        # 2 seconds capacity, 0.5s chunks = 4 chunks max
        buf = AudioRingBuffer(capacity_seconds=2.0, sample_rate=16000)
        chunk_size = 16000  # 0.5s
        chunks = [bytes([i % 256]) * chunk_size for i in range(6)]
        for c in chunks:
            buf.write_chunk(c)
        # Chunks 0, 1 should be evicted; chunks 2-5 should remain
        assert buf.read_chunk(0) is None
        assert buf.read_chunk(1) is None
        assert buf.read_chunk(2) is not None
        assert buf.read_chunk(5) is not None

    def test_read_window_returns_recent_audio(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk_size = 16000
        for i in range(6):
            buf.write_chunk(bytes([i]) * chunk_size)
        # Read last 1.5 seconds = 3 chunks
        window = buf.read_window(duration_seconds=1.5)
        assert len(window) == chunk_size * 3
        # Should contain the 3 most recent chunks (indices 3, 4, 5)
        assert window[:chunk_size] == bytes([3]) * chunk_size
        assert window[-chunk_size:] == bytes([5]) * chunk_size

    def test_read_window_clamps_to_available(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        buf.write_chunk(b"\x42" * 16000)
        # Request more than available
        window = buf.read_window(duration_seconds=10.0)
        assert len(window) == 16000  # only 1 chunk available

    def test_chunk_timestamp_ms(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        buf.write_chunk(b"\x00" * 16000)  # chunk 0: 0-500ms
        buf.write_chunk(b"\x00" * 16000)  # chunk 1: 500-1000ms
        buf.write_chunk(b"\x00" * 16000)  # chunk 2: 1000-1500ms
        assert buf.chunk_timestamp_ms(0) == 0
        assert buf.chunk_timestamp_ms(1) == 500
        assert buf.chunk_timestamp_ms(2) == 1000

    def test_chunk_count(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        assert buf.chunk_count == 0
        buf.write_chunk(b"\x00" * 16000)
        assert buf.chunk_count == 1
        buf.write_chunk(b"\x00" * 16000)
        assert buf.chunk_count == 2

    def test_read_window_as_numpy(self):
        numpy = pytest.importorskip("numpy")
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        # Write a known signal: 0.5s of silence
        buf.write_chunk(b"\x00\x00" * 8000)
        arr = buf.read_window_numpy(duration_seconds=0.5)
        assert arr.dtype == numpy.int16
        assert len(arr) == 8000  # 8000 samples at 16kHz for 0.5s
        assert arr.sum() == 0
