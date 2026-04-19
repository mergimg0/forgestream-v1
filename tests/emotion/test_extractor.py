"""Tests for EmotionExtractor — the sliding-window feature extraction orchestrator."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from forgestream.emotion.buffer import AudioRingBuffer
from forgestream.emotion.extractor import EmotionExtractor
from forgestream.events.schema import EventType


def _make_sine_chunk(freq: float = 200.0) -> bytes:
    """Generate a 0.5s PCM chunk of a sine wave."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    signal = (10000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal.tobytes()


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.session_id = uuid4()
    orch.process_event = AsyncMock(return_value=True)
    return orch


@pytest.fixture
def audio_buffer():
    return AudioRingBuffer(capacity_seconds=10.0)


class TestEmotionExtractor:
    @pytest.mark.asyncio
    async def test_emits_prosodic_feature_event_after_stride(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=1.5,   # 3 chunks
            stride_seconds=1.0,   # every 2 chunks
        )
        chunk = _make_sine_chunk(200.0)

        # Feed 2 chunks (= 1 stride)
        await extractor.process_chunk(chunk, chunk_index=0)
        assert mock_orchestrator.process_event.call_count == 0

        await extractor.process_chunk(chunk, chunk_index=1)
        assert mock_orchestrator.process_event.call_count == 1

        event = mock_orchestrator.process_event.call_args[0][0]
        assert event.event_type == EventType.PROSODIC_FEATURE
        assert event.author == "emotion_extractor"
        assert "f0_mean" in event.payload
        assert "arousal" in event.payload

    @pytest.mark.asyncio
    async def test_does_not_emit_before_stride(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=1.5,
            stride_seconds=1.0,
        )
        chunk = _make_sine_chunk(200.0)
        await extractor.process_chunk(chunk, chunk_index=0)
        assert mock_orchestrator.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_payload_contains_all_required_fields(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=0.5,  # 1 chunk window
            stride_seconds=0.5,  # emit every chunk
        )
        chunk = _make_sine_chunk(200.0)
        await extractor.process_chunk(chunk, chunk_index=0)

        event = mock_orchestrator.process_event.call_args[0][0]
        payload = event.payload
        required_fields = [
            "speaker_id", "timestamp_ms", "chunk_index", "window_duration_ms",
            "f0_mean", "f0_std", "f0_contour", "energy_rms",
            "jitter_local", "shimmer_local", "hnr", "spectral_centroid",
            "arousal", "valence", "dominance",
        ]
        for f in required_fields:
            assert f in payload, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_graceful_with_silence(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=0.5,
            stride_seconds=0.5,
        )
        silence = b"\x00\x00" * 8000  # 0.5s silence
        await extractor.process_chunk(silence, chunk_index=0)

        event = mock_orchestrator.process_event.call_args[0][0]
        assert event.payload["f0_mean"] == 0.0
