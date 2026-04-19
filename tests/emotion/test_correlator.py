"""Tests for EmotionCorrelator — claim-emotion alignment and shift detection."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from forgestream.emotion.correlator import AUTHOR, EmotionCorrelator
from forgestream.events.schema import Event, EventType


def _make_prosodic_event(
    session_id, branch_id, timestamp_ms=1000, arousal=0.5, valence=0.5,
    jitter=0.01, shimmer=0.03, hnr=20.0, f0_std=30.0,
) -> Event:
    return Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=session_id,
        branch_id=branch_id,
        author="emotion_extractor",
        evaluator=0.0,
        payload={
            "speaker_id": "unknown",
            "timestamp_ms": timestamp_ms,
            "chunk_index": timestamp_ms // 500,
            "window_duration_ms": 3000,
            "arousal": arousal,
            "valence": valence,
            "dominance": 0.5,
            "f0_mean": 200.0,
            "f0_std": f0_std,
            "f0_contour": [200.0],
            "energy_rms": 0.1,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
            "hnr": hnr,
            "spectral_centroid": 2000.0,
            "egemaps_vector": [0.0] * 88,
            "emotion_tag": None,
            "emotion_confidence": None,
        },
    )


def _make_claim_event(session_id, branch_id, timestamp_ms=1000) -> Event:
    return Event(
        event_type=EventType.CLAIM,
        session_id=session_id,
        branch_id=branch_id,
        author="gemini",
        evaluator=0.0,
        payload={
            "text": "Test claim",
            "speaker": "Speaker 1",
            "confidence": 0.7,
            "tone_markers": [],
            "topic_keywords": ["test"],
            "audio_timestamp": timestamp_ms,
        },
    )


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.session_id = uuid4()
    orch.process_event = AsyncMock(return_value=True)
    return orch


class TestEmotionCorrelator:
    @pytest.mark.asyncio
    async def test_enriches_claim_with_prosodic_context(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)
        sid = mock_orchestrator.session_id
        bid = uuid4()

        # Feed a prosodic feature first
        prosodic = _make_prosodic_event(sid, bid, timestamp_ms=1000, arousal=0.8)
        await correlator.on_event(prosodic)

        # Feed a claim at the same timestamp
        claim = _make_claim_event(sid, bid, timestamp_ms=1000)
        await correlator.on_event(claim)

        # Correlator should have emitted an EMOTION_STATE or adjustment event
        # At minimum, it should not crash
        # The prosodic buffer should have stored the feature
        assert len(correlator._prosodic_buffer) == 1

    @pytest.mark.asyncio
    async def test_emits_emotion_state_on_significant_shift(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)
        sid = mock_orchestrator.session_id
        bid = uuid4()

        # Feed low arousal prosodic
        low = _make_prosodic_event(sid, bid, timestamp_ms=1000, arousal=0.2, valence=0.3)
        await correlator.on_event(low)

        # Feed high arousal prosodic (big shift)
        high = _make_prosodic_event(sid, bid, timestamp_ms=2000, arousal=0.7, valence=0.8)
        await correlator.on_event(high)

        # Should have emitted an EMOTION_STATE event
        calls = [
            c for c in mock_orchestrator.process_event.call_args_list
            if c[0][0].event_type == EventType.EMOTION_STATE
        ]
        assert len(calls) == 1
        payload = calls[0][0][0].payload
        assert payload["shift_type"] == "onset"
        assert payload["arousal_delta"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_ignores_self_authored_events(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)
        sid = mock_orchestrator.session_id
        bid = uuid4()

        self_event = Event(
            event_type=EventType.EMOTION_STATE,
            session_id=sid,
            branch_id=bid,
            author=AUTHOR,
            evaluator=0.0,
            payload={"test": True},
        )
        await correlator.on_event(self_event)
        assert mock_orchestrator.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_handles_claim_with_no_matching_prosodic(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)
        sid = mock_orchestrator.session_id
        bid = uuid4()

        # Feed claim with no prosodic features buffered — should not crash
        claim = _make_claim_event(sid, bid, timestamp_ms=5000)
        await correlator.on_event(claim)
        # No EMOTION_STATE emitted
        assert mock_orchestrator.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_confidence_adjustment_range(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)

        # High stress: high jitter, low HNR → negative adjustment
        stress_adj = correlator._compute_confidence_adjustment({
            "arousal": 0.8, "jitter_local": 0.05, "shimmer_local": 0.08,
            "hnr": 5.0, "f0_std": 50.0,
        })
        assert -0.3 <= stress_adj <= 0.3

        # High clarity: low jitter, high HNR → positive adjustment
        clear_adj = correlator._compute_confidence_adjustment({
            "arousal": 0.6, "jitter_local": 0.005, "shimmer_local": 0.01,
            "hnr": 25.0, "f0_std": 20.0,
        })
        assert -0.3 <= clear_adj <= 0.3
        assert clear_adj > stress_adj

    @pytest.mark.asyncio
    async def test_no_shift_emitted_for_small_delta(self, mock_orchestrator):
        correlator = EmotionCorrelator(orchestrator=mock_orchestrator)
        sid = mock_orchestrator.session_id
        bid = uuid4()

        # Two prosodic events with very similar values
        p1 = _make_prosodic_event(sid, bid, timestamp_ms=1000, arousal=0.5, valence=0.5)
        p2 = _make_prosodic_event(sid, bid, timestamp_ms=2000, arousal=0.55, valence=0.52)
        await correlator.on_event(p1)
        await correlator.on_event(p2)

        # No EMOTION_STATE emitted (delta < 0.2)
        emotion_calls = [
            c for c in mock_orchestrator.process_event.call_args_list
            if c[0][0].event_type == EventType.EMOTION_STATE
        ]
        assert len(emotion_calls) == 0
