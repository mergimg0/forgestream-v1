"""Tests for end-to-end emotion pipeline wiring — EventBus subscriptions."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator


class TestEmotionEventBusWiring:
    @pytest.mark.asyncio
    async def test_attach_emotion_correlator(self):
        config = ForgeStreamConfig(emotion_enabled=True)
        orchestrator = Orchestrator(config=config)
        correlator = orchestrator.attach_emotion_correlator()
        assert correlator is not None
        # Correlator should be subscribed to EventBus
        assert len(orchestrator.event_bus._subscribers) >= 1

    @pytest.mark.asyncio
    async def test_attach_dynamics_engine(self):
        config = ForgeStreamConfig(emotion_enabled=True)
        orchestrator = Orchestrator(config=config)
        engine = orchestrator.attach_dynamics_engine()
        assert engine is not None
        assert len(orchestrator.event_bus._subscribers) >= 1

    @pytest.mark.asyncio
    async def test_prosodic_event_reaches_correlator(self):
        config = ForgeStreamConfig(emotion_enabled=True)
        orchestrator = Orchestrator(config=config)
        correlator = orchestrator.attach_emotion_correlator()

        prosodic = Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=orchestrator.session_id,
            branch_id=uuid4(),
            author="emotion_extractor",
            evaluator=0.0,
            payload={
                "speaker_id": "sp0", "timestamp_ms": 1000,
                "arousal": 0.7, "valence": 0.5, "dominance": 0.5,
            },
        )
        await orchestrator.process_event(prosodic)
        # Correlator should have buffered the prosodic event
        assert len(correlator._prosodic_buffer) == 1

    @pytest.mark.asyncio
    async def test_prosodic_event_reaches_dynamics_engine(self):
        config = ForgeStreamConfig(emotion_enabled=True)
        orchestrator = Orchestrator(config=config)
        engine = orchestrator.attach_dynamics_engine()

        prosodic = Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=orchestrator.session_id,
            branch_id=uuid4(),
            author="emotion_extractor",
            evaluator=0.0,
            payload={
                "speaker_id": "sp0", "timestamp_ms": 1000,
                "f0_mean": 200.0, "energy_rms": 0.1,
                "arousal": 0.5, "f0_std": 20.0,
            },
        )
        await orchestrator.process_event(prosodic)
        # DynamicsEngine should have accumulated the feature
        assert "sp0" in engine._speaker_series.speaker_ids()

    @pytest.mark.asyncio
    async def test_live_stream_wires_all_emotion_subscribers(self):
        config = ForgeStreamConfig(emotion_enabled=True)
        orchestrator = Orchestrator(config=config)
        source = MagicMock()

        from forgestream.live_stream import GeminiLiveStream
        stream = GeminiLiveStream(config, orchestrator, source)

        # Should have correlator + dynamics engine subscribed
        assert stream.emotion_correlator is not None
        assert stream.dynamics_engine is not None
        # EventBus should have at least 2 emotion subscribers
        assert len(orchestrator.event_bus._subscribers) >= 2

    @pytest.mark.asyncio
    async def test_disabled_emotion_skips_subscribers(self):
        config = ForgeStreamConfig(emotion_enabled=False)
        orchestrator = Orchestrator(config=config)
        source = MagicMock()

        from forgestream.live_stream import GeminiLiveStream
        stream = GeminiLiveStream(config, orchestrator, source)

        assert stream.emotion_correlator is None
        assert stream.dynamics_engine is None
