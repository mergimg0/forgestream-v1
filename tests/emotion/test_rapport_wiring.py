"""Test RapportEngine is wired into the live pipeline."""

from unittest.mock import MagicMock

from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


def test_attach_rapport_engine():
    config = ForgeStreamConfig(rapport_enabled=True)
    orch = Orchestrator(config=config)
    engine = orch.attach_rapport_engine()
    assert engine is not None
    assert len(orch.event_bus._subscribers) >= 1


def test_live_stream_wires_rapport_engine():
    config = ForgeStreamConfig(emotion_enabled=True, rapport_enabled=True)
    orch = Orchestrator(config=config)
    source = MagicMock()
    stream = GeminiLiveStream(config, orch, source)
    assert stream.rapport_engine is not None


def test_rapport_disabled_skips_wiring():
    config = ForgeStreamConfig(emotion_enabled=True, rapport_enabled=False)
    orch = Orchestrator(config=config)
    source = MagicMock()
    stream = GeminiLiveStream(config, orch, source)
    assert stream.rapport_engine is None
