"""Test that GeminiLiveStream wires the emotion pipeline correctly."""

from unittest.mock import MagicMock
from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


def test_live_stream_creates_audio_buffer_when_emotion_enabled():
    config = ForgeStreamConfig(emotion_enabled=True)
    orchestrator = Orchestrator(config=config)
    source = MagicMock()

    stream = GeminiLiveStream(config, orchestrator, source)
    assert stream.audio_buffer is not None
    assert stream.emotion_extractor is not None


def test_live_stream_skips_emotion_when_disabled():
    config = ForgeStreamConfig(emotion_enabled=False)
    orchestrator = Orchestrator(config=config)
    source = MagicMock()

    stream = GeminiLiveStream(config, orchestrator, source)
    assert stream.audio_buffer is None
    assert stream.emotion_extractor is None
