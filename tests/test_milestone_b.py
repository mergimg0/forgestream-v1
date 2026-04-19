"""Milestone B integration tests -- audio, streaming, agents, GRPO."""

import math
import struct
import tempfile
import wave
from pathlib import Path
from uuid import uuid4

from forgestream.audio.file_replay import FileReplaySource
from forgestream.audio.microphone import MicrophoneSource
from forgestream.audio.source import AudioSource
from forgestream.audio.system_audio import SystemAudioSource
from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.live_stream import GeminiLiveStream
from forgestream.agent_dispatcher import AgentDispatcher
from forgestream.orchestrator import Orchestrator
from forgestream.post_meeting import PostMeetingSynthesis


def _create_test_wav(path: Path, duration_s: float = 1.0) -> None:
    n_samples = int(16000 * duration_s)
    samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


class TestAudioSourceUniformInterface:
    async def test_file_replay_chunk_size(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _create_test_wav(Path(f.name))

        source = FileReplaySource(f.name, speed=0)
        await source.start()

        async for chunk in source.chunks():
            assert len(chunk) == source.chunk_size_bytes
            break

        await source.stop()
        Path(f.name).unlink()

    def test_all_sources_share_base(self):
        assert issubclass(FileReplaySource, AudioSource)
        assert issubclass(MicrophoneSource, AudioSource)
        assert issubclass(SystemAudioSource, AudioSource)

    def test_all_sources_same_sample_rate(self):
        f = FileReplaySource.__new__(FileReplaySource)
        m = MicrophoneSource.__new__(MicrophoneSource)
        s = SystemAudioSource.__new__(SystemAudioSource)
        assert f.sample_rate == m.sample_rate == s.sample_rate == 16000

    def test_all_sources_same_chunk_size(self):
        f = FileReplaySource.__new__(FileReplaySource)
        m = MicrophoneSource.__new__(MicrophoneSource)
        assert f.chunk_size_bytes == m.chunk_size_bytes == 16000


class TestGeminiLiveStreamUnit:
    def test_stream_connects_source_to_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = FileReplaySource.__new__(FileReplaySource)
        source._active = False

        stream = GeminiLiveStream(
            config=config, orchestrator=orch,
            audio_source=source, mode="collaborative"
        )
        assert stream.orchestrator is orch
        assert stream.audio_source is source


class TestAgentDispatcherUnit:
    def test_dispatcher_integrates_with_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)
        assert dispatcher.orchestrator is orch

    def test_prompt_includes_context(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        from forgestream.agents.registry import AgentType
        prompt = dispatcher.build_prompt(
            AgentType.RESEARCH, "Test query", ["context claim 1"]
        )
        assert "context claim 1" in prompt


class TestPostMeetingIntegration:
    def test_full_synthesis_pipeline(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.meetings_dir = f"{tmpdir}/meetings"
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)

            sid = uuid4()
            bid = uuid4()
            events = [
                Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                      author="gemini", evaluator=0.4,
                      payload={"text": "test", "topic_keywords": ["A"]}),
                Event(event_type=EventType.VERIFIED_FINDING, session_id=sid, branch_id=bid,
                      author="research", evaluator=0.5,
                      payload={"finding": "result", "sources": ["x"], "confidence": 0.9}),
            ]

            report = synthesis.generate_report(events, "Integration Test")
            assert "Integration Test" in report

            path = synthesis.save_report(events, "integration-test")
            assert Path(path).exists()

            weights = synthesis.tune_weights(events, human_score=0.7)
            synthesis.save_weights(weights, meeting_count=1)

            loaded = synthesis.load_weights()
            assert abs(sum(loaded.values()) - 1.0) < 0.01
