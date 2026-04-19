"""GeminiLiveStream tests."""

from forgestream.audio.source import AudioSource
from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream, MODE_INSTRUCTIONS
from forgestream.orchestrator import Orchestrator


class MockAudioSource(AudioSource):
    def __init__(self, num_chunks: int = 3) -> None:
        super().__init__()
        self._num_chunks = num_chunks

    async def chunks(self):
        for _ in range(self._num_chunks):
            yield b"\x00" * self.chunk_size_bytes
        self._active = False


class TestGeminiLiveStream:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = MockAudioSource()

        stream = GeminiLiveStream(
            config=config, orchestrator=orch,
            audio_source=source, mode="collaborative",
        )
        assert stream.mode == "collaborative"
        assert stream.audio_source is source

    def test_mode_instructions_exist(self):
        assert "extract" in MODE_INSTRUCTIONS
        assert "collaborative" in MODE_INSTRUCTIONS
        assert "knowledge" in MODE_INSTRUCTIONS

    def test_parse_jsonl_valid(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())

        text = '{"text": "hello", "confidence": 0.9, "topic_keywords": ["test"]}\n{"text": "world", "confidence": 0.8, "topic_keywords": ["test2"]}'
        claims = stream._parse_jsonl(text)
        assert len(claims) == 2
        assert claims[0]["text"] == "hello"

    def test_parse_jsonl_handles_garbage(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())

        text = '```json\n{"text": "hello", "confidence": 0.9}\nnot json\n```'
        claims = stream._parse_jsonl(text)
        assert len(claims) == 1

    def test_set_mode(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())
        stream.set_mode("knowledge")
        assert stream.mode == "knowledge"
