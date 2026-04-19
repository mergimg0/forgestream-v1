import asyncio

from forgestream.audio.source import AudioSource


class TestAudioSource:
    def test_defaults(self):
        source = AudioSource()
        assert source.sample_rate == 16000
        assert source.channels == 1
        assert source.chunk_duration == 0.5
        assert source.is_active is False

    def test_chunk_size_bytes(self):
        source = AudioSource()
        # 16kHz * 2 bytes/sample * 1 channel * 0.5s = 16000 bytes
        assert source.chunk_size_bytes == 16000

    def test_cannot_iterate_base_class(self):
        source = AudioSource()
        try:
            gen = source.chunks()
            asyncio.get_event_loop().run_until_complete(gen.__anext__())
            assert False, "Should have raised"
        except (NotImplementedError, StopAsyncIteration):
            pass
