import math
import struct
import tempfile
import wave
from pathlib import Path

from forgestream.audio.file_replay import FileReplaySource
from forgestream.audio.source import AudioSource


def create_test_wav(path: Path, duration_s: float = 2.0, sample_rate: int = 16000) -> None:
    """Create a test WAV file with a sine wave."""
    n_samples = int(sample_rate * duration_s)
    samples = [int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


class TestFileReplaySource:
    def test_is_audio_source(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            create_test_wav(Path(f.name))
            source = FileReplaySource(f.name)
            assert isinstance(source, AudioSource)

    async def test_yields_chunks(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
            create_test_wav(path, duration_s=1.5)

        source = FileReplaySource(str(path), speed=0)
        await source.start()

        chunks = []
        async for chunk in source.chunks():
            chunks.append(chunk)
            if len(chunks) > 10:
                break

        await source.stop()
        path.unlink()

        assert len(chunks) >= 2
        assert all(len(c) == source.chunk_size_bytes for c in chunks)

    async def test_folder_of_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                create_test_wav(Path(tmpdir) / f"part_{i}.wav", duration_s=1.0)

            source = FileReplaySource(tmpdir, speed=0)
            await source.start()

            chunks = []
            async for chunk in source.chunks():
                chunks.append(chunk)

            await source.stop()

            # 3 files * 1.0s / 0.5s = 6 chunks
            assert len(chunks) == 6

    async def test_not_active_after_exhausted(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
            create_test_wav(path, duration_s=0.5)

        source = FileReplaySource(str(path), speed=0)
        await source.start()

        async for _ in source.chunks():
            pass

        assert source.is_active is False
        path.unlink()

    def test_speed_multiplier(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            create_test_wav(Path(f.name))
            source = FileReplaySource(f.name, speed=2.0)
            assert source.speed == 2.0
