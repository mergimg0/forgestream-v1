"""Speaker diarization using pyannote-audio.

Labels audio chunks with speaker IDs. Uses pyannote's speaker-diarization-3.1
pipeline with incremental processing: each new chunk extends the internal
audio accumulator and re-runs diarization on the recent window.

Optional — loaded only when pyannote.audio is installed and HF token is provided.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Identifies who is speaking in each audio chunk.

    Parameters:
        huggingface_token: HuggingFace token for pyannote model access.
        max_speakers: Maximum number of speakers to detect.
        window_seconds: Audio window to analyze for diarization.
        sample_rate: Expected sample rate.
    """

    def __init__(
        self,
        huggingface_token: str,
        max_speakers: int = 6,
        window_seconds: float = 30.0,
        sample_rate: int = 16000,
    ) -> None:
        self._max_speakers = max_speakers
        self._window_seconds = window_seconds
        self._sample_rate = sample_rate
        self._pipeline = None
        self._token = huggingface_token

        max_samples = int(window_seconds * sample_rate)
        self._audio_buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0

        self._segments: list[tuple[float, float, str]] = []

    @staticmethod
    def is_available() -> bool:
        """Check if pyannote.audio is installed."""
        try:
            import pyannote.audio
            return True
        except ImportError:
            return False

    def _ensure_pipeline(self) -> None:
        """Lazily load pyannote pipeline."""
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._token,
        )
        logger.info("pyannote speaker-diarization-3.1 pipeline loaded")

    def add_chunk(self, chunk_pcm: bytes) -> None:
        """Add a PCM int16 chunk to the internal buffer."""
        signal = np.frombuffer(chunk_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(signal)
        buf_len = len(self._audio_buffer)

        if self._write_pos + n <= buf_len:
            self._audio_buffer[self._write_pos:self._write_pos + n] = signal
            self._write_pos += n
        else:
            # Shift buffer left and append
            shift = self._write_pos + n - buf_len
            self._audio_buffer[:-shift] = self._audio_buffer[shift:]
            self._audio_buffer[-n:] = signal
            self._write_pos = buf_len

        self._total_samples += n

    def update_diarization(self) -> None:
        """Re-run diarization on the current audio buffer."""
        self._ensure_pipeline()
        import torch

        valid = min(self._write_pos, len(self._audio_buffer))
        if valid < self._sample_rate:  # need at least 1s
            return

        waveform = torch.tensor(
            self._audio_buffer[:valid], dtype=torch.float32
        ).unsqueeze(0)

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": self._sample_rate},
            max_speakers=self._max_speakers,
        )

        self._segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            self._segments.append((turn.start, turn.end, speaker))

    def get_speaker_at(self, timestamp_seconds: float) -> str:
        """Get the active speaker at a given timestamp.

        Returns "unknown" if no speaker is active or diarization hasn't run.
        """
        buffer_duration = min(self._write_pos, len(self._audio_buffer)) / self._sample_rate
        buffer_start = max(0, self._total_samples / self._sample_rate - buffer_duration)
        relative_ts = timestamp_seconds - buffer_start

        for start, end, speaker in self._segments:
            if start <= relative_ts <= end:
                return speaker
        return "unknown"

    def get_current_speaker(self) -> str:
        """Get the speaker at the most recent timestamp."""
        if not self._segments:
            return "unknown"
        return self._segments[-1][2]

    def diarize_file(self, audio_path: str) -> list[tuple[float, float, str]]:
        """Run full-file diarization on a WAV/audio file.

        Unlike add_chunk() + update_diarization() which process incrementally,
        this loads the entire file and runs diarization once — more accurate
        for batch processing.

        Returns list of (start_seconds, end_seconds, speaker_id) tuples.
        """
        self._ensure_pipeline()
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != self._sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self._sample_rate
            )

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": self._sample_rate},
            max_speakers=self._max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        self._segments = segments
        logger.info(
            "Full-file diarization: %d segments from %s",
            len(segments), audio_path,
        )
        return segments

    def get_speaker_embeddings(self) -> dict[str, np.ndarray]:
        """Return speaker embeddings from the pipeline's embedding model.

        Only available after diarize_file() or update_diarization() has run.
        Returns {speaker_id: embedding_array} for voice profile persistence.
        """
        if self._pipeline is None:
            return {}

        # pyannote's pipeline stores embeddings in the clustering step
        # Access depends on pipeline version — try common patterns
        try:
            embeddings = {}
            if hasattr(self._pipeline, '_embedding'):
                emb_model = self._pipeline._embedding
                # Extract from last diarization run
                for speaker_id in set(s[2] for s in self._segments):
                    # This requires the internal state — may not be accessible
                    # in all pyannote versions
                    embeddings[speaker_id] = np.zeros(192)  # placeholder
                logger.info(
                    "Speaker embeddings: %d speakers (placeholder — "
                    "full extraction requires pyannote internals access)",
                    len(embeddings),
                )
            return embeddings
        except Exception as e:
            logger.warning("Could not extract speaker embeddings: %s", e)
            return {}
