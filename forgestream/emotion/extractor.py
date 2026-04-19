"""EmotionExtractor — sliding-window prosodic feature extraction.

Accumulates audio chunks in a sliding window. At each stride boundary,
concatenates the window, runs Parselmouth + openSMILE, derives dimensional
emotion, and emits a PROSODIC_FEATURE event through the Orchestrator.

Runs feature extraction in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from forgestream.events.schema import Event, EventType

from .buffer import AudioRingBuffer
from .features import (
    EGeMAPSExtractor,
    PraatExtractor,
    ProsodicFeatures,
    derive_dimensional_emotion,
)

logger = logging.getLogger(__name__)

AUTHOR = "emotion_extractor"


class EmotionExtractor:
    """Sliding-window prosodic feature extractor.

    Parameters:
        orchestrator: The ForgeStream Orchestrator to emit events through.
        audio_buffer: The shared AudioRingBuffer.
        branch_id: The current branch ID for emitted events.
        window_seconds: Duration of the analysis window (default 3.0s).
        stride_seconds: How often to emit features (default 1.0s).
        chunk_duration: Duration of each incoming chunk (default 0.5s).
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        audio_buffer: AudioRingBuffer,
        branch_id: UUID,
        window_seconds: float = 3.0,
        stride_seconds: float = 1.0,
        chunk_duration: float = 0.5,
        diarizer: "SpeakerDiarizer | None" = None,
        classifier: "SenseVoiceClassifier | None" = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._audio_buffer = audio_buffer
        self._branch_id = branch_id
        self._diarizer = diarizer
        self._classifier = classifier

        self._window_seconds = window_seconds
        self._stride_seconds = stride_seconds
        self._chunk_duration = chunk_duration

        # Chunk accumulation
        max_window_chunks = max(1, int(window_seconds / chunk_duration))
        self._chunk_window: deque[bytes] = deque(maxlen=max_window_chunks)
        self._chunks_per_stride = max(1, int(stride_seconds / chunk_duration))
        self._stride_counter = 0
        self._chunk_counter = 0

        # Extractors (initialized lazily)
        self._praat = PraatExtractor(sample_rate=16000)
        self._egemaps: EGeMAPSExtractor | None = None
        self._egemaps_available: bool | None = None

        # Thread pool for CPU-bound feature extraction
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_egemaps(self) -> EGeMAPSExtractor | None:
        """Lazily initialize eGeMAPS extractor. Returns None if unavailable."""
        if self._egemaps_available is None:
            try:
                self._egemaps = EGeMAPSExtractor(sample_rate=16000)
                self._egemaps_available = True
            except ImportError:
                logger.info("openSMILE not available — eGeMAPS features disabled")
                self._egemaps_available = False
        return self._egemaps if self._egemaps_available else None

    async def process_chunk(self, chunk: bytes, chunk_index: int) -> None:
        """Process an incoming audio chunk.

        Accumulates chunks and runs extraction at stride boundaries.
        Also feeds the diarizer if available.
        """
        self._chunk_window.append(chunk)
        self._stride_counter += 1
        self._chunk_counter += 1

        # Feed diarizer
        if self._diarizer is not None:
            self._diarizer.add_chunk(chunk)
            # Re-run diarization periodically (every 10 chunks = 5 seconds)
            if self._chunk_counter % 10 == 0:
                try:
                    self._diarizer.update_diarization()
                except Exception as e:
                    logger.warning("Diarization failed: %s", e)

        if self._stride_counter >= self._chunks_per_stride:
            self._stride_counter = 0
            window_bytes = b"".join(self._chunk_window)
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                self._executor,
                self._extract_features,
                window_bytes,
            )
            await self._emit_event(features, chunk_index)

    def _extract_features(self, window_bytes: bytes) -> ProsodicFeatures:
        """Run feature extraction on a concatenated audio window (CPU-bound)."""
        import numpy as np

        signal = np.frombuffer(window_bytes, dtype=np.int16)

        # Parselmouth: F0, jitter, shimmer, HNR, spectral centroid
        praat_result = self._praat.extract(signal)

        # eGeMAPS: 88-dim vector (if available)
        egemaps = self._get_egemaps()
        egemaps_vector = egemaps.extract(signal) if egemaps else [0.0] * 88

        # Derive dimensional emotion
        arousal, valence, dominance = derive_dimensional_emotion(
            f0_mean=praat_result["f0_mean"],
            f0_std=praat_result["f0_std"],
            energy_rms=praat_result["energy_rms"],
            hnr=praat_result["hnr"],
            spectral_centroid=praat_result["spectral_centroid"],
        )

        # Categorical emotion via SenseVoice (if classifier is wired in)
        emotion_tag: str | None = None
        emotion_confidence: float | None = None
        if self._classifier is not None:
            try:
                emotion_tag, emotion_confidence = self._classifier.classify(signal)
            except Exception as exc:
                logger.warning("SenseVoice classify error: %s", exc)

        return ProsodicFeatures(
            f0_mean=praat_result["f0_mean"],
            f0_std=praat_result["f0_std"],
            f0_contour=praat_result["f0_contour"],
            energy_rms=praat_result["energy_rms"],
            jitter_local=praat_result["jitter_local"],
            shimmer_local=praat_result["shimmer_local"],
            hnr=praat_result["hnr"],
            spectral_centroid=praat_result["spectral_centroid"],
            egemaps_vector=egemaps_vector,
            arousal=arousal,
            valence=valence,
            dominance=dominance,
            emotion_tag=emotion_tag,
            emotion_confidence=emotion_confidence,
        )

    async def _emit_event(
        self, features: ProsodicFeatures, chunk_index: int
    ) -> None:
        """Emit a PROSODIC_FEATURE event through the Orchestrator."""
        payload = features.to_payload()
        payload.update({
            "speaker_id": self._diarizer.get_current_speaker() if self._diarizer else "unknown",
            "timestamp_ms": self._audio_buffer.chunk_timestamp_ms(chunk_index),
            "chunk_index": chunk_index,
            "window_duration_ms": int(self._window_seconds * 1000),
        })

        event = Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=self._orchestrator.session_id,
            branch_id=self._branch_id,
            author=AUTHOR,
            evaluator=0.0,
            payload=payload,
        )
        await self._orchestrator.process_event(event)
