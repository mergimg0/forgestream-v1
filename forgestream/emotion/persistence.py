"""Persist raw audio segments aligned with prosodic features and claims.

After each meeting, save:
1. Full meeting audio as WAV
2. Feature index: prosodic features + claims aligned by timestamp
3. Human feedback scores per meeting

This builds a proprietary training corpus over time.
"""

from __future__ import annotations

import json
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

from forgestream.events.schema import Event, EventType

from .buffer import AudioRingBuffer


class EmotionCorpus:
    """Manages the cross-meeting emotion training corpus.

    Parameters:
        corpus_dir: Directory to store audio files and feature indices.
    """

    def __init__(self, corpus_dir: str = "data/emotion_corpus") -> None:
        self._corpus_dir = Path(corpus_dir)
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

    def save_meeting_audio(
        self, session_id: str, audio_buffer: AudioRingBuffer
    ) -> str:
        """Save full meeting audio from ring buffer to WAV file.

        Returns the path to the saved WAV file.
        """
        audio_dir = self._corpus_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"{date_str}_{session_id}.wav"
        path = audio_dir / filename

        # Read all available audio from buffer
        raw_audio = audio_buffer.read_window(
            duration_seconds=600.0  # up to 10 minutes
        )

        # Write as WAV: 16kHz, mono, 16-bit PCM
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(16000)
            wf.writeframes(raw_audio)

        return str(path)

    def save_feature_index(
        self,
        session_id: str,
        prosodic_events: list[Event],
        claim_events: list[Event],
    ) -> str:
        """Save aligned feature-claim index as JSON.

        Returns the path to the saved index file.
        """
        index_dir = self._corpus_dir / "indices"
        index_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"{date_str}_{session_id}.json"
        path = index_dir / filename

        index = {
            "session_id": session_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "prosodic_features": [
                {
                    "event_id": str(e.id),
                    "timestamp_ms": e.payload.get("timestamp_ms", 0),
                    "arousal": e.payload.get("arousal", 0.5),
                    "valence": e.payload.get("valence", 0.5),
                    "dominance": e.payload.get("dominance", 0.5),
                    "speaker_id": e.payload.get("speaker_id", "unknown"),
                    "emotion_tag": e.payload.get("emotion_tag"),
                }
                for e in prosodic_events
            ],
            "claims": [
                {
                    "event_id": str(e.id),
                    "text": e.payload.get("text", ""),
                    "audio_timestamp": e.payload.get("audio_timestamp"),
                    "confidence": e.payload.get("confidence", 0.5),
                    "speaker": e.payload.get("speaker", "unknown"),
                }
                for e in claim_events
            ],
        }

        path.write_text(json.dumps(index, indent=2))
        return str(path)

    def get_training_samples(self) -> list[dict]:
        """Load all feature indices for training data generation.

        Returns a list of index dicts, one per saved session.
        """
        index_dir = self._corpus_dir / "indices"
        if not index_dir.exists():
            return []

        samples = []
        for index_file in sorted(index_dir.glob("*.json")):
            data = json.loads(index_file.read_text())
            samples.append(data)
        return samples
