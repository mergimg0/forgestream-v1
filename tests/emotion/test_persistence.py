"""Tests for EmotionCorpus — cross-meeting audio + feature persistence."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from forgestream.emotion.buffer import AudioRingBuffer
from forgestream.emotion.persistence import EmotionCorpus
from forgestream.events.schema import Event, EventType


def _make_event(event_type: EventType, payload: dict) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.0,
        payload=payload,
    )


class TestEmotionCorpus:
    def test_save_meeting_audio_creates_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = EmotionCorpus(corpus_dir=tmpdir)
            buf = AudioRingBuffer(capacity_seconds=5.0)

            # Write 1 second of audio (2 chunks)
            chunk = b"\x00\x01" * 8000
            buf.write_chunk(chunk)
            buf.write_chunk(chunk)

            path = corpus.save_meeting_audio("session-123", buf)
            assert Path(path).exists()
            assert Path(path).suffix == ".wav"
            # WAV header (44 bytes) + 2 chunks of 16000 bytes each
            assert Path(path).stat().st_size > 32000

    def test_save_feature_index_creates_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = EmotionCorpus(corpus_dir=tmpdir)
            prosodic_events = [
                _make_event(EventType.PROSODIC_FEATURE, {
                    "timestamp_ms": 1000, "arousal": 0.7, "speaker_id": "sp0",
                }),
            ]
            claim_events = [
                _make_event(EventType.CLAIM, {
                    "text": "Test claim", "audio_timestamp": 1000,
                }),
            ]
            path = corpus.save_feature_index("session-123", prosodic_events, claim_events)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert len(data["prosodic_features"]) == 1
            assert len(data["claims"]) == 1
            assert data["session_id"] == "session-123"

    def test_get_training_samples_returns_saved_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = EmotionCorpus(corpus_dir=tmpdir)

            # Save two sessions
            for sid in ["s1", "s2"]:
                prosodic = [
                    _make_event(EventType.PROSODIC_FEATURE, {
                        "timestamp_ms": 1000, "arousal": 0.6,
                    }),
                ]
                claims = [
                    _make_event(EventType.CLAIM, {
                        "text": f"Claim from {sid}", "audio_timestamp": 1000,
                    }),
                ]
                corpus.save_feature_index(sid, prosodic, claims)

            samples = corpus.get_training_samples()
            assert len(samples) == 2

    def test_save_to_nonexistent_dir_creates_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "deep" / "nested" / "corpus"
            corpus = EmotionCorpus(corpus_dir=str(nested))
            buf = AudioRingBuffer(capacity_seconds=2.0)
            buf.write_chunk(b"\x00" * 16000)
            path = corpus.save_meeting_audio("test", buf)
            assert Path(path).exists()
