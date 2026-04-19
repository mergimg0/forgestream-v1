"""Tests for PostMeetingSynthesis emotion integration."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.emotion.buffer import AudioRingBuffer
from forgestream.events.schema import Event, EventType
from forgestream.post_meeting import PostMeetingSynthesis


def _make_event(event_type: EventType, payload: dict) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.0,
        payload=payload,
    )


class TestPostMeetingEmotionIntegration:
    @pytest.mark.asyncio
    async def test_run_saves_emotion_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(emotion_enabled=True, data_dir=tmpdir)
            pms = PostMeetingSynthesis(config=config, data_dir=tmpdir)

            buf = AudioRingBuffer(capacity_seconds=5.0)
            buf.write_chunk(b"\x00" * 16000)

            events = [
                _make_event(EventType.CLAIM, {
                    "text": "Test", "topic_keywords": ["x"],
                    "audio_timestamp": 1000, "confidence": 0.7,
                    "speaker": "sp0",
                }),
                _make_event(EventType.PROSODIC_FEATURE, {
                    "timestamp_ms": 1000, "arousal": 0.6,
                    "speaker_id": "sp0",
                }),
            ]

            result = await pms.run(
                events, meeting_name="test", audio_buffer=buf,
            )
            assert "corpus_audio_path" in result
            assert "corpus_index_path" in result
            assert Path(result["corpus_audio_path"]).exists()
            assert Path(result["corpus_index_path"]).exists()

    @pytest.mark.asyncio
    async def test_run_tunes_tone_adjustments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(emotion_enabled=True, data_dir=tmpdir)
            pms = PostMeetingSynthesis(config=config, data_dir=tmpdir)

            events = [
                _make_event(EventType.CLAIM, {
                    "text": "Test", "topic_keywords": ["x"],
                    "confidence": 0.7, "tone_markers": ["hesitation"],
                }),
                _make_event(EventType.PROSODIC_FEATURE, {
                    "timestamp_ms": 1000, "arousal": 0.4,
                    "jitter_local": 0.03, "shimmer_local": 0.05, "hnr": 12.0,
                }),
            ]

            result = await pms.run(events, meeting_name="test", human_score=0.6)
            assert "tone_adjustments" in result
            assert "hesitation_penalty" in result["tone_adjustments"]

    @pytest.mark.asyncio
    async def test_run_works_without_audio_buffer(self):
        """Backward compat: run() without audio_buffer still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(emotion_enabled=True, data_dir=tmpdir)
            pms = PostMeetingSynthesis(config=config, data_dir=tmpdir)

            events = [
                _make_event(EventType.CLAIM, {
                    "text": "Test", "topic_keywords": ["x"],
                }),
            ]

            result = await pms.run(events, meeting_name="test")
            assert "report_path" in result
            assert "weights" in result
            # No corpus saved when no audio buffer
            assert result.get("corpus_audio_path") is None
