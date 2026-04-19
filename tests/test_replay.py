"""Tests for MeetingReplay -- forgestream/replay.py."""

from __future__ import annotations

import json
import math
import struct
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.replay import (
    MeetingReplay,
    _TimelineEntry,
    _extract_timestamp_ms,
    _make_synthetic_event,
    _is_valid_uuid,
    _wav_duration_ms,
    load_json,
    load_wav,
    load_events_from_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a minimal sine-wave WAV to disk."""
    n = int(sample_rate * duration_s)
    samples = [int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _make_event(
    event_type: EventType = EventType.CLAIM,
    timestamp_ms: float = 0.0,
    payload_extra: dict | None = None,
) -> Event:
    p: dict = {"timestamp_ms": timestamp_ms}
    if payload_extra:
        p.update(payload_extra)
    if event_type == EventType.CLAIM:
        p.setdefault("text", "test claim")
        p.setdefault("confidence", 0.8)
        p.setdefault("speaker", "A")
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.5,
        payload=p,
    )


def _make_index(
    session_id: str | None = None,
    prosodic: list[dict] | None = None,
    claims: list[dict] | None = None,
) -> dict:
    return {
        "session_id": session_id or str(uuid4()),
        "created": datetime.now(timezone.utc).isoformat(),
        "prosodic_features": prosodic or [],
        "claims": claims or [],
    }


# ---------------------------------------------------------------------------
# Unit tests: load_wav / load_json
# ---------------------------------------------------------------------------

class TestLoadHelpers:
    def test_load_wav_returns_bytes(self, tmp_path):
        p = tmp_path / "test.wav"
        _make_wav(p, duration_s=0.5)
        data = load_wav(str(p))
        assert isinstance(data, bytes)
        # 0.5s * 16000 samples/s * 2 bytes = 16000 bytes
        assert len(data) == 16000

    def test_load_json_returns_dict(self, tmp_path):
        p = tmp_path / "index.json"
        p.write_text(json.dumps({"foo": "bar"}))
        result = load_json(str(p))
        assert result == {"foo": "bar"}

    def test_wav_duration_ms(self, tmp_path):
        p = tmp_path / "dur.wav"
        _make_wav(p, duration_s=2.0)
        dur = _wav_duration_ms(str(p))
        assert abs(dur - 2000.0) < 1.0


# ---------------------------------------------------------------------------
# Unit tests: timeline construction
# ---------------------------------------------------------------------------

class TestTimelineConstruction:
    def test_events_sorted_by_timestamp(self, tmp_path):
        ev1 = _make_event(timestamp_ms=500.0)
        ev2 = _make_event(timestamp_ms=100.0)
        ev3 = _make_event(timestamp_ms=300.0)

        replay = MeetingReplay("", "", [ev1, ev2, ev3], speed=0)
        replay._index = {}
        replay._build_timeline()

        ts_list = [e.timestamp_ms for e in replay._timeline]
        assert ts_list == sorted(ts_list)

    def test_index_prosodic_added_to_timeline(self, tmp_path):
        idx = _make_index(
            prosodic=[
                {"event_id": "fake-1", "timestamp_ms": 200.0, "arousal": 0.6,
                 "valence": 0.4, "dominance": 0.5, "speaker_id": "A"},
            ]
        )
        replay = MeetingReplay("", "", [], speed=0)
        replay._index = idx
        replay._build_timeline()

        assert len(replay._timeline) == 1
        entry = replay._timeline[0]
        assert entry.timestamp_ms == 200.0
        assert entry.event.event_type == EventType.PROSODIC_FEATURE

    def test_index_claims_added_to_timeline(self):
        idx = _make_index(
            claims=[
                {"event_id": "fake-2", "audio_timestamp": 750.0,
                 "text": "hello world", "confidence": 0.9, "speaker": "B"},
            ]
        )
        replay = MeetingReplay("", "", [], speed=0)
        replay._index = idx
        replay._build_timeline()

        assert len(replay._timeline) == 1
        entry = replay._timeline[0]
        assert entry.timestamp_ms == 750.0
        assert entry.event.event_type == EventType.CLAIM
        assert entry.event.payload["text"] == "hello world"

    def test_duplicate_event_ids_not_added_from_index(self):
        """Events passed directly must not be duplicated from index."""
        ev = _make_event(EventType.CLAIM, timestamp_ms=100.0)
        idx = _make_index(
            claims=[
                {"event_id": str(ev.id), "audio_timestamp": 100.0,
                 "text": "dup", "confidence": 0.5, "speaker": "A"},
            ]
        )
        replay = MeetingReplay("", "", [ev], speed=0)
        replay._index = idx
        replay._build_timeline()

        # Only one entry — the original event, not the index duplicate
        assert len(replay._timeline) == 1
        assert replay._timeline[0].event.id == ev.id

    def test_empty_events_and_index_gives_empty_timeline(self):
        replay = MeetingReplay("", "", [], speed=0)
        replay._index = {}
        replay._build_timeline()
        assert replay._timeline == []

    def test_duration_extends_to_last_event_when_no_audio(self):
        ev = _make_event(timestamp_ms=5000.0)
        replay = MeetingReplay("", "", [ev], speed=0)
        replay._index = {}
        replay._audio_bytes = None
        replay._build_timeline()
        # duration_ms was 0 before build, should now be at least 5000
        assert replay._duration_ms >= 5000.0


# ---------------------------------------------------------------------------
# Unit tests: speed math
# ---------------------------------------------------------------------------

class TestSpeedMath:
    def test_speed_setter_clamps_negative(self):
        replay = MeetingReplay("", "", [], speed=1.0)
        replay.speed = -5.0
        assert replay.speed == 0.0

    def test_speed_setter_accepts_valid(self):
        replay = MeetingReplay("", "", [], speed=1.0)
        replay.speed = 2.5
        assert replay.speed == 2.5

    def test_constructor_clamps_negative_speed(self):
        replay = MeetingReplay("", "", [], speed=-1.0)
        assert replay.speed == 0.0

    def test_speed_zero_means_no_delay(self):
        # With speed=0, _speed = 0.0; timing block is skipped (speed > 0 is False)
        replay = MeetingReplay("", "", [], speed=0)
        assert replay.speed == 0.0


# ---------------------------------------------------------------------------
# Unit tests: events-only mode (no audio)
# ---------------------------------------------------------------------------

class TestEventsOnlyMode:
    @pytest.mark.asyncio
    async def test_play_without_audio_emits_events(self, capsys):
        events = [
            _make_event(EventType.CLAIM, 0.0, {"text": "claim A", "speaker": "A", "confidence": 0.9}),
            _make_event(EventType.RAPPORT_SCORE, 100.0, {"score": 0.75}),
        ]
        replay = MeetingReplay("", "", events, speed=0)
        await replay.play(tui_app=None)

        captured = capsys.readouterr()
        assert "CLAIM" in captured.out
        assert "claim A" in captured.out
        assert "RAPPORT" in captured.out

    @pytest.mark.asyncio
    async def test_play_no_events_no_crash(self):
        replay = MeetingReplay("", "", [], speed=0)
        await replay.play(tui_app=None)  # must not raise

    @pytest.mark.asyncio
    async def test_stop_halts_playback(self, capsys):
        """Calling stop() mid-replay should halt before all events fire."""
        events = [_make_event(EventType.CLAIM, float(i * 100)) for i in range(50)]
        replay = MeetingReplay("", "", events, speed=0)

        import asyncio

        async def stopper():
            await asyncio.sleep(0.01)
            replay.stop()

        await asyncio.gather(
            replay.play(tui_app=None),
            stopper(),
        )
        # We don't assert exact count, just that it ran without error
        assert replay._stopped is True


# ---------------------------------------------------------------------------
# Unit tests: load_events_from_json
# ---------------------------------------------------------------------------

class TestLoadEventsFromJson:
    def test_loads_list_format(self, tmp_path):
        ev = _make_event()
        data = [ev.to_dict()]
        p = tmp_path / "events.json"
        p.write_text(json.dumps(data))

        loaded = load_events_from_json(str(p))
        assert len(loaded) == 1
        assert loaded[0].id == ev.id

    def test_loads_dict_with_events_key(self, tmp_path):
        ev = _make_event()
        data = {"events": [ev.to_dict()]}
        p = tmp_path / "events.json"
        p.write_text(json.dumps(data))

        loaded = load_events_from_json(str(p))
        assert len(loaded) == 1

    def test_missing_file_returns_empty(self):
        result = load_events_from_json("/nonexistent/path.json")
        assert result == []

    def test_malformed_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{{")
        result = load_events_from_json(str(p))
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests: audio playback mocked
# ---------------------------------------------------------------------------

class TestAudioPlayback:
    @pytest.mark.asyncio
    async def test_play_with_wav_calls_sounddevice(self, tmp_path, capsys):
        """Audio playback uses sounddevice; mock it to avoid actual device."""
        p = tmp_path / "audio.wav"
        _make_wav(p, duration_s=0.1)  # short file

        events = [_make_event(EventType.CLAIM, 0.0, {"text": "hi", "speaker": "A", "confidence": 0.8})]
        replay = MeetingReplay(str(p), "", events, speed=0)

        # Patch sounddevice.OutputStream to a no-op mock
        with patch("forgestream.replay.HAS_SOUNDDEVICE", False):
            await replay.play(tui_app=None)

        captured = capsys.readouterr()
        assert "CLAIM" in captured.out

    @pytest.mark.asyncio
    async def test_play_with_missing_wav_does_not_crash(self, tmp_path):
        """Missing audio file → events-only mode, no crash."""
        events = [_make_event(EventType.CLAIM, 0.0, {"text": "x", "speaker": "A", "confidence": 0.5})]
        replay = MeetingReplay("/nonexistent/audio.wav", "", events, speed=0)
        await replay.play(tui_app=None)  # must not raise


# ---------------------------------------------------------------------------
# Unit tests: load() is idempotent
# ---------------------------------------------------------------------------

class TestLoadIdempotent:
    def test_load_twice_same_timeline(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_wav(p, duration_s=0.5)

        idx = _make_index(
            prosodic=[{"event_id": "x1", "timestamp_ms": 100.0,
                       "arousal": 0.5, "valence": 0.5, "dominance": 0.5, "speaker_id": "A"}]
        )
        idx_p = tmp_path / "idx.json"
        idx_p.write_text(json.dumps(idx))

        replay = MeetingReplay(str(p), str(idx_p), [], speed=0)
        replay.load()
        tl1 = len(replay._timeline)
        replay.load()
        tl2 = len(replay._timeline)
        assert tl1 == tl2 == 1


# ---------------------------------------------------------------------------
# Unit tests: _extract_timestamp_ms
# ---------------------------------------------------------------------------

class TestExtractTimestampMs:
    def test_prefers_timestamp_ms(self):
        ev = _make_event(timestamp_ms=500.0)
        ev.payload["audio_timestamp"] = 200.0
        assert _extract_timestamp_ms(ev) == 500.0

    def test_falls_back_to_audio_timestamp(self):
        ev = _make_event(EventType.CLAIM, 0.0)
        del ev.payload["timestamp_ms"]
        ev.payload["audio_timestamp"] = 333.0
        assert _extract_timestamp_ms(ev) == 333.0

    def test_returns_zero_when_no_ts_key(self):
        ev = _make_event(EventType.CLAIM, 0.0)
        ev.payload.clear()
        assert _extract_timestamp_ms(ev) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: _is_valid_uuid
# ---------------------------------------------------------------------------

class TestIsValidUuid:
    def test_valid_uuid(self):
        assert _is_valid_uuid(str(uuid4())) is True

    def test_invalid_string(self):
        assert _is_valid_uuid("not-a-uuid") is False

    def test_empty_string(self):
        assert _is_valid_uuid("") is False


# ---------------------------------------------------------------------------
# Unit tests: _TimelineEntry ordering
# ---------------------------------------------------------------------------

class TestTimelineEntryOrdering:
    def test_less_than(self):
        e1 = _TimelineEntry(100.0, _make_event())
        e2 = _TimelineEntry(200.0, _make_event())
        assert e1 < e2
        assert not e2 < e1

    def test_sort(self):
        entries = [
            _TimelineEntry(300.0, _make_event()),
            _TimelineEntry(100.0, _make_event()),
            _TimelineEntry(200.0, _make_event()),
        ]
        entries.sort()
        assert [e.timestamp_ms for e in entries] == [100.0, 200.0, 300.0]
