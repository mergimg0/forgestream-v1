"""Tests for emotion-related dashboard API endpoints."""

from unittest.mock import MagicMock

import pytest

from forgestream.dashboard.api import create_router


def _make_firestore_db(events: list[dict]):
    """Create a mock Firestore DB that returns given events."""
    mock_docs = []
    for e in events:
        doc = MagicMock()
        doc.to_dict.return_value = e
        mock_docs.append(doc)

    db = MagicMock()
    db.collection.return_value.order_by.return_value.stream.return_value = mock_docs
    return db


class TestEmotionEndpoints:
    @pytest.mark.asyncio
    async def test_emotion_timeline_returns_prosodic_events(self):
        events = [
            {"event_type": "prosodic_feature", "payload": {
                "timestamp_ms": 1000, "speaker_id": "sp0",
                "arousal": 0.7, "valence": 0.5, "dominance": 0.6,
                "emotion_tag": "excited",
            }},
            {"event_type": "prosodic_feature", "payload": {
                "timestamp_ms": 2000, "speaker_id": "sp1",
                "arousal": 0.3, "valence": 0.6, "dominance": 0.4,
                "emotion_tag": "calm",
            }},
            {"event_type": "claim", "payload": {"text": "ignore me"}},
        ]
        db = _make_firestore_db(events)
        router = create_router(db)

        # Find the endpoint function
        timeline_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/emotion/timeline":
                timeline_fn = route.endpoint
                break
        assert timeline_fn is not None, "endpoint /emotion/timeline not found"

        result = await timeline_fn()
        assert len(result["timeline"]) == 2
        assert result["timeline"][0]["arousal"] == 0.7
        assert result["timeline"][1]["speaker_id"] == "sp1"

    @pytest.mark.asyncio
    async def test_emotion_entrainment_returns_snapshots(self):
        events = [
            {"event_type": "entrainment_snapshot", "payload": {
                "timestamp_ms": 60000,
                "speaker_pairs": [{"speaker_a": "sp0", "speaker_b": "sp1"}],
                "group_metrics": {"participation_parity": 0.85},
            }},
        ]
        db = _make_firestore_db(events)
        router = create_router(db)

        entrainment_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/emotion/entrainment":
                entrainment_fn = route.endpoint
                break
        assert entrainment_fn is not None

        result = await entrainment_fn()
        assert len(result["snapshots"]) == 1
        assert result["snapshots"][0]["group_metrics"]["participation_parity"] == 0.85

    @pytest.mark.asyncio
    async def test_emotion_speakers_returns_summary(self):
        events = [
            {"event_type": "prosodic_feature", "payload": {
                "speaker_id": "sp0", "arousal": 0.7, "valence": 0.5,
            }},
            {"event_type": "prosodic_feature", "payload": {
                "speaker_id": "sp0", "arousal": 0.9, "valence": 0.6,
            }},
            {"event_type": "prosodic_feature", "payload": {
                "speaker_id": "sp1", "arousal": 0.3, "valence": 0.4,
            }},
        ]
        db = _make_firestore_db(events)
        router = create_router(db)

        speakers_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/emotion/speakers":
                speakers_fn = route.endpoint
                break
        assert speakers_fn is not None

        result = await speakers_fn()
        assert len(result["speakers"]) == 2
        sp0 = next(s for s in result["speakers"] if s["speaker_id"] == "sp0")
        assert sp0["mean_arousal"] == pytest.approx(0.8, abs=0.01)
        assert sp0["count"] == 2
