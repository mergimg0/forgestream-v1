"""Test rapport dashboard endpoint."""

from unittest.mock import MagicMock

import pytest

from forgestream.dashboard.api import create_router


def _make_firestore_db(events):
    mock_docs = []
    for e in events:
        doc = MagicMock()
        doc.to_dict.return_value = e
        mock_docs.append(doc)
    db = MagicMock()
    db.collection.return_value.order_by.return_value.stream.return_value = mock_docs
    return db


class TestRapportEndpoint:
    @pytest.mark.asyncio
    async def test_rapport_endpoint_returns_scores(self):
        events = [
            {"event_type": "rapport_score", "payload": {
                "timestamp_ms": 60000,
                "group_composite": 0.72,
                "group_trend": 0.05,
                "pair_scores": [{"speaker_a": "sp0", "speaker_b": "sp1", "composite": 0.72}],
                "disengaged_speakers": [],
            }},
        ]
        db = _make_firestore_db(events)
        router = create_router(db)

        rapport_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/emotion/rapport":
                rapport_fn = route.endpoint
                break
        assert rapport_fn is not None

        result = await rapport_fn()
        assert len(result["scores"]) == 1
        assert result["scores"][0]["group_composite"] == 0.72
        assert result["latest_trend"] == 0.05
