"""Tests for MetaGleaningEngine — cross-meeting pattern analysis."""

from __future__ import annotations

from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.meta_gleanings import MetaGleaning, MetaGleaningEngine


def _make_sid_bid():
    return uuid4(), uuid4()


def _claim(sid, bid, text: str, keywords: list[str] = None, ts_offset: float = 0.0) -> Event:
    from datetime import datetime, timezone, timedelta
    return Event(
        event_type=EventType.CLAIM,
        session_id=sid,
        branch_id=bid,
        author="gemini",
        evaluator=0.5,
        payload={
            "text": text,
            "topic_keywords": keywords or [],
        },
        timestamp=datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=ts_offset),
    )


def _requirement(sid, bid, desc: str, keywords: list[str] = None) -> Event:
    return Event(
        event_type=EventType.REQUIREMENT,
        session_id=sid,
        branch_id=bid,
        author="synthesis",
        evaluator=0.7,
        payload={
            "description": desc,
            "topic_keywords": keywords or [],
        },
    )


def _artifact(sid, bid) -> Event:
    return Event(
        event_type=EventType.ARTIFACT,
        session_id=sid,
        branch_id=bid,
        author="scaffold",
        evaluator=0.8,
        payload={"compiles": True, "tests_pass": True, "files_created": ["x.py"]},
    )


class TestMetaGleaning:
    def test_dataclass_construction(self):
        g = MetaGleaning(
            gleaning_type="topic_outcome",
            description="keyword 'auth' co-occurs with requirements",
            confidence=0.8,
            supporting_events=5,
            metadata={},
        )
        assert g.gleaning_type == "topic_outcome"
        assert g.confidence == 0.8
        assert g.supporting_events == 5

    def test_to_dict(self):
        g = MetaGleaning(
            gleaning_type="discussion_evolution",
            description="Claim density converges in second half",
            confidence=0.7,
            supporting_events=10,
            metadata={"claim_density_early": 0.3, "claim_density_late": 0.7},
        )
        d = g.to_dict()
        assert d["gleaning_type"] == "discussion_evolution"
        assert d["confidence"] == 0.7
        assert "metadata" in d


class TestMetaGleaningEngine:
    def test_analyze_returns_list(self):
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        events = [_claim(sid, bid, "We need auth", ["auth", "security"])]
        result = engine.analyze(events)
        assert isinstance(result, list)

    def test_analyze_empty_returns_empty(self):
        engine = MetaGleaningEngine()
        result = engine.analyze([])
        assert isinstance(result, list)

    def test_topic_outcome_mapping_detected(self):
        """Keywords that co-occur with REQUIREMENTs should appear in a gleaning."""
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        events = [
            _claim(sid, bid, "auth is needed", ["auth", "security"]),
            _requirement(sid, bid, "Build auth service", ["auth"]),
            _claim(sid, bid, "scaling is important", ["scaling"]),
            _requirement(sid, bid, "Design for scaling", ["scaling"]),
            _claim(sid, bid, "UI needs refresh", ["ui"]),
        ]
        gleanings = engine.analyze(events)
        topic_types = [g for g in gleanings if g.gleaning_type == "topic_outcome"]
        assert len(topic_types) > 0

    def test_discussion_evolution_detected(self):
        """Dense claims followed by sparser = converge pattern."""
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        # Early: 6 claims in 10 min, Late: 2 claims in 10 min
        events = []
        for i in range(6):
            events.append(_claim(sid, bid, f"early claim {i}", ts_offset=float(i)))
        for i in range(2):
            events.append(_claim(sid, bid, f"late claim {i}", ts_offset=float(20 + i * 5)))

        gleanings = engine.analyze(events)
        evo_types = [g for g in gleanings if g.gleaning_type == "discussion_evolution"]
        assert len(evo_types) > 0

    def test_engagement_outcome_detected(self):
        """Topics with many claims + artifacts should yield engagement gleaning."""
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        events = [
            _claim(sid, bid, "topic A lots", ["topicA"]),
            _claim(sid, bid, "topic A more", ["topicA"]),
            _claim(sid, bid, "topic A again", ["topicA"]),
            _artifact(sid, bid),
            _claim(sid, bid, "topic B once", ["topicB"]),
        ]
        gleanings = engine.analyze(events)
        engagement_types = [g for g in gleanings if g.gleaning_type == "engagement_outcome"]
        assert len(engagement_types) > 0

    def test_all_gleanings_have_required_fields(self):
        """Every MetaGleaning from analyze() has required fields."""
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        events = [
            _claim(sid, bid, "auth needed", ["auth"]),
            _requirement(sid, bid, "auth service", ["auth"]),
            _artifact(sid, bid),
        ]
        for g in engine.analyze(events):
            assert isinstance(g.gleaning_type, str)
            assert isinstance(g.description, str)
            assert 0.0 <= g.confidence <= 1.0
            assert isinstance(g.supporting_events, int)
            assert isinstance(g.metadata, dict)

    def test_analyze_with_no_requirements(self):
        """Should not raise if no REQUIREMENTs in events."""
        engine = MetaGleaningEngine()
        sid, bid = _make_sid_bid()
        events = [_claim(sid, bid, f"claim {i}", ts_offset=float(i)) for i in range(5)]
        result = engine.analyze(events)
        assert isinstance(result, list)
