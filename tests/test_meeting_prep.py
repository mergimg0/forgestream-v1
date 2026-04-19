"""Meeting prep tests — TDD for MeetingPrep class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType


def _make_events(sid=None, bid=None) -> list[Event]:
    sid = sid or uuid4()
    bid = bid or uuid4()
    return [
        Event(
            event_type=EventType.CLAIM,
            session_id=sid,
            branch_id=bid,
            author="gemini",
            evaluator=0.4,
            payload={
                "text": "Quantum entanglement enables teleportation",
                "topic_keywords": ["quantum", "entanglement"],
                "confidence": 0.3,  # low confidence → knowledge gap
            },
        ),
        Event(
            event_type=EventType.CLAIM,
            session_id=sid,
            branch_id=bid,
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "Photons are massless particles",
                "topic_keywords": ["photons", "particles"],
                "confidence": 0.9,  # high confidence → not a gap
            },
        ),
        Event(
            event_type=EventType.CONTRADICTION,
            session_id=sid,
            branch_id=bid,
            author="gemini",
            evaluator=0.45,
            payload={
                "concept_a": "quantum",
                "concept_b": "classical",
                "description": "Quantum vs classical interpretation",
                "resolved": False,
            },
        ),
        Event(
            event_type=EventType.SEED,
            session_id=sid,
            branch_id=bid,
            author="gemini",
            evaluator=0.5,
            payload={
                "hypothesis": "Entanglement can be harnessed for computing",
                "status": "active",
                "confidence": 0.4,
            },
        ),
    ]


class TestMeetingPrep:
    def test_prepare_generates_markdown(self):
        """prepare() must return a non-empty string that looks like markdown."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            prep = MeetingPrep(data_dir=tmpdir)
            result = prep.prepare(topic="quantum")

        assert isinstance(result, str)
        assert len(result) > 50
        assert "#" in result  # must have at least one markdown heading

    def test_knowledge_gaps_detected(self):
        """Concepts with confidence < 0.5 should be flagged as knowledge gaps."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            prep = MeetingPrep(data_dir=tmpdir)
            # inject events into the graph
            events = _make_events()
            gaps = prep._find_knowledge_gaps_from_events(events)

        # "quantum" and "entanglement" have confidence 0.3 < 0.5
        gap_names = [g["name"] for g in gaps]
        assert any(
            name in gap_names for name in ["quantum", "entanglement"]
        ), f"Expected low-confidence concepts in gaps, got: {gap_names}"

    def test_high_confidence_not_a_gap(self):
        """Concepts with confidence >= 0.5 should not appear in gaps."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            prep = MeetingPrep(data_dir=tmpdir)
            events = _make_events()
            gaps = prep._find_knowledge_gaps_from_events(events)

        gap_names = [g["name"] for g in gaps]
        assert "photons" not in gap_names
        assert "particles" not in gap_names

    def test_prepare_from_saved_events(self):
        """prepare() should load events from data/ and produce a report."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a minimal events export
            events = _make_events()
            events_data = [e.to_dict() for e in events]
            Path(tmpdir).joinpath("events_export.json").write_text(
                json.dumps(events_data)
            )

            prep = MeetingPrep(data_dir=tmpdir)
            result = prep.prepare(topic="")

        assert isinstance(result, str)
        assert "#" in result

    def test_empty_data_dir_returns_markdown(self):
        """When data/ has no events, prepare() still returns valid markdown."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            prep = MeetingPrep(data_dir=tmpdir)
            result = prep.prepare(topic="")

        assert isinstance(result, str)
        assert "#" in result

    def test_topic_filter_affects_questions(self):
        """Questions generated with a topic hint should reflect that topic."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            prep = MeetingPrep(data_dir=tmpdir)
            result_quantum = prep.prepare(topic="quantum")
            result_empty = prep.prepare(topic="")

        # Both should be valid markdown; topic variant should mention the topic
        assert "quantum" in result_quantum.lower()

    def test_contradictions_section_present(self):
        """When there are unresolved contradictions, they appear in the doc."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            events = _make_events()
            events_data = [e.to_dict() for e in events]
            Path(tmpdir).joinpath("events_export.json").write_text(
                json.dumps(events_data)
            )

            prep = MeetingPrep(data_dir=tmpdir)
            result = prep.prepare(topic="")

        # Should mention contradictions somehow
        assert "contradiction" in result.lower() or "Contradiction" in result

    def test_active_seeds_section_present(self):
        """Active seeds appear in the prep document."""
        from forgestream.meeting_prep import MeetingPrep

        with tempfile.TemporaryDirectory() as tmpdir:
            events = _make_events()
            events_data = [e.to_dict() for e in events]
            Path(tmpdir).joinpath("events_export.json").write_text(
                json.dumps(events_data)
            )

            prep = MeetingPrep(data_dir=tmpdir)
            result = prep.prepare(topic="")

        assert "seed" in result.lower() or "Seed" in result
