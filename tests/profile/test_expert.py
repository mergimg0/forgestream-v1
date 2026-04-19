"""Tests for ExpertProfile + ExpertProfileManager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType


def _make_claim(speaker: str, keywords: list[str], confidence: float = 0.8, sid=None, bid=None) -> Event:
    sid = sid or uuid4()
    bid = bid or uuid4()
    return Event(
        event_type=EventType.CLAIM,
        session_id=sid,
        branch_id=bid,
        author="gemini",
        evaluator=0.5,
        payload={
            "text": f"Claim about {keywords}",
            "speaker_id": speaker,
            "topic_keywords": keywords,
            "confidence": confidence,
        },
    )


def _make_prosodic(speaker: str, arousal: float = 0.6, f0_var: float = 0.3, energy: float = 0.4, sid=None, bid=None) -> Event:
    sid = sid or uuid4()
    bid = bid or uuid4()
    return Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=sid,
        branch_id=bid,
        author="audio",
        evaluator=0.5,
        payload={
            "speaker_id": speaker,
            "arousal": arousal,
            "f0_variability": f0_var,
            "energy": energy,
        },
    )


def _make_rapport(speaker: str, score: float = 0.7, sid=None, bid=None) -> Event:
    sid = sid or uuid4()
    bid = bid or uuid4()
    return Event(
        event_type=EventType.RAPPORT_SCORE,
        session_id=sid,
        branch_id=bid,
        author="rapport",
        evaluator=0.5,
        payload={
            "speaker_id": speaker,
            "rapport_score": score,
        },
    )


class TestExpertProfile:
    def test_default_profile(self):
        """ExpertProfile should have sensible defaults."""
        from forgestream.profile.expert import ExpertProfile

        ep = ExpertProfile(speaker_id="alice")
        assert ep.speaker_id == "alice"
        assert ep.expertise_topics == {}
        assert ep.communication_style == {}
        assert ep.rapport_with_user == 0.5
        assert ep.meetings_count == 0
        assert ep.total_claims == 0

    def test_to_dict_roundtrip(self):
        """to_dict / from_dict roundtrip should preserve all fields."""
        from forgestream.profile.expert import ExpertProfile

        ep = ExpertProfile(
            speaker_id="bob",
            expertise_topics={"quantum": 0.9, "ml": 0.7},
            communication_style={"arousal": 0.6, "f0_var": 0.3},
            rapport_with_user=0.8,
            meetings_count=3,
            total_claims=42,
        )
        d = ep.to_dict()
        ep2 = ExpertProfile.from_dict(d)

        assert ep2.speaker_id == "bob"
        assert ep2.expertise_topics == {"quantum": 0.9, "ml": 0.7}
        assert ep2.communication_style == {"arousal": 0.6, "f0_var": 0.3}
        assert abs(ep2.rapport_with_user - 0.8) < 1e-9
        assert ep2.meetings_count == 3
        assert ep2.total_claims == 42

    def test_save_load(self):
        """save/load should persist and restore an ExpertProfile."""
        from forgestream.profile.expert import ExpertProfile, ExpertProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            ep = ExpertProfile(
                speaker_id="carol",
                expertise_topics={"nlp": 0.8},
                rapport_with_user=0.65,
                meetings_count=2,
            )
            mgr.save_profile(ep)

            loaded = mgr.load_profile("carol")
            assert loaded.speaker_id == "carol"
            assert loaded.expertise_topics == {"nlp": 0.8}
            assert abs(loaded.rapport_with_user - 0.65) < 1e-9
            assert loaded.meetings_count == 2

    def test_load_missing_returns_default(self):
        """load_profile for unknown speaker returns a default profile."""
        from forgestream.profile.expert import ExpertProfile, ExpertProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            ep = mgr.load_profile("unknown_speaker")

        assert ep.speaker_id == "unknown_speaker"
        assert ep.expertise_topics == {}
        assert ep.rapport_with_user == 0.5


class TestExpertProfileManager:
    def test_update_from_events_claim_topics(self):
        """CLAIM events should increment topic frequency for the speaker."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events = [
            _make_claim("alice", ["quantum", "physics"], sid=sid, bid=bid),
            _make_claim("alice", ["quantum", "computing"], sid=sid, bid=bid),
            _make_claim("bob", ["ml", "nlp"], sid=sid, bid=bid),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            profiles = mgr.update_from_events(events)

        alice = next(p for p in profiles if p.speaker_id == "alice")
        assert "quantum" in alice.expertise_topics
        assert alice.expertise_topics["quantum"] > 0

    def test_update_from_events_prosodic_style(self):
        """PROSODIC_FEATURE events should update communication_style."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events = [
            _make_prosodic("alice", arousal=0.7, f0_var=0.4, energy=0.5, sid=sid, bid=bid),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            profiles = mgr.update_from_events(events)

        alice = next(p for p in profiles if p.speaker_id == "alice")
        assert "arousal" in alice.communication_style
        assert abs(alice.communication_style["arousal"] - 0.7) < 0.01

    def test_update_from_events_rapport(self):
        """RAPPORT_SCORE events should update rapport_with_user."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events = [
            _make_rapport("alice", score=0.8, sid=sid, bid=bid),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            profiles = mgr.update_from_events(events)

        alice = next(p for p in profiles if p.speaker_id == "alice")
        # EMA: starts at 0.5, updated toward 0.8
        assert alice.rapport_with_user > 0.5

    def test_expertise_accumulates_across_meetings(self):
        """Calling update_from_events twice should accumulate expertise."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events1 = [_make_claim("alice", ["quantum"], sid=sid, bid=bid)]
        events2 = [_make_claim("alice", ["quantum", "photons"], sid=sid, bid=bid)]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            mgr.update_from_events(events1)
            profiles2 = mgr.update_from_events(events2)

        alice = next(p for p in profiles2 if p.speaker_id == "alice")
        assert alice.total_claims >= 2
        assert "quantum" in alice.expertise_topics

    def test_meetings_count_increments(self):
        """Each update_from_events call should increment meetings_count."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events = [_make_claim("alice", ["topic"], sid=sid, bid=bid)]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            mgr.update_from_events(events)
            mgr.update_from_events(events)
            profiles = mgr.update_from_events(events)

        alice = next(p for p in profiles if p.speaker_id == "alice")
        assert alice.meetings_count == 3

    def test_get_expert_for_topic(self):
        """get_expert_for_topic returns speaker with highest expertise in that topic."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()
        events = [
            _make_claim("alice", ["quantum", "physics"], sid=sid, bid=bid),
            _make_claim("alice", ["quantum"], sid=sid, bid=bid),
            _make_claim("alice", ["quantum"], sid=sid, bid=bid),
            _make_claim("bob", ["ml"], sid=sid, bid=bid),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            mgr.update_from_events(events)
            expert = mgr.get_expert_for_topic("quantum")

        assert expert is not None
        assert expert.speaker_id == "alice"

    def test_get_expert_for_topic_none_when_no_profiles(self):
        """get_expert_for_topic returns None if no profiles exist."""
        from forgestream.profile.expert import ExpertProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            result = mgr.get_expert_for_topic("quantum")

        assert result is None

    def test_ema_smoothing_rapport(self):
        """Multiple rapport updates should smooth via EMA, not spike."""
        from forgestream.profile.expert import ExpertProfileManager

        sid, bid = uuid4(), uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            # First meeting: low rapport
            mgr.update_from_events([_make_rapport("alice", score=0.1, sid=sid, bid=bid)])
            # Second meeting: high rapport
            profiles = mgr.update_from_events([_make_rapport("alice", score=0.9, sid=sid, bid=bid)])

        alice = next(p for p in profiles if p.speaker_id == "alice")
        # Should be somewhere between 0.1 and 0.9 (EMA), not snapped to 0.9
        assert 0.1 < alice.rapport_with_user < 0.9

    def test_no_events_returns_empty_list(self):
        """update_from_events with empty list returns empty profiles list."""
        from forgestream.profile.expert import ExpertProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertProfileManager(profiles_dir=tmpdir)
            profiles = mgr.update_from_events([])

        assert profiles == []
