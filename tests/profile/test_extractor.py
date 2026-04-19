"""Tests for UserProfileExtractor — builds/updates user profile from events."""

from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.profile.extractor import UserProfileExtractor
from forgestream.profile.model import UserProfile


def _make_event(
    event_type: EventType,
    payload: dict,
    author: str = "user",
    evaluator: float = 0.5,
) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author=author,
        evaluator=evaluator,
        payload=payload,
    )


def _make_prosodic(
    arousal: float = 0.5,
    f0_std: float = 20.0,
    energy: float = 0.1,
    speaker: str = "user",
) -> Event:
    return _make_event(
        EventType.PROSODIC_FEATURE,
        {"arousal": arousal, "f0_std": f0_std, "energy": energy, "speaker": speaker},
    )


def _make_claim(keywords: list[str], author: str = "gemini") -> Event:
    return _make_event(
        EventType.CLAIM,
        {"text": "some claim", "topic_keywords": keywords, "confidence": 0.8},
        author=author,
    )


def _make_rapport_score(pair: tuple[str, str], composite: float) -> Event:
    return _make_event(
        EventType.RAPPORT_SCORE,
        {"pair_scores": {f"{pair[0]}_{pair[1]}": composite}, "speaker_a": pair[0], "speaker_b": pair[1]},
    )


class TestUserProfileExtractor:
    def test_update_communication_style(self):
        """Extractor computes avg_arousal, f0_variability, energy from prosodic events."""
        extractor = UserProfileExtractor()
        profile = UserProfile()

        prosodic_events = [
            _make_prosodic(arousal=0.6, f0_std=25.0, energy=0.15),
            _make_prosodic(arousal=0.8, f0_std=35.0, energy=0.20),
        ]
        events = prosodic_events

        updated = extractor.update(events, profile)

        # EMA applied: alpha=0.2, starting from defaults (0.5, 0.0, 0.1)
        # Meeting arousal avg = (0.6 + 0.8) / 2 = 0.7
        # EMA: 0.2 * 0.7 + 0.8 * 0.5 = 0.14 + 0.40 = 0.54
        assert updated.avg_arousal > 0.5, "arousal should increase from baseline 0.5"
        assert updated.avg_arousal < 0.8, "arousal should not jump to full meeting value"

        # f0 variability should be updated
        assert updated.avg_f0_variability > 0.0

        # expressiveness_score should be positive
        assert 0.0 < updated.expressiveness_score < 1.0

    def test_update_engagement_signature(self):
        """Extractor identifies peak engagement topics from high-arousal prosodic + claim co-occurrence."""
        extractor = UserProfileExtractor()
        profile = UserProfile()

        # High-arousal window + claim with keywords
        events = [
            _make_prosodic(arousal=0.8),  # high arousal
            _make_claim(["AI", "machine learning"]),  # claim co-occurring
            _make_prosodic(arousal=0.3),  # low arousal (disengagement)
            _make_claim(["budget", "admin"]),  # claim during disengagement
        ]

        updated = extractor.update(events, profile)

        # High-arousal topics should appear in peak_engagement_topics
        assert "AI" in updated.peak_engagement_topics or "machine learning" in updated.peak_engagement_topics

        # Low-arousal topics may appear in disengagement_triggers
        assert "budget" in updated.disengagement_triggers or "admin" in updated.disengagement_triggers

    def test_update_topic_preferences(self):
        """Extractor counts keyword frequency and depth from CLAIM events."""
        extractor = UserProfileExtractor()
        profile = UserProfile(topic_frequency={"Kafka": 2})

        events = [
            _make_claim(["Kafka", "streaming"]),
            _make_claim(["Kafka"]),
            _make_claim(["latency", "streaming"]),
        ]

        updated = extractor.update(events, profile)

        # Kafka was 2 before, plus 2 new occurrences
        assert updated.topic_frequency.get("Kafka", 0) >= 4
        assert updated.topic_frequency.get("streaming", 0) >= 2
        assert updated.topic_frequency.get("latency", 0) >= 1

        # topic_depth should be set for any topic
        assert len(updated.topic_depth) > 0

    def test_ema_smoothing_across_updates(self):
        """Multiple update() calls use EMA — numeric fields converge toward the signal."""
        extractor = UserProfileExtractor()
        profile = UserProfile(avg_arousal=0.5)

        # Simulate 5 meetings with high arousal
        for _ in range(5):
            events = [_make_prosodic(arousal=0.9)]
            profile = extractor.update(events, profile)

        # After 5 updates, avg_arousal should be approaching 0.9
        # EMA with alpha=0.2: 0.5 -> 0.58 -> 0.664 -> 0.731 -> 0.785 -> 0.828
        assert profile.avg_arousal > 0.65, "should have converged toward high arousal"
        assert profile.avg_arousal < 0.95, "should not overshoot"

    def test_empty_events_no_crash(self):
        """update() with empty events list does not crash and returns a valid profile."""
        extractor = UserProfileExtractor()
        profile = UserProfile(meetings_count=3, avg_arousal=0.6)

        updated = extractor.update([], profile)

        # meetings_count should increment
        assert updated.meetings_count == 4

        # Other fields should be preserved (no prosodic data = no change)
        assert updated.avg_arousal == 0.6

    def test_meetings_count_increments(self):
        """update() increments meetings_count by 1."""
        extractor = UserProfileExtractor()
        profile = UserProfile(meetings_count=7)

        updated = extractor.update([], profile)
        assert updated.meetings_count == 8

    def test_last_updated_is_set(self):
        """update() sets last_updated to a non-empty ISO timestamp."""
        extractor = UserProfileExtractor()
        profile = UserProfile()

        updated = extractor.update([], profile)
        assert updated.last_updated != ""
        # Should be a valid ISO date string
        assert "2026" in updated.last_updated or "T" in updated.last_updated

    def test_rapport_affinity_extraction(self):
        """Best rapport speakers are identified from RAPPORT_SCORE events."""
        extractor = UserProfileExtractor()
        profile = UserProfile()

        events = [
            _make_rapport_score(("user", "alice"), 0.85),
            _make_rapport_score(("user", "bob"), 0.40),
            _make_rapport_score(("user", "alice"), 0.90),
        ]

        updated = extractor.update(events, profile)

        # alice should appear in best_rapport_speakers (consistently high)
        assert "alice" in updated.best_rapport_speakers
