"""Tests for UserProfile dataclass — model, serialization, persistence."""

import json
import tempfile
from pathlib import Path

from forgestream.profile.model import UserProfile


class TestUserProfile:
    def test_default_profile(self):
        """UserProfile instantiates with correct defaults."""
        p = UserProfile()

        # Communication style defaults
        assert p.avg_arousal == 0.5
        assert p.avg_f0_variability == 0.0
        assert p.preferred_energy == 0.1
        assert p.expressiveness_score == 0.5

        # Engagement defaults
        assert p.peak_engagement_topics == []
        assert p.disengagement_triggers == []
        assert p.avg_meeting_engagement == 0.5
        assert p.engagement_trend == 0.0

        # Topic defaults
        assert p.topic_frequency == {}
        assert p.topic_depth == {}

        # Rapport defaults
        assert p.best_rapport_speakers == []
        assert p.rapport_component_weights == {}

        # Suggestion responsiveness defaults
        assert p.suggestion_uptake_rate == 0.5
        assert p.preferred_priority_level == "strategic"
        assert p.ignored_categories == []

        # Habits defaults
        assert p.avg_meeting_duration_minutes == 30.0
        assert p.preferred_mode == "collaborative"
        assert p.meetings_count == 0

        # Meta
        assert p.last_updated == ""

    def test_profile_serializes_to_json(self):
        """to_dict() produces a JSON-serializable dict with all fields."""
        p = UserProfile(
            avg_arousal=0.7,
            avg_f0_variability=0.3,
            preferred_energy=0.2,
            expressiveness_score=0.65,
            peak_engagement_topics=["AI", "latency"],
            disengagement_triggers=["budget"],
            avg_meeting_engagement=0.6,
            engagement_trend=0.05,
            topic_frequency={"Kafka": 3, "latency": 2},
            topic_depth={"Kafka": 2.5},
            best_rapport_speakers=["alice"],
            rapport_component_weights={"attentiveness": 0.4},
            suggestion_uptake_rate=0.8,
            preferred_priority_level="tactical",
            ignored_categories=["admin"],
            avg_meeting_duration_minutes=45.0,
            preferred_mode="extract",
            meetings_count=5,
            last_updated="2026-03-28T00:00:00Z",
        )

        d = p.to_dict()

        # Verify all expected keys exist
        assert "avg_arousal" in d
        assert "avg_f0_variability" in d
        assert "preferred_energy" in d
        assert "expressiveness_score" in d
        assert "peak_engagement_topics" in d
        assert "disengagement_triggers" in d
        assert "avg_meeting_engagement" in d
        assert "engagement_trend" in d
        assert "topic_frequency" in d
        assert "topic_depth" in d
        assert "best_rapport_speakers" in d
        assert "rapport_component_weights" in d
        assert "suggestion_uptake_rate" in d
        assert "preferred_priority_level" in d
        assert "ignored_categories" in d
        assert "avg_meeting_duration_minutes" in d
        assert "preferred_mode" in d
        assert "meetings_count" in d
        assert "last_updated" in d

        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # Spot-check values
        assert d["avg_arousal"] == 0.7
        assert d["peak_engagement_topics"] == ["AI", "latency"]
        assert d["topic_frequency"] == {"Kafka": 3, "latency": 2}
        assert d["meetings_count"] == 5

    def test_profile_loads_from_json(self):
        """from_dict() correctly reconstructs a UserProfile from a dict."""
        p = UserProfile(
            avg_arousal=0.7,
            peak_engagement_topics=["AI"],
            topic_frequency={"Kafka": 3},
            meetings_count=5,
            last_updated="2026-03-28T00:00:00Z",
        )
        d = p.to_dict()
        restored = UserProfile.from_dict(d)

        assert restored.avg_arousal == 0.7
        assert restored.peak_engagement_topics == ["AI"]
        assert restored.topic_frequency == {"Kafka": 3}
        assert restored.meetings_count == 5
        assert restored.last_updated == "2026-03-28T00:00:00Z"

        # Defaults preserved
        assert restored.preferred_priority_level == "strategic"
        assert restored.avg_meeting_engagement == 0.5

    def test_profile_save_and_load(self):
        """save() writes to disk; load() reads it back correctly."""
        p = UserProfile(
            avg_arousal=0.8,
            meetings_count=3,
            preferred_mode="extract",
            last_updated="2026-03-28T00:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "user_profile.json"
            p.save(str(path))

            assert path.exists()
            restored = UserProfile.load(str(path))

            assert restored.avg_arousal == 0.8
            assert restored.meetings_count == 3
            assert restored.preferred_mode == "extract"

    def test_load_missing_file_returns_default(self):
        """load() returns a default UserProfile when file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            p = UserProfile.load(str(path))
            # Should be a fresh default profile
            assert p.avg_arousal == 0.5
            assert p.meetings_count == 0
