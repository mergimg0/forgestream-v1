"""Tests for StyleAdapter — adapts report format and priorities based on UserProfile."""

from forgestream.profile.adaptation import StyleAdapter
from forgestream.profile.model import UserProfile


class TestStyleAdapter:
    def test_high_expressiveness_gets_detailed_style(self):
        """High expressiveness_score (>= 0.7) yields report_style = 'detailed'."""
        adapter = StyleAdapter()
        profile = UserProfile(expressiveness_score=0.8)

        result = adapter.adapt(profile)

        assert result["report_style"] == "detailed"

    def test_low_expressiveness_gets_concise_style(self):
        """Low expressiveness_score (<= 0.3) yields report_style = 'concise'."""
        adapter = StyleAdapter()
        profile = UserProfile(expressiveness_score=0.2)

        result = adapter.adapt(profile)

        assert result["report_style"] == "concise"

    def test_default_profile_gets_balanced(self):
        """Default expressiveness_score (0.5) yields report_style = 'balanced'."""
        adapter = StyleAdapter()
        profile = UserProfile()  # expressiveness_score=0.5

        result = adapter.adapt(profile)

        assert result["report_style"] == "balanced"

    def test_adapt_returns_priority_adjustments(self):
        """adapt() returns priority_adjustments dict."""
        adapter = StyleAdapter()
        profile = UserProfile(
            peak_engagement_topics=["AI", "performance"],
            ignored_categories=["admin"],
            preferred_priority_level="tactical",
        )

        result = adapter.adapt(profile)

        assert "priority_adjustments" in result
        pa = result["priority_adjustments"]
        assert isinstance(pa, dict)

        # Peak engagement topics should be boosted
        assert pa.get("boost_topics") == ["AI", "performance"]

        # Ignored categories should be deprioritized
        assert pa.get("deprioritize_categories") == ["admin"]

        # Preferred priority level should be reflected
        assert pa.get("preferred_level") == "tactical"

    def test_adapt_returns_mode_suggestion(self):
        """adapt() returns mode_suggestion based on profile habits."""
        adapter = StyleAdapter()

        # Consistent collaborative user
        profile_collab = UserProfile(preferred_mode="collaborative", meetings_count=5)
        result_collab = adapter.adapt(profile_collab)
        assert result_collab["mode_suggestion"] == "collaborative"

        # Short-duration meetings -> suggest extract
        profile_short = UserProfile(
            avg_meeting_duration_minutes=10.0,
            preferred_mode="extract",
            meetings_count=3,
        )
        result_short = adapter.adapt(profile_short)
        assert result_short["mode_suggestion"] == "extract"

    def test_adapt_with_zero_meetings_returns_defaults(self):
        """adapt() with meetings_count=0 (new user) returns safe defaults."""
        adapter = StyleAdapter()
        profile = UserProfile(meetings_count=0)

        result = adapter.adapt(profile)

        assert result["report_style"] in {"balanced", "detailed", "concise"}
        assert "priority_adjustments" in result
        assert "mode_suggestion" in result

    def test_expressiveness_boundary_at_07(self):
        """expressiveness_score exactly 0.7 yields 'detailed'."""
        adapter = StyleAdapter()
        profile = UserProfile(expressiveness_score=0.7)
        result = adapter.adapt(profile)
        assert result["report_style"] == "detailed"

    def test_expressiveness_boundary_at_03(self):
        """expressiveness_score exactly 0.3 yields 'concise'."""
        adapter = StyleAdapter()
        profile = UserProfile(expressiveness_score=0.3)
        result = adapter.adapt(profile)
        assert result["report_style"] == "concise"
