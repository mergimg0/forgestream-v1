"""StyleAdapter — adapts ForgeStream output based on accumulated UserProfile."""

from __future__ import annotations

from typing import Any

from .model import UserProfile

# Expressiveness thresholds for report style selection
_HIGH_EXPRESSIVENESS = 0.7
_LOW_EXPRESSIVENESS = 0.3

# Meeting duration threshold below which "extract" mode is suggested (minutes)
_SHORT_MEETING_THRESHOLD = 20.0


class StyleAdapter:
    """Reads a UserProfile and produces adaptation hints for report format,
    suggestion priorities, and recommended meeting mode.
    """

    def adapt(self, profile: UserProfile) -> dict[str, Any]:
        """Produce adaptation hints from a UserProfile.

        Returns a dict with keys:
            - report_style: "detailed" | "concise" | "balanced"
            - priority_adjustments: dict with boost_topics, deprioritize_categories,
              preferred_level
            - mode_suggestion: str — recommended meeting mode
        """
        return {
            "report_style": self._report_style(profile),
            "priority_adjustments": self._priority_adjustments(profile),
            "mode_suggestion": self._mode_suggestion(profile),
        }

    # --- Private helpers ---

    def _report_style(self, profile: UserProfile) -> str:
        """Map expressiveness_score to a report style."""
        if profile.expressiveness_score >= _HIGH_EXPRESSIVENESS:
            return "detailed"
        if profile.expressiveness_score <= _LOW_EXPRESSIVENESS:
            return "concise"
        return "balanced"

    def _priority_adjustments(self, profile: UserProfile) -> dict[str, Any]:
        """Build priority adjustments from engagement and responsiveness data."""
        return {
            "boost_topics": list(profile.peak_engagement_topics),
            "deprioritize_categories": list(profile.ignored_categories),
            "preferred_level": profile.preferred_priority_level,
        }

    def _mode_suggestion(self, profile: UserProfile) -> str:
        """Suggest a meeting mode based on usage habits.

        Rules (in priority order):
        1. Short meetings (avg < 20 min) → "extract"
        2. Preferred mode from profile → use it
        3. Default → "collaborative"
        """
        if profile.avg_meeting_duration_minutes < _SHORT_MEETING_THRESHOLD:
            return "extract"
        if profile.preferred_mode:
            return profile.preferred_mode
        return "collaborative"
