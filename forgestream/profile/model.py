"""UserProfile dataclass — persistent behavioral profile accumulated across meetings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserProfile:
    """Persistent user profile accumulated across meetings.

    Numeric fields use EMA smoothing (alpha=0.2) across updates.
    List fields use set-union semantics (items are added, never removed).
    """

    # Communication style
    avg_arousal: float = 0.5            # baseline emotional expressiveness
    avg_f0_variability: float = 0.0     # pitch dynamism
    preferred_energy: float = 0.1       # typical speaking energy
    expressiveness_score: float = 0.5   # composite: high = animated, low = measured

    # Engagement signature
    peak_engagement_topics: list[str] = field(default_factory=list)
    disengagement_triggers: list[str] = field(default_factory=list)
    avg_meeting_engagement: float = 0.5
    engagement_trend: float = 0.0       # across meetings: improving or declining

    # Topic preferences
    topic_frequency: dict[str, int] = field(default_factory=dict)
    topic_depth: dict[str, float] = field(default_factory=dict)

    # Rapport affinity
    best_rapport_speakers: list[str] = field(default_factory=list)
    rapport_component_weights: dict[str, float] = field(default_factory=dict)

    # Suggestion responsiveness
    suggestion_uptake_rate: float = 0.5
    preferred_priority_level: str = "strategic"
    ignored_categories: list[str] = field(default_factory=list)

    # Habits
    avg_meeting_duration_minutes: float = 30.0
    preferred_mode: str = "collaborative"
    meetings_count: int = 0

    # Meta
    last_updated: str = ""

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "avg_arousal": self.avg_arousal,
            "avg_f0_variability": self.avg_f0_variability,
            "preferred_energy": self.preferred_energy,
            "expressiveness_score": self.expressiveness_score,
            "peak_engagement_topics": list(self.peak_engagement_topics),
            "disengagement_triggers": list(self.disengagement_triggers),
            "avg_meeting_engagement": self.avg_meeting_engagement,
            "engagement_trend": self.engagement_trend,
            "topic_frequency": dict(self.topic_frequency),
            "topic_depth": dict(self.topic_depth),
            "best_rapport_speakers": list(self.best_rapport_speakers),
            "rapport_component_weights": dict(self.rapport_component_weights),
            "suggestion_uptake_rate": self.suggestion_uptake_rate,
            "preferred_priority_level": self.preferred_priority_level,
            "ignored_categories": list(self.ignored_categories),
            "avg_meeting_duration_minutes": self.avg_meeting_duration_minutes,
            "preferred_mode": self.preferred_mode,
            "meetings_count": self.meetings_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UserProfile:
        """Reconstruct a UserProfile from a serialized dict."""
        p = cls()
        p.avg_arousal = float(d.get("avg_arousal", p.avg_arousal))
        p.avg_f0_variability = float(d.get("avg_f0_variability", p.avg_f0_variability))
        p.preferred_energy = float(d.get("preferred_energy", p.preferred_energy))
        p.expressiveness_score = float(d.get("expressiveness_score", p.expressiveness_score))
        p.peak_engagement_topics = list(d.get("peak_engagement_topics", []))
        p.disengagement_triggers = list(d.get("disengagement_triggers", []))
        p.avg_meeting_engagement = float(d.get("avg_meeting_engagement", p.avg_meeting_engagement))
        p.engagement_trend = float(d.get("engagement_trend", p.engagement_trend))
        p.topic_frequency = dict(d.get("topic_frequency", {}))
        p.topic_depth = dict(d.get("topic_depth", {}))
        p.best_rapport_speakers = list(d.get("best_rapport_speakers", []))
        p.rapport_component_weights = dict(d.get("rapport_component_weights", {}))
        p.suggestion_uptake_rate = float(d.get("suggestion_uptake_rate", p.suggestion_uptake_rate))
        p.preferred_priority_level = str(d.get("preferred_priority_level", p.preferred_priority_level))
        p.ignored_categories = list(d.get("ignored_categories", []))
        p.avg_meeting_duration_minutes = float(
            d.get("avg_meeting_duration_minutes", p.avg_meeting_duration_minutes)
        )
        p.preferred_mode = str(d.get("preferred_mode", p.preferred_mode))
        p.meetings_count = int(d.get("meetings_count", p.meetings_count))
        p.last_updated = str(d.get("last_updated", p.last_updated))
        return p

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Write profile to JSON file at path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> UserProfile:
        """Load profile from JSON file. Returns default profile if file does not exist."""
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text()))
