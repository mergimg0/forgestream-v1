"""Expert profile model and manager — per-speaker expertise accumulated across meetings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events.schema import Event, EventType


# EMA smoothing factor for numeric fields updated across meetings.
_EMA_ALPHA: float = 0.3


@dataclass
class ExpertProfile:
    """Persistent per-speaker profile accumulated across meetings.

    Numeric fields use EMA smoothing (alpha=0.3) across meeting updates.
    Dict fields accumulate monotonically.
    """

    speaker_id: str

    # topic → normalized frequency score (0–1)
    expertise_topics: dict[str, float] = field(default_factory=dict)

    # prosodic style: "arousal", "f0_var", "energy"
    communication_style: dict[str, float] = field(default_factory=dict)

    rapport_with_user: float = 0.5
    meetings_count: int = 0
    total_claims: int = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "expertise_topics": dict(self.expertise_topics),
            "communication_style": dict(self.communication_style),
            "rapport_with_user": self.rapport_with_user,
            "meetings_count": self.meetings_count,
            "total_claims": self.total_claims,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExpertProfile:
        return cls(
            speaker_id=str(d["speaker_id"]),
            expertise_topics=dict(d.get("expertise_topics", {})),
            communication_style=dict(d.get("communication_style", {})),
            rapport_with_user=float(d.get("rapport_with_user", 0.5)),
            meetings_count=int(d.get("meetings_count", 0)),
            total_claims=int(d.get("total_claims", 0)),
        )


class ExpertProfileManager:
    """Load, update, and persist ExpertProfile objects.

    Profiles are stored as individual JSON files under profiles_dir:
    ``<profiles_dir>/<speaker_id>.json``
    """

    def __init__(self, profiles_dir: str = "data/expert_profiles") -> None:
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_profile(self, speaker_id: str) -> ExpertProfile:
        """Load a profile from disk; returns a default profile if not found."""
        path = self.profiles_dir / f"{speaker_id}.json"
        if not path.exists():
            return ExpertProfile(speaker_id=speaker_id)
        return ExpertProfile.from_dict(json.loads(path.read_text()))

    def save_profile(self, profile: ExpertProfile) -> None:
        """Persist a profile to disk."""
        path = self.profiles_dir / f"{profile.speaker_id}.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Update logic
    # ------------------------------------------------------------------

    def update_from_events(self, events: list[Event]) -> list[ExpertProfile]:
        """Update all speaker profiles from a set of meeting events.

        Groups events by speaker, applies EMA smoothing on numeric fields,
        saves updated profiles, and returns all updated profiles.

        Event types processed:
        - CLAIM: topic_keywords → expertise_topics frequency
        - PROSODIC_FEATURE: arousal / f0_variability / energy → communication_style
        - RAPPORT_SCORE: rapport_score → rapport_with_user (EMA)
        """
        if not events:
            return []

        # Collect per-speaker data from this meeting
        topic_counts: dict[str, dict[str, int]] = {}   # speaker → topic → count
        prosodic_sums: dict[str, dict[str, list[float]]] = {}  # speaker → field → values
        rapport_scores: dict[str, list[float]] = {}
        claim_counts: dict[str, int] = {}

        for event in events:
            if event.event_type == EventType.CLAIM:
                speaker = event.payload.get("speaker_id", "")
                if not speaker:
                    continue
                keywords = event.payload.get("topic_keywords", [])
                tc = topic_counts.setdefault(speaker, {})
                for kw in keywords:
                    tc[kw] = tc.get(kw, 0) + 1
                claim_counts[speaker] = claim_counts.get(speaker, 0) + 1

            elif event.event_type == EventType.PROSODIC_FEATURE:
                speaker = event.payload.get("speaker_id", "")
                if not speaker:
                    continue
                ps = prosodic_sums.setdefault(speaker, {})
                for field_name, payload_key in [
                    ("arousal", "arousal"),
                    ("f0_var", "f0_variability"),
                    ("energy", "energy"),
                ]:
                    val = event.payload.get(payload_key)
                    if val is not None:
                        ps.setdefault(field_name, []).append(float(val))

            elif event.event_type == EventType.RAPPORT_SCORE:
                speaker = event.payload.get("speaker_id", "")
                if not speaker:
                    continue
                score = event.payload.get("rapport_score")
                if score is not None:
                    rapport_scores.setdefault(speaker, []).append(float(score))

        # Collect all speakers seen in this meeting
        all_speakers: set[str] = (
            set(topic_counts.keys())
            | set(prosodic_sums.keys())
            | set(rapport_scores.keys())
        )

        updated: list[ExpertProfile] = []
        for speaker in all_speakers:
            profile = self.load_profile(speaker)
            profile.meetings_count += 1

            # --- Expertise topics ---
            if speaker in topic_counts:
                tc = topic_counts[speaker]
                total = sum(tc.values()) or 1
                for kw, count in tc.items():
                    new_score = count / total
                    existing = profile.expertise_topics.get(kw, 0.0)
                    # EMA blend
                    profile.expertise_topics[kw] = (
                        (1 - _EMA_ALPHA) * existing + _EMA_ALPHA * new_score
                    )
                profile.total_claims += claim_counts.get(speaker, 0)

            # --- Communication style ---
            if speaker in prosodic_sums:
                ps = prosodic_sums[speaker]
                for field_name, vals in ps.items():
                    mean_val = sum(vals) / len(vals)
                    existing = profile.communication_style.get(field_name, mean_val)
                    profile.communication_style[field_name] = (
                        (1 - _EMA_ALPHA) * existing + _EMA_ALPHA * mean_val
                    )

            # --- Rapport ---
            if speaker in rapport_scores:
                scores = rapport_scores[speaker]
                mean_rapport = sum(scores) / len(scores)
                profile.rapport_with_user = (
                    (1 - _EMA_ALPHA) * profile.rapport_with_user
                    + _EMA_ALPHA * mean_rapport
                )

            self.save_profile(profile)
            updated.append(profile)

        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_expert_for_topic(self, topic: str) -> ExpertProfile | None:
        """Return the speaker with the highest expertise score for a given topic.

        Returns None if no profiles exist or no speaker has expertise in that topic.
        """
        best: ExpertProfile | None = None
        best_score: float = -1.0

        for path in self.profiles_dir.glob("*.json"):
            try:
                profile = ExpertProfile.from_dict(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue
            score = profile.expertise_topics.get(topic, 0.0)
            if score > best_score:
                best_score = score
                best = profile

        return best if best_score > 0.0 else None
