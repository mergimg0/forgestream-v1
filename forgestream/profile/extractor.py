"""UserProfileExtractor — builds and updates UserProfile from meeting events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..events.schema import Event, EventType
from .model import UserProfile

# EMA smoothing factor — recent meetings weighted more
_ALPHA = 0.2
# Arousal threshold to consider a window "high engagement"
_HIGH_AROUSAL_THRESHOLD = 0.7
# Arousal threshold below which we consider disengagement
_LOW_AROUSAL_THRESHOLD = 0.4
# Rapport composite threshold to include a speaker in best_rapport_speakers
_RAPPORT_THRESHOLD = 0.65


def _ema(current: float, new_value: float, alpha: float = _ALPHA) -> float:
    """Exponential moving average: blends new_value into current."""
    return alpha * new_value + (1.0 - alpha) * current


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] given expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


class UserProfileExtractor:
    """Updates a UserProfile with data extracted from one meeting's events.

    Numeric fields are smoothed with EMA (alpha=0.2).
    List fields use set-union semantics.
    """

    def update(self, events: list[Event], current_profile: UserProfile) -> UserProfile:
        """Update profile with data from this meeting. Returns a new UserProfile."""
        # Work on a copy of current state
        p = UserProfile.from_dict(current_profile.to_dict())

        prosodic_events = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]
        claim_events = [e for e in events if e.event_type == EventType.CLAIM]
        rapport_events = [e for e in events if e.event_type == EventType.RAPPORT_SCORE]

        if prosodic_events:
            comm_style = self._extract_communication_style(prosodic_events)
            p.avg_arousal = _ema(p.avg_arousal, comm_style["avg_arousal"])
            p.avg_f0_variability = _ema(p.avg_f0_variability, comm_style["avg_f0_variability"])
            p.preferred_energy = _ema(p.preferred_energy, comm_style["avg_energy"])
            p.expressiveness_score = _ema(
                p.expressiveness_score, comm_style["expressiveness_score"]
            )

        if events:
            engagement = self._extract_engagement_signature(prosodic_events, claim_events)
            # Union engagement topics
            for topic in engagement.get("peak_engagement_topics", []):
                if topic not in p.peak_engagement_topics:
                    p.peak_engagement_topics.append(topic)
            for topic in engagement.get("disengagement_triggers", []):
                if topic not in p.disengagement_triggers:
                    p.disengagement_triggers.append(topic)
            if "avg_meeting_engagement" in engagement:
                p.avg_meeting_engagement = _ema(
                    p.avg_meeting_engagement, engagement["avg_meeting_engagement"]
                )

        if claim_events:
            topic_prefs = self._extract_topic_preferences(claim_events)
            # Accumulate frequency counts
            for kw, count in topic_prefs.get("frequency", {}).items():
                p.topic_frequency[kw] = p.topic_frequency.get(kw, 0) + count
            # Update depth via EMA per keyword
            for kw, depth in topic_prefs.get("depth", {}).items():
                if kw in p.topic_depth:
                    p.topic_depth[kw] = _ema(p.topic_depth[kw], depth)
                else:
                    p.topic_depth[kw] = depth

        if rapport_events:
            rapport_info = self._extract_rapport_affinity(rapport_events)
            for speaker in rapport_info.get("best_speakers", []):
                if speaker not in p.best_rapport_speakers:
                    p.best_rapport_speakers.append(speaker)
            for k, v in rapport_info.get("component_weights", {}).items():
                p.rapport_component_weights[k] = v

        # Increment meeting count and stamp timestamp
        p.meetings_count += 1
        p.last_updated = datetime.now(timezone.utc).isoformat()

        return p

    def _extract_communication_style(self, prosodic_events: list[Event]) -> dict[str, float]:
        """Compute mean prosodic features and expressiveness score."""
        arousals = [e.payload.get("arousal", 0.5) for e in prosodic_events]
        f0_stds = [e.payload.get("f0_std", 0.0) for e in prosodic_events]
        energies = [e.payload.get("energy", 0.1) for e in prosodic_events]

        avg_arousal = sum(arousals) / len(arousals)
        avg_f0_var = sum(f0_stds) / len(f0_stds)
        avg_energy = sum(energies) / len(energies)

        # Normalize each dimension to [0, 1] for the composite score
        # Typical f0_std range: 0–100 Hz; energy range: 0–1.0
        norm_arousal = _normalize(avg_arousal, 0.0, 1.0)
        norm_f0 = _normalize(avg_f0_var, 0.0, 100.0)
        norm_energy = _normalize(avg_energy, 0.0, 1.0)

        expressiveness = (
            0.4 * norm_arousal
            + 0.3 * norm_f0
            + 0.3 * norm_energy
        )

        return {
            "avg_arousal": avg_arousal,
            "avg_f0_variability": avg_f0_var,
            "avg_energy": avg_energy,
            "expressiveness_score": expressiveness,
        }

    def _extract_engagement_signature(
        self,
        prosodic_events: list[Event],
        claim_events: list[Event],
    ) -> dict[str, Any]:
        """Identify peak-engagement and disengagement topics and overall engagement level."""
        peak_topics: list[str] = []
        disengage_topics: list[str] = []

        if prosodic_events and claim_events:
            # Determine mean arousal of prosodic window surrounding each claim
            # Simple heuristic: if any prosodic event in the sequence has high arousal,
            # the next claim's keywords are engagement topics
            high_arousal_flag = False
            low_arousal_flag = False

            for event in prosodic_events:
                ar = event.payload.get("arousal", 0.5)
                if ar >= _HIGH_AROUSAL_THRESHOLD:
                    high_arousal_flag = True
                elif ar <= _LOW_AROUSAL_THRESHOLD:
                    low_arousal_flag = True

            if high_arousal_flag:
                # Find claims closest in sequence to high-arousal windows
                for claim in claim_events:
                    for kw in claim.payload.get("topic_keywords", []):
                        if kw and kw not in peak_topics:
                            peak_topics.append(kw)
                # Limit to first 3 to avoid over-populating
                # But only take claims that come after a high-arousal prosodic event
                # Full implementation: use timestamps; here use positional heuristic
                peak_topics = _topics_near_arousal(
                    prosodic_events, claim_events, high=True
                )

            if low_arousal_flag:
                disengage_topics = _topics_near_arousal(
                    prosodic_events, claim_events, high=False
                )

        arousals = [e.payload.get("arousal", 0.5) for e in prosodic_events]
        avg_engagement = sum(arousals) / len(arousals) if arousals else 0.5

        return {
            "peak_engagement_topics": peak_topics,
            "disengagement_triggers": disengage_topics,
            "avg_meeting_engagement": avg_engagement,
        }

    def _extract_topic_preferences(self, claim_events: list[Event]) -> dict[str, Any]:
        """Count keyword frequency and compute depth (claims per unique topic)."""
        frequency: dict[str, int] = {}
        for event in claim_events:
            for kw in event.payload.get("topic_keywords", []):
                if kw:
                    frequency[kw] = frequency.get(kw, 0) + 1

        # Depth: how many claims reference each topic keyword
        depth: dict[str, float] = {kw: float(cnt) for kw, cnt in frequency.items()}

        return {"frequency": frequency, "depth": depth}

    def _extract_rapport_affinity(self, rapport_events: list[Event]) -> dict[str, Any]:
        """Identify speakers with consistently high rapport composite scores."""
        # Accumulate scores per speaker pair
        speaker_scores: dict[str, list[float]] = {}

        for event in rapport_events:
            pair_scores = event.payload.get("pair_scores", {})
            # Handle both list format (from RapportEngine) and dict format
            if isinstance(pair_scores, list):
                for ps in pair_scores:
                    if isinstance(ps, dict):
                        composite = ps.get("composite", 0.5)
                        pair_key = ps.get("pair", "")
                        parts = pair_key.split("_") if pair_key else []
                        # Also try speaker_a / speaker_b keys
                        if not parts or len(parts) < 2:
                            parts = [ps.get("speaker_a", ""), ps.get("speaker_b", "")]
                        for speaker in parts:
                            if speaker and speaker != "user":
                                speaker_scores.setdefault(speaker, []).append(float(composite))
            elif isinstance(pair_scores, dict):
                for pair_key, composite in pair_scores.items():
                    parts = pair_key.split("_")
                    if len(parts) >= 2:
                        for speaker in parts:
                            if speaker != "user":
                                speaker_scores.setdefault(speaker, []).append(float(composite))

        best_speakers = [
            speaker
            for speaker, scores in speaker_scores.items()
            if scores and (sum(scores) / len(scores)) >= _RAPPORT_THRESHOLD
        ]

        return {"best_speakers": best_speakers, "component_weights": {}}


def _topics_near_arousal(
    prosodic_events: list[Event],
    claim_events: list[Event],
    high: bool,
) -> list[str]:
    """Return topic keywords from claims that temporally follow a high/low arousal event.

    Uses a simple sequential scan: once a qualifying prosodic event is seen,
    all subsequent claim keywords until the next qualifying event are collected.
    """
    threshold = _HIGH_AROUSAL_THRESHOLD if high else _LOW_AROUSAL_THRESHOLD
    topics: list[str] = []

    # Interleave prosodic and claim events by their position in the original event list
    # Since we don't have a shared timestamp ordering, use a simpler positional heuristic:
    # track whether the last prosodic event seen qualifies
    qualifying_active = False

    # Build a combined ordering using object identity won't work; use a flag approach
    # Process: alternate between checking prosodic states and collecting claim topics
    p_idx = 0
    for claim in claim_events:
        # Advance prosodic pointer to consume any preceding prosodic events
        while p_idx < len(prosodic_events):
            ar = prosodic_events[p_idx].payload.get("arousal", 0.5)
            if high and ar >= threshold:
                qualifying_active = True
            elif not high and ar <= threshold:
                qualifying_active = True
            else:
                qualifying_active = False
            p_idx += 1
            break  # consume one prosodic per claim (simple pairing)

        if qualifying_active:
            for kw in claim.payload.get("topic_keywords", []):
                if kw and kw not in topics:
                    topics.append(kw)

    return topics
