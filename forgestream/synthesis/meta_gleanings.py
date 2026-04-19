"""Meta-gleanings: cross-meeting pattern analysis over event logs.

Identifies structural patterns in how meetings produce knowledge:
- Which topic keywords co-occur with REQUIREMENT events
- Whether claim density follows explore→converge pattern
- Whether high-engagement topics lead to more artifacts
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

from forgestream.events.schema import Event, EventType


@dataclass
class MetaGleaning:
    """A detected structural pattern in the meeting event log.

    Attributes:
        gleaning_type: Category — "topic_outcome", "discussion_evolution",
                       or "engagement_outcome".
        description: Human-readable description of the pattern.
        confidence: Confidence score 0–1.
        supporting_events: Number of events that support this gleaning.
        metadata: Additional structured data for the pattern.
    """

    gleaning_type: str
    description: str
    confidence: float
    supporting_events: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


class MetaGleaningEngine:
    """Analyzes a meeting event log to extract meta-gleanings.

    Three analysis passes:
    1. _topic_outcome_mapping: Which keywords co-occur with REQUIREMENTs.
    2. _discussion_evolution: Claim density over time (explore→converge).
    3. _engagement_outcome: High-engagement topics → more artifacts.
    """

    # Minimum events to emit a gleaning (avoid noise on tiny logs)
    MIN_SUPPORT = 2

    def analyze(self, events: list[Event]) -> list[MetaGleaning]:
        """Run all three analysis passes and return all detected gleanings.

        Args:
            events: Full event log for the meeting.

        Returns:
            List of MetaGleaning objects (may be empty).
        """
        if not events:
            return []

        gleanings: list[MetaGleaning] = []
        gleanings.extend(self._topic_outcome_mapping(events))
        gleanings.extend(self._discussion_evolution(events))
        gleanings.extend(self._engagement_outcome(events))
        return gleanings

    def _topic_outcome_mapping(self, events: list[Event]) -> list[MetaGleaning]:
        """Detect keywords that co-occur with REQUIREMENT events vs plain CLAIMs.

        A keyword is "outcome-linked" if it appears in at least one REQUIREMENT
        event (or a CLAIM just before a REQUIREMENT) significantly more than in
        plain claims alone.
        """
        # Collect keywords from REQUIREMENT events
        req_keywords: Counter = Counter()
        for e in events:
            if e.event_type == EventType.REQUIREMENT:
                for kw in e.payload.get("topic_keywords", []):
                    req_keywords[kw] += 1

        if not req_keywords:
            return []

        # Also collect keywords from CLAIM events for comparison
        claim_keywords: Counter = Counter()
        for e in events:
            if e.event_type == EventType.CLAIM:
                for kw in e.payload.get("topic_keywords", []):
                    claim_keywords[kw] += 1

        # Keywords that appear in requirements but not as common in plain claims
        outcome_linked = [
            kw for kw, count in req_keywords.items()
            if count >= 1 and claim_keywords.get(kw, 0) <= count
        ]

        if len(outcome_linked) < 1:
            return []

        n_reqs = len([e for e in events if e.event_type == EventType.REQUIREMENT])
        confidence = min(1.0, len(outcome_linked) / max(len(req_keywords), 1))

        return [
            MetaGleaning(
                gleaning_type="topic_outcome",
                description=(
                    f"Keywords {outcome_linked[:5]} co-occur with requirement events, "
                    f"suggesting these topics drive requirements ({n_reqs} requirements)."
                ),
                confidence=confidence,
                supporting_events=n_reqs,
                metadata={
                    "outcome_linked_keywords": outcome_linked[:10],
                    "requirement_count": n_reqs,
                },
            )
        ]

    def _discussion_evolution(self, events: list[Event]) -> list[MetaGleaning]:
        """Detect explore→converge pattern in claim density over time.

        Split the timeline into two halves. If claim density in the first half
        exceeds claim density in the second half, the meeting converged.
        """
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if len(claims) < self.MIN_SUPPORT:
            return []

        # Sort claims by timestamp
        sorted_claims = sorted(claims, key=lambda e: e.timestamp)
        mid = len(sorted_claims) // 2
        early = sorted_claims[:mid]
        late = sorted_claims[mid:]

        if not early or not late:
            return []

        # Density = claims / time_span (in minutes); fall back to count if no spread
        def density(evts: list[Event]) -> float:
            if len(evts) < 2:
                return float(len(evts))
            span_minutes = (
                (evts[-1].timestamp - evts[0].timestamp).total_seconds() / 60.0
            )
            if span_minutes < 0.01:
                return float(len(evts))
            return len(evts) / span_minutes

        early_density = density(early)
        late_density = density(late)

        # Explore→converge: early dense, late sparse
        if early_density > late_density * 1.2:
            pattern = "explore→converge"
            description = (
                f"Claim density decreased from {early_density:.1f} to "
                f"{late_density:.1f} claims/min — classic explore-then-converge pattern."
            )
            confidence = min(
                1.0, (early_density - late_density) / max(early_density, 0.01)
            )
        elif late_density > early_density * 1.2:
            pattern = "converge→explore"
            description = (
                f"Claim density increased from {early_density:.1f} to "
                f"{late_density:.1f} claims/min — late-meeting exploration spike."
            )
            confidence = min(
                1.0, (late_density - early_density) / max(late_density, 0.01)
            )
        else:
            # Steady pattern — no gleaning
            return []

        return [
            MetaGleaning(
                gleaning_type="discussion_evolution",
                description=description,
                confidence=confidence,
                supporting_events=len(claims),
                metadata={
                    "pattern": pattern,
                    "early_density": early_density,
                    "late_density": late_density,
                    "total_claims": len(claims),
                },
            )
        ]

    def _engagement_outcome(self, events: list[Event]) -> list[MetaGleaning]:
        """Detect whether high-engagement topics (many claims) produce more artifacts.

        Groups claims by shared keywords and checks if high-claim-count keyword
        groups coincide with ARTIFACT events in the same session.
        """
        # Count claims per keyword
        keyword_claim_counts: Counter = Counter()
        for e in events:
            if e.event_type == EventType.CLAIM:
                for kw in e.payload.get("topic_keywords", []):
                    keyword_claim_counts[kw] += 1

        if not keyword_claim_counts:
            return []

        artifact_count = len([e for e in events if e.event_type == EventType.ARTIFACT])

        # Find "high-engagement" keywords (top half by claim count)
        sorted_kws = keyword_claim_counts.most_common()
        if len(sorted_kws) < 2:
            # Only one keyword — still check if artifacts exist
            top_kws = [kws for kws, _ in sorted_kws]
        else:
            split = max(1, len(sorted_kws) // 2)
            top_kws = [kw for kw, _ in sorted_kws[:split]]

        top_claim_count = sum(keyword_claim_counts[kw] for kw in top_kws)
        total_claims = sum(keyword_claim_counts.values())

        if total_claims < self.MIN_SUPPORT:
            return []

        high_engagement_ratio = top_claim_count / max(total_claims, 1)

        # Only emit if artifacts exist (or high engagement regardless)
        if artifact_count == 0 and high_engagement_ratio <= 0.5:
            return []

        confidence = min(1.0, high_engagement_ratio * 0.8 + 0.1 * min(1.0, artifact_count / 3))

        return [
            MetaGleaning(
                gleaning_type="engagement_outcome",
                description=(
                    f"High-engagement topics {top_kws[:3]} account for "
                    f"{high_engagement_ratio:.0%} of claims; "
                    f"{artifact_count} artifact(s) produced."
                ),
                confidence=confidence,
                supporting_events=top_claim_count + artifact_count,
                metadata={
                    "high_engagement_keywords": top_kws[:5],
                    "high_engagement_ratio": high_engagement_ratio,
                    "artifact_count": artifact_count,
                },
            )
        ]
