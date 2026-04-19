"""Pluggable evaluator function for the SOS Governor.

Replace compute() to change what 'improvement' means.
The SOS convergence theorems hold for ANY evaluator
that satisfies the three axioms.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgestream.events.schema import Event, EventType


@dataclass
class EvaluatorMetrics:
    knowledge_density: float
    verification_rate: float
    scaffold_success: float
    suggestion_uptake: float
    emotional_engagement: float
    composite: float


class Evaluator:
    """Computes E(pi) over a window of events.

    Weights are configurable and will be tuned via GRPO
    in the self-improvement loop (SP-10).
    """

    DEFAULT_WEIGHTS = {
        "knowledge": 0.25,
        "verification": 0.25,
        "scaffold": 0.20,
        "uptake": 0.15,
        "engagement": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def compute(self, events: list[Event]) -> float:
        """Compute the composite evaluator score."""
        metrics = self.compute_metrics(events)
        return metrics.composite

    def compute_metrics(self, events: list[Event]) -> EvaluatorMetrics:
        """Compute individual metric components and the composite."""
        kd = self._knowledge_density(events)
        vr = self._verification_rate(events)
        ss = self._scaffold_success(events)
        su = self._suggestion_uptake(events)
        ee = self._emotional_engagement(events)

        composite = (
            self.weights.get("knowledge", 0.0) * kd
            + self.weights.get("verification", 0.0) * vr
            + self.weights.get("scaffold", 0.0) * ss
            + self.weights.get("uptake", 0.0) * su
            + self.weights.get("engagement", 0.0) * ee
        )
        composite = max(0.0, min(1.0, composite))

        return EvaluatorMetrics(
            knowledge_density=kd,
            verification_rate=vr,
            scaffold_success=ss,
            suggestion_uptake=su,
            emotional_engagement=ee,
            composite=composite,
        )

    @staticmethod
    def _knowledge_density(events: list[Event]) -> float:
        """Unique concepts extracted per claim, scaled by average confidence.

        The confidence multiplier means high-confidence extraction runs produce
        higher knowledge_density than low-confidence runs with the same keywords.
        This makes E(π) sensitive to extraction quality, not just quantity.
        """
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if not claims:
            return 0.0
        all_keywords: set[str] = set()
        for c in claims:
            all_keywords.update(c.payload.get("topic_keywords", []))
        avg_confidence = sum(c.payload.get("confidence", 0.5) for c in claims) / len(claims)
        raw_density = len(all_keywords) / max(len(claims), 1)
        return min(1.0, raw_density * avg_confidence)

    @staticmethod
    def _verification_rate(events: list[Event]) -> float:
        """Ratio of verified findings to total research events."""
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if not claims:
            return 0.0
        return min(1.0, len(findings) / max(len(claims), 1))

    @staticmethod
    def _scaffold_success(events: list[Event]) -> float:
        """Ratio of compiling artifacts to total artifacts."""
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        if not artifacts:
            return 0.0
        compiling = sum(
            1 for a in artifacts if a.payload.get("compiles", False)
        )
        return compiling / len(artifacts)

    @staticmethod
    def _suggestion_uptake(events: list[Event]) -> float:
        """Returns 0.0 — uptake tracking not yet implemented.

        Previously returned 0.5 which created a GRPO feedback loop
        (constant signal attracted weight without contributing information).
        Returns 0.0 so the uptake weight dimension has no pull until implemented.
        """
        return 0.0

    @staticmethod
    def _emotional_engagement(events: list[Event]) -> float:
        """Compute emotional engagement from PROSODIC_FEATURE and RAPPORT_SCORE events.

        Uses rapport composite when available (weighted 0.4), falls back to
        prosodic features (arousal + F0 variability + energy).

        Returns 0.5 if no relevant events exist.
        """
        rapport_events = [
            e for e in events
            if e.event_type == EventType.RAPPORT_SCORE
        ]
        prosodic = [
            e for e in events
            if e.event_type == EventType.PROSODIC_FEATURE
        ]

        if not prosodic and not rapport_events:
            return 0.5

        # Rapport composite (if available)
        rapport_composite = None
        if rapport_events:
            rapport_composite = rapport_events[-1].payload.get("group_composite", 0.5)

        # Prosodic fallback components
        if prosodic:
            arousals = [e.payload.get("arousal", 0.5) for e in prosodic]
            f0_stds = [e.payload.get("f0_std", 0.0) for e in prosodic]
            energies = [e.payload.get("energy_rms", 0.0) for e in prosodic]
            mean_arousal = sum(arousals) / len(arousals)
            mean_f0_var = min(1.0, (sum(f0_stds) / len(f0_stds)) / 80.0)
            mean_energy = min(1.0, (sum(energies) / len(energies)) / 0.15)
        else:
            mean_arousal = 0.5
            mean_f0_var = 0.0
            mean_energy = 0.0

        if rapport_composite is not None:
            engagement = (
                0.4 * rapport_composite
                + 0.3 * mean_arousal
                + 0.3 * mean_f0_var
            )
        else:
            engagement = (
                0.4 * mean_arousal
                + 0.3 * mean_f0_var
                + 0.3 * mean_energy
            )

        return max(0.0, min(1.0, engagement))
