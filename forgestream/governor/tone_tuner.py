"""GRPO tuning of tone adjustment values.

Replaces the hardcoded HESITATION_PENALTY, BACKTRACK_PENALTY, EMPHASIS_BOOST,
EXCITEMENT_BOOST in extraction.py with data-driven values tuned via GRPO
after each meeting.
"""

from __future__ import annotations

import random

from forgestream.events.schema import Event, EventType


class ToneAdjustmentTuner:
    """GRPO-style tuning for tone adjustment parameters."""

    DEFAULT_ADJUSTMENTS = {
        "hesitation_penalty": 0.15,
        "backtrack_penalty": 0.20,
        "emphasis_boost": 0.20,
        "excitement_boost": 0.15,
    }

    def __init__(self, perturbation_scale: float = 0.03) -> None:
        self.perturbation_scale = perturbation_scale

    def generate_perturbations(
        self, current: dict[str, float], n: int = 10
    ) -> list[dict[str, float]]:
        """Generate N perturbed adjustment sets, all values >= 0."""
        perturbations = []
        for _ in range(n):
            perturbed = {
                k: max(0.0, v + random.gauss(0, self.perturbation_scale))
                for k, v in current.items()
            }
            perturbations.append(perturbed)
        return perturbations

    def tune(
        self,
        current: dict[str, float],
        events: list[Event],
        human_score: float = 0.5,
    ) -> dict[str, float]:
        """Tune tone adjustments using GRPO.

        For each perturbation, simulate applying tone adjustments to claims
        that have matching prosodic features. Score by correlation with
        human feedback. Blend best with current (70/30 conservative).
        """
        perturbations = self.generate_perturbations(current, n=10)

        claims = [e for e in events if e.event_type == EventType.CLAIM]
        prosodic = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]

        if not claims or not prosodic:
            return current

        # Score each perturbation
        scores: list[tuple[float, dict[str, float]]] = []
        for p_adj in perturbations:
            sim_score = self._simulate_adjustments(p_adj, claims, prosodic)
            correlation = 1.0 - abs(sim_score - human_score)
            scores.append((correlation, p_adj))

        scores.sort(key=lambda x: x[0], reverse=True)
        best = scores[0][1]

        # Blend: 70% current + 30% best
        blended = {
            k: max(0.0, 0.7 * current[k] + 0.3 * best[k])
            for k in current
        }
        return blended

    @staticmethod
    def _simulate_adjustments(
        adjustments: dict[str, float],
        claims: list[Event],
        prosodic: list[Event],
    ) -> float:
        """Simulate applying tone adjustments and compute a quality score.

        Higher score = adjustments that correctly penalize stressed claims
        and boost confident ones.
        """
        if not claims:
            return 0.5

        total_confidence = 0.0
        for claim in claims:
            conf = claim.payload.get("confidence", 0.5)
            markers = claim.payload.get("tone_markers", [])

            if "hesitation" in markers:
                conf -= adjustments.get("hesitation_penalty", 0.15)
            if "backtracking" in markers:
                conf -= adjustments.get("backtrack_penalty", 0.20)
            if "emphasis" in markers:
                conf += adjustments.get("emphasis_boost", 0.20)
            if "excitement" in markers:
                conf += adjustments.get("excitement_boost", 0.15)

            total_confidence += max(0.0, min(1.0, conf))

        return total_confidence / len(claims)
