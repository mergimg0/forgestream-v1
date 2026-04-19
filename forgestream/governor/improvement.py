"""Self-improvement mechanisms -- GRPO weight tuning, prompt evolution, synthesis."""

from __future__ import annotations

import random
from typing import Any

from forgestream.events.schema import Event, EventType
from .evaluator import Evaluator


class WeightTuner:
    """GRPO-style evaluator weight tuning.

    After each meeting:
    1. Generate N perturbed weight vectors
    2. Retroactively compute E_meso for each
    3. Update toward best-performing perturbation
    """

    def __init__(self, perturbation_scale: float = 0.05) -> None:
        self.perturbation_scale = perturbation_scale

    def generate_perturbations(
        self, weights: dict[str, float], n: int = 10
    ) -> list[dict[str, float]]:
        """Generate N perturbed weight vectors, each normalized to sum to 1.0."""
        perturbations = []
        keys = list(weights.keys())

        for _ in range(n):
            perturbed = {
                k: max(0.01, weights[k] + random.gauss(0, self.perturbation_scale))
                for k in keys
            }
            total = sum(perturbed.values())
            normalized = {k: v / total for k, v in perturbed.items()}
            perturbations.append(normalized)

        return perturbations

    def tune_multi_objective(
        self,
        current_weights: dict[str, float],
        events: list[Event],
        component_targets: dict[str, float],
    ) -> dict[str, float]:
        """Tune weights toward per-component targets using multi-objective GRPO.

        Each perturbation is scored by:
            sum over keys: (1 - |component_actual - component_target|)

        where component_actual is the weight value in the perturbation itself
        (as a proxy for where emphasis is placed), and component_target is
        the desired weight for that key.

        The best perturbation is blended 70/30 with the current weights.

        Args:
            current_weights: Current weight dict (any keys).
            events: Meeting event log (used for consistency with tune()).
            component_targets: Desired target for each weight key.

        Returns:
            Normalized weight dict summing to 1.0.
        """
        perturbations = self.generate_perturbations(current_weights, n=10)

        scores = []
        for p_weights in perturbations:
            score = sum(
                1.0 - abs(p_weights.get(k, 0.0) - component_targets.get(k, 0.0))
                for k in current_weights
            )
            scores.append((score, p_weights))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_weights = scores[0][1]

        blended = {
            k: 0.7 * current_weights[k] + 0.3 * best_weights[k]
            for k in current_weights
        }
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

    def tune(
        self,
        current_weights: dict[str, float],
        events: list[Event],
        human_score: float = 0.5,
    ) -> dict[str, float]:
        """Tune weights using GRPO: generate perturbations, evaluate, update.

        human_score: post-meeting human feedback (0-1), used as target.
        """
        perturbations = self.generate_perturbations(current_weights, n=10)

        # Evaluate each perturbation against the event log
        scores = []
        for p_weights in perturbations:
            evaluator = Evaluator(weights=p_weights)
            e_meso = evaluator.compute(events)
            # Score = correlation with human feedback
            score = 1.0 - abs(e_meso - human_score)
            scores.append((score, p_weights))

        # Sort by score descending, take the best
        scores.sort(key=lambda x: x[0], reverse=True)
        best_weights = scores[0][1]

        # Blend: 70% current + 30% best perturbation (conservative update)
        blended = {
            k: 0.7 * current_weights[k] + 0.3 * best_weights[k]
            for k in current_weights
        }
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}


class PromptEvolution:
    """Tracks and scores agent prompt variants across meetings."""

    HUMAN_ACTION_SCORES = {
        "used": 1.0,
        "modified": 0.6,
        "discarded": 0.1,
    }

    def score_prompt(
        self,
        prompt: str,
        output_useful: bool,
        human_action: str,
    ) -> float:
        """Score a prompt variant based on its output quality."""
        action_score = self.HUMAN_ACTION_SCORES.get(human_action, 0.3)
        useful_score = 1.0 if output_useful else 0.2
        return action_score * 0.6 + useful_score * 0.4


class MeetingSynthesizer:
    """Generates post-meeting synthesis reports."""

    def generate_summary(self, events: list[Event]) -> str:
        """Generate a meeting summary from the event log."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]
        contradictions = [e for e in events if e.event_type == EventType.CONTRADICTION]

        lines = [
            "# Meeting Summary\n",
            f"## Knowledge Extracted\n- {len(claims)} claims captured",
            f"- {len(findings)} verified findings",
            f"- {len(contradictions)} contradictions detected\n",
            f"## Requirements\n- {len(requirements)} requirements identified",
        ]

        for req in requirements[:10]:
            lines.append(f"  - {req.payload.get('description', 'N/A')}")

        lines.append(f"\n## Artifacts\n- {len(artifacts)} scaffolds produced")
        for art in artifacts[:10]:
            compiles = art.payload.get("compiles", False)
            tests = art.payload.get("tests_pass", False)
            status = "pass" if compiles and tests else "partial" if compiles else "fail"
            files = art.payload.get("files_created", [])
            lines.append(f"  - [{status}] {len(files)} files")

        return "\n".join(lines)
