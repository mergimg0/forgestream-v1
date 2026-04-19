"""Weight sensitivity analysis for the SOS evaluator.

Measures which evaluator weight has the most impact on E(pi) by
perturbing each weight independently and observing variance in the output.
High variance => high sensitivity => that weight is most influential.
"""

from __future__ import annotations

import random

import numpy as np

from forgestream.events.schema import Event
from .evaluator import Evaluator


class WeightSensitivityAnalyzer:
    """Measures each evaluator weight's impact on E(pi) via perturbation variance.

    For each weight key:
      1. Generate 20 perturbations of ONLY that weight (±0.1 Gaussian noise).
      2. Renormalize the full weight vector to sum to 1.0 after each perturbation.
      3. Compute E(pi) using the Evaluator with the perturbed weights.
      4. Report variance of E(pi) across the 20 perturbations.

    High variance indicates the evaluator output is sensitive to that weight,
    meaning it has the most impact on measured meeting quality.
    """

    def __init__(
        self,
        n_perturbations: int = 20,
        perturbation_scale: float = 0.1,
    ) -> None:
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale

    def analyze(
        self,
        events: list[Event],
        weights: dict[str, float],
    ) -> dict:
        """Measure each weight's impact on E(pi) by perturbation variance.

        Args:
            events: Meeting event log used to evaluate E(pi).
            weights: Current evaluator weight dict (keys must match Evaluator keys).

        Returns:
            Dict with:
              "sensitivities": ordered dict (highest variance first) mapping each
                weight key to {"variance", "mean", "range"}.
              "most_impactful": key of the weight with the highest variance.
        """
        if not weights:
            return {"sensitivities": {}, "most_impactful": None}

        sensitivities: dict[str, dict[str, float]] = {}

        for key in weights:
            e_values: list[float] = []
            for _ in range(self.n_perturbations):
                perturbed = weights.copy()
                perturbed[key] = max(
                    0.01,
                    perturbed[key] + random.gauss(0, self.perturbation_scale),
                )
                total = sum(perturbed.values())
                perturbed = {k: v / total for k, v in perturbed.items()}
                evaluator = Evaluator(weights=perturbed)
                e_values.append(evaluator.compute(events))

            sensitivities[key] = {
                "variance": float(np.var(e_values)),
                "mean": float(np.mean(e_values)),
                "range": float(max(e_values) - min(e_values)),
            }

        ranked = sorted(
            sensitivities.items(),
            key=lambda x: x[1]["variance"],
            reverse=True,
        )
        return {
            "sensitivities": dict(ranked),
            "most_impactful": ranked[0][0],
        }
