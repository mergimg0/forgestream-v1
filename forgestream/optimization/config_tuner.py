"""ConfigTuner -- GRPO-style system configuration parameter tuning."""

from __future__ import annotations

import random

from .analyzer import AnalysisResult


class ConfigTuner:
    """GRPO-tunes system configuration parameters after each meeting.

    Uses the same perturbation-selection-blend algorithm as WeightTuner,
    applied to a distinct parameter space of system config values.

    Persistence is handled externally (caller saves to data/config_overrides.json).
    """

    TUNABLE_PARAMS: dict[str, tuple[float, float]] = {
        "spawn_cooldown_seconds": (10.0, 120.0),
        "scaffold_timeout_minutes": (5.0, 30.0),
        "max_concurrent_research": (2.0, 8.0),
        "max_concurrent_scaffold": (2.0, 8.0),
        "emotion_stride_seconds": (0.5, 3.0),
        "emotion_window_seconds": (1.0, 5.0),
    }

    def __init__(self, perturbation_scale: float = 0.05, n_perturbations: int = 10) -> None:
        self.perturbation_scale = perturbation_scale
        self.n_perturbations = n_perturbations

    def generate_perturbations(
        self, params: dict[str, float]
    ) -> list[dict[str, float]]:
        """Generate N perturbed parameter dicts, each clipped to valid ranges."""
        result = []
        for _ in range(self.n_perturbations):
            perturbed: dict[str, float] = {}
            for key, (lo, hi) in self.TUNABLE_PARAMS.items():
                base = params.get(key, (lo + hi) / 2.0)
                # Scale perturbation relative to range size
                scale = (hi - lo) * self.perturbation_scale
                new_val = base + random.gauss(0, scale)
                perturbed[key] = max(lo, min(hi, new_val))
            result.append(perturbed)
        return result

    def _score_perturbation(
        self,
        params: dict[str, float],
        analysis: AnalysisResult,
        human_score: float,
    ) -> float:
        """Retrospective quality score for a parameter configuration.

        Higher score = better:
        - Fewer bottlenecks → shorter spawn_cooldown is fine
        - Fewer timeouts → shorter scaffold_timeout is acceptable (rewarded)
        - Better emotion correlation → better emotion window params
        - human_score provides overall quality signal
        """
        score = human_score  # Base quality signal

        # If many bottlenecks, prefer shorter spawn_cooldown (more responsive)
        if analysis.bottleneck_count > 2:
            lo, hi = self.TUNABLE_PARAMS["spawn_cooldown_seconds"]
            normalized = (params["spawn_cooldown_seconds"] - lo) / (hi - lo)
            score += 0.1 * (1.0 - normalized)  # reward lower cooldown

        # If agent timeouts occurred, prefer longer scaffold_timeout
        if analysis.agent_timeout_count > 0:
            lo, hi = self.TUNABLE_PARAMS["scaffold_timeout_minutes"]
            normalized = (params["scaffold_timeout_minutes"] - lo) / (hi - lo)
            score += 0.1 * normalized  # reward higher timeout

        # If emotion correlation is strong, reward current emotion params
        if analysis.emotion_claim_correlation is not None:
            score += 0.05 * abs(analysis.emotion_claim_correlation)

        return score

    def tune(
        self,
        current_params: dict[str, float],
        analysis: AnalysisResult,
        human_score: float = 0.5,
    ) -> dict[str, float]:
        """Tune system config params using GRPO: perturb → score → blend.

        Only keys in TUNABLE_PARAMS are tuned; unknown keys in current_params
        are ignored (backward compat).

        Returns a new dict with only TUNABLE_PARAMS keys, values clipped to range.
        """
        # Extract only the tunable subset from current
        base: dict[str, float] = {}
        for key, (lo, hi) in self.TUNABLE_PARAMS.items():
            base[key] = max(lo, min(hi, current_params.get(key, (lo + hi) / 2.0)))

        perturbations = self.generate_perturbations(base)

        # Score each perturbation
        scored = [
            (self._score_perturbation(p, analysis, human_score), p)
            for p in perturbations
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        # Conservative blend: 70% current + 30% best perturbation
        blended: dict[str, float] = {}
        for key, (lo, hi) in self.TUNABLE_PARAMS.items():
            blended[key] = max(lo, min(hi, 0.7 * base[key] + 0.3 * best[key]))

        return blended
