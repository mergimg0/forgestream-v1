"""GRPO-style prompt parameter tuning for Gemini extraction instructions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from forgestream.events.schema import Event


@dataclass
class PromptParams:
    """Tunable parameters that modulate the Gemini extraction instruction.

    Attributes:
        extraction_granularity: 0.0 = coarse (summaries only),
                                1.0 = fine (every atomic claim).
        tone_sensitivity: 0.0 = ignore tone markers,
                          1.0 = capture all tone/emotion signals.
        context_injection_minutes: How many minutes of prior context
                                   to prepend to the instruction (0 = none).
    """

    extraction_granularity: float = 0.5
    tone_sensitivity: float = 0.5
    context_injection_minutes: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromptParams:
        """Deserialize from a plain dict."""
        return cls(
            extraction_granularity=float(d.get("extraction_granularity", 0.5)),
            tone_sensitivity=float(d.get("tone_sensitivity", 0.5)),
            context_injection_minutes=float(d.get("context_injection_minutes", 3.0)),
        )

    def save(self, path: str) -> None:
        """Persist params to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> PromptParams:
        """Load params from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


class PromptTuner:
    """GRPO-style tuner for PromptParams.

    Generates N perturbed PromptParams variants, scores each against
    human feedback (or a proxy), and blends the best with the current.
    """

    def __init__(
        self,
        perturbation_scale: float = 0.1,
        n_perturbations: int = 10,
        blend_keep: float = 0.7,
    ) -> None:
        self.perturbation_scale = perturbation_scale
        self.n_perturbations = n_perturbations
        self.blend_keep = blend_keep

    def apply_params(self, base_instruction: str, params: PromptParams) -> str:
        """Modify base_instruction according to the current PromptParams.

        Adds modifiers based on extraction_granularity and tone_sensitivity.
        Optionally prepends a context injection note.

        Args:
            base_instruction: The template instruction string.
            params: PromptParams to apply.

        Returns:
            Modified instruction string.
        """
        parts = [base_instruction]

        # Extraction granularity modifier
        if params.extraction_granularity >= 0.7:
            parts.append(
                "Extract EVERY atomic claim — even brief or implicit ones. "
                "Maximum granularity."
            )
        elif params.extraction_granularity <= 0.3:
            parts.append(
                "Extract only the most substantive claims. "
                "Skip minor or transitional statements."
            )
        else:
            parts.append("Extract substantive claims with normal granularity.")

        # Tone sensitivity modifier
        if params.tone_sensitivity >= 0.7:
            parts.append(
                "Capture all tone markers: uncertainty, emphasis, hedging, "
                "emotional signals."
            )
        elif params.tone_sensitivity <= 0.3:
            parts.append("Ignore tone markers; focus on factual content only.")

        # Context injection
        if params.context_injection_minutes > 0:
            mins = int(params.context_injection_minutes)
            parts.append(
                f"Use the last {mins} minute(s) of conversation as context "
                "to improve claim relevance."
            )

        return " ".join(parts)

    def _perturb(self, params: PromptParams) -> PromptParams:
        """Generate a single perturbed PromptParams."""
        scale = self.perturbation_scale
        return PromptParams(
            extraction_granularity=max(
                0.0,
                min(1.0, params.extraction_granularity + random.gauss(0, scale)),
            ),
            tone_sensitivity=max(
                0.0,
                min(1.0, params.tone_sensitivity + random.gauss(0, scale)),
            ),
            context_injection_minutes=max(
                0.0,
                params.context_injection_minutes + random.gauss(0, scale * 5),
            ),
        )

    def _score(self, params: PromptParams, events: list[Event], human_score: float) -> float:
        """Score a PromptParams variant.

        Simple heuristic: extraction_granularity should correlate with
        human satisfaction (more granular = better when human_score is high).
        Combine with human_score proximity.
        """
        # Proxy: correlate granularity with human score
        granularity_alignment = 1.0 - abs(params.extraction_granularity - human_score)
        tone_neutral = 1.0 - abs(params.tone_sensitivity - 0.5)
        return 0.7 * granularity_alignment + 0.3 * tone_neutral

    def tune(
        self,
        current: PromptParams,
        events: list[Event],
        human_score: float = 0.5,
    ) -> PromptParams:
        """GRPO-perturb PromptParams toward better extraction quality.

        Generates N perturbations, scores each, blends best with current.

        Args:
            current: Current PromptParams.
            events: Meeting event log.
            human_score: Post-meeting human feedback (0–1).

        Returns:
            Updated PromptParams.
        """
        candidates = [self._perturb(current) for _ in range(self.n_perturbations)]

        scored = [(self._score(c, events, human_score), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        keep = self.blend_keep
        return PromptParams(
            extraction_granularity=keep * current.extraction_granularity
            + (1 - keep) * best.extraction_granularity,
            tone_sensitivity=keep * current.tone_sensitivity
            + (1 - keep) * best.tone_sensitivity,
            context_injection_minutes=keep * current.context_injection_minutes
            + (1 - keep) * best.context_injection_minutes,
        )
