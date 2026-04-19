"""Trust region dynamics -- the system earns autonomy through convergence."""

from __future__ import annotations

import json
import math
from pathlib import Path


class TrustRegion:
    """Manages the trust region epsilon that governs agent autonomy.

    epsilon starts conservative (0.3) and expands as the system
    demonstrates competence across meetings. Axiom violations contract it.
    """

    EPSILON_BASE = 0.3
    EPSILON_FLOOR = 0.15
    EPSILON_CEILING = 0.9
    AUTO_SPAWN_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._consecutive_improvements: int = 0
        self._total_violations: int = 0
        self._meeting_count: int = 0
        self._volatility: float = 0.0

    @property
    def epsilon(self) -> float:
        """Current trust region value."""
        competence = self._competence_multiplier()
        stability = self._stability_factor()
        raw = self.EPSILON_BASE * competence * stability
        return max(self.EPSILON_FLOOR, min(self.EPSILON_CEILING, raw))

    def record_meeting_result(
        self, e_macro_improved: bool, axiom_violations: int,
        rapport_trend: float = 0.0,
    ) -> None:
        """Record the outcome of a meeting."""
        self._meeting_count += 1
        self._total_violations += axiom_violations

        if e_macro_improved:
            self._consecutive_improvements += 1
            # Rapport trend boost: building rapport + improving E = stronger evidence
            if rapport_trend > 0.1:
                self._consecutive_improvements += 0.5
        else:
            self._consecutive_improvements = max(
                0, self._consecutive_improvements - 1
            )

    def record_axiom_violation(self) -> None:
        """Record a single axiom violation (contracts trust region)."""
        self._total_violations += 1
        self._consecutive_improvements = max(
            0, self._consecutive_improvements - 2
        )

    def set_volatility(self, volatility: float) -> None:
        """Update the micro-evaluator volatility."""
        self._volatility = max(0.0, min(1.0, volatility))

    def get_resource_limits(self) -> dict:
        """Resource limits scaled by current epsilon."""
        e = self.epsilon
        scale = e / self.EPSILON_CEILING  # 0 to 1

        return {
            "max_concurrent_research": max(2, int(2 + 3 * scale)),
            "max_concurrent_scaffold": max(2, int(2 + 4 * scale)),
            "spawn_cooldown_seconds": max(10, int(60 - 50 * scale)),
            "scaffold_timeout_minutes": max(10, int(10 + 10 * scale)),
            "auto_spawn": e >= self.AUTO_SPAWN_THRESHOLD,
            "branch_auto_allocate": e >= 0.7,
        }

    def _competence_multiplier(self) -> float:
        """Sigmoid based on improvements vs violations. Range [0.5, 3.0]."""
        alpha = 0.3
        beta = 0.5
        x = alpha * self._consecutive_improvements - beta * self._total_violations
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        return 0.5 + 2.5 * sigmoid

    def _stability_factor(self) -> float:
        """Low volatility = stable = higher factor. Range [0.3, 1.0]."""
        return max(0.3, 1.0 - self._volatility)

    @staticmethod
    def _initial_competence() -> float:
        """Competence multiplier at initialization (0 improvements, 0 violations)."""
        sigmoid = 1.0 / (1.0 + math.exp(0))  # x=0 -> sigmoid=0.5
        return 0.5 + 2.5 * sigmoid  # = 1.75

    def save(self, path: str | Path, history_path: str | Path | None = None) -> None:
        """Save trust region state to JSON file.

        Also appends the current meeting snapshot to the epsilon history file
        so the autonomy-progression panel can track ε across meetings.

        Args:
            path: Primary state file (state is always overwritten).
            history_path: History array file.  Defaults to
                ``<path.parent>/trust_region_history.json``.
        """
        data = {
            "consecutive_improvements": self._consecutive_improvements,
            "total_violations": self._total_violations,
            "meeting_count": self._meeting_count,
            "volatility": self._volatility,
            "epsilon": self.epsilon,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

        # Append to history file
        if history_path is None:
            history_path = p.parent / "trust_region_history.json"
        history_path = Path(history_path)
        self._append_history(history_path)

    def _append_history(self, history_path: Path) -> None:
        """Append the current meeting snapshot to the history JSON array."""
        if history_path.exists():
            try:
                history: list[dict] = json.loads(history_path.read_text())
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError):
                history = []
        else:
            history = []

        # Determine meeting number (1-based, or one beyond last recorded)
        if history:
            last_meeting = max(e.get("meeting", 0) for e in history)
            meeting_num = last_meeting + 1
        else:
            meeting_num = max(1, self._meeting_count)

        history.append({
            "meeting": meeting_num,
            "epsilon": round(self.epsilon, 6),
            "improvements": int(self._consecutive_improvements),
            "violations": self._total_violations,
        })

        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrustRegion":
        """Load trust region state from JSON file."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        tr = cls()
        tr._consecutive_improvements = data.get("consecutive_improvements", 0)
        tr._total_violations = data.get("total_violations", 0)
        tr._meeting_count = data.get("meeting_count", 0)
        tr._volatility = data.get("volatility", 0.0)
        return tr
