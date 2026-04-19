"""SOS axiom checking -- runtime invariants for convergence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AxiomResult:
    axiom: str
    holds: bool
    reason: str = ""


class AxiomChecker:
    """Checks the three SOS axioms at runtime.

    Axiom 1 (Monotone Improvement): E(window_n+1) >= E(window_n)
    Axiom 2 (Bounded Step): step size within trust region epsilon
    Axiom 3 (Constraint Preservation): invariants maintained
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        consecutive_violations_threshold: int = 3,
    ) -> None:
        self.epsilon = epsilon
        self.consecutive_violations_threshold = consecutive_violations_threshold

    def check_monotone(
        self, trajectory: list[float], window_size: int = 3
    ) -> AxiomResult:
        """Check Axiom 1: moving average of E is non-decreasing.

        Allows individual dips but flags N consecutive declining windows.
        """
        if len(trajectory) < window_size + 1:
            return AxiomResult(axiom="monotone", holds=True, reason="insufficient data")

        windows = []
        for i in range(len(trajectory) - window_size + 1):
            window_avg = sum(trajectory[i : i + window_size]) / window_size
            windows.append(window_avg)

        consecutive_declines = 0
        for i in range(1, len(windows)):
            if windows[i] < windows[i - 1]:
                consecutive_declines += 1
                if consecutive_declines >= self.consecutive_violations_threshold:
                    return AxiomResult(
                        axiom="monotone",
                        holds=False,
                        reason=(
                            f"{consecutive_declines} consecutive declining windows "
                            f"detected (threshold: {self.consecutive_violations_threshold})"
                        ),
                    )
            else:
                consecutive_declines = 0

        return AxiomResult(axiom="monotone", holds=True)

    def check_bounded_step(
        self,
        semantic_drift: float,
        resource_delta: int,
        scope_delta: int,
    ) -> AxiomResult:
        """Check Axiom 2: all step components within trust region."""
        violations = []

        if semantic_drift > self.epsilon:
            violations.append(
                f"semantic drift {semantic_drift:.2f} > epsilon {self.epsilon:.2f}"
            )
        if resource_delta > self.epsilon * 10:
            violations.append(
                f"resource delta {resource_delta} > bound {self.epsilon * 10:.0f}"
            )
        if scope_delta > self.epsilon * 50:
            violations.append(
                f"scope delta {scope_delta} > bound {self.epsilon * 50:.0f}"
            )

        if violations:
            return AxiomResult(
                axiom="bounded_step",
                holds=False,
                reason="; ".join(violations),
            )
        return AxiomResult(axiom="bounded_step", holds=True)

    def check_constraint(
        self,
        verified_claims_intact: bool,
        compilation_preserved: bool,
        source_chain_valid: bool,
    ) -> AxiomResult:
        """Check Axiom 3: all constraints preserved."""
        violations = []

        if not verified_claims_intact:
            violations.append("verified claims modified or removed")
        if not compilation_preserved:
            violations.append("previously compiling artifact no longer compiles")
        if not source_chain_valid:
            violations.append("verified finding missing source chain")

        if violations:
            return AxiomResult(
                axiom="constraint",
                holds=False,
                reason="; ".join(violations),
            )
        return AxiomResult(axiom="constraint", holds=True)

    def check_rapport_trend(
        self, rapport_trajectory: list[float], disengaged: bool,
    ) -> AxiomResult:
        """Advisory check for rapport degradation.

        Not a formal axiom violation — just a warning when rapport is
        declining AND disengagement is detected.
        """
        if not disengaged or len(rapport_trajectory) < 3:
            return AxiomResult(axiom="rapport_advisory", holds=True)

        consecutive_declines = 0
        for i in range(1, len(rapport_trajectory)):
            if rapport_trajectory[i] < rapport_trajectory[i - 1]:
                consecutive_declines += 1
            else:
                consecutive_declines = 0

        if consecutive_declines >= 3:
            return AxiomResult(
                axiom="rapport_advisory",
                holds=False,
                reason=f"{consecutive_declines} consecutive declining rapport windows with active disengagement",
            )
        return AxiomResult(axiom="rapport_advisory", holds=True)
