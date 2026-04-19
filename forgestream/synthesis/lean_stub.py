"""LeanStubGenerator — template-based Lean 4 stub generation from claim components.

This is deterministic and auditable — no LLM required. Stubs use `sorry` as a
placeholder; ProofForge's GRPO loop attempts to fill the proof.
"""

from __future__ import annotations

import re


class LeanStubGenerator:
    """Generate Lean 4 theorem stubs from natural language claim components."""

    # Patterns for extracting if/when clauses (hypothesis + conclusion)
    _IF_THEN_PATTERN = re.compile(
        r"(?:if|when|whenever)\s+(.+?)(?:,\s*|\s+then\s+)(.+)",
        re.IGNORECASE,
    )

    # Patterns for universal quantification
    _FOR_ALL_PATTERN = re.compile(
        r"(?:for all|for every|for any)\s+([^,]+)(?:,\s*)?(.+)?",
        re.IGNORECASE,
    )

    def generate_stub(
        self,
        claim_text: str,
        variables: list[str],
        hypotheses: list[str],
        conclusion: str,
    ) -> str:
        """Generate a Lean 4 theorem stub with sorry.

        Args:
            claim_text: Original natural language claim (used only for the name slug).
            variables: List of variable declarations, e.g. ["E : Evaluator"].
            hypotheses: List of hypothesis strings, e.g. ["monotone E", "bounded E"].
            conclusion: The conclusion string, e.g. "converges E".

        Returns:
            A Lean 4 theorem string ending with `sorry`.
        """
        lines = ["theorem auto_obligation"]

        # Variable declarations (implicit args)
        for var in variables:
            lines.append(f"  ({var})")

        # Hypothesis declarations (named h1, h2, ...)
        for i, hyp in enumerate(hypotheses):
            lines.append(f"  (h{i + 1} : {hyp})")

        # Conclusion + sorry
        lines.append(f"  : {conclusion} := by")
        lines.append("  sorry")

        return "\n".join(lines)

    def extract_components(self, claim_text: str) -> dict:
        """Heuristically extract variables, hypotheses, and conclusion from claim text.

        This is intentionally imperfect — the human review step catches extraction
        errors. The goal is to surface obligations, not to produce correct Lean code.

        Returns a dict with keys: variables (list), hypotheses (list), conclusion (str).
        """
        variables: list[str] = []
        hypotheses: list[str] = []
        conclusion: str = ""

        # Try if/when/whenever → hypothesis + conclusion
        m = self._IF_THEN_PATTERN.search(claim_text)
        if m:
            hyp_text = m.group(1).strip().rstrip(",")
            conclusion = m.group(2).strip() if m.group(2) else ""
            if hyp_text:
                hypotheses.append(hyp_text)
            return {
                "variables": variables,
                "hypotheses": hypotheses,
                "conclusion": conclusion,
            }

        # Try for all/every/any → variable + conclusion
        m = self._FOR_ALL_PATTERN.search(claim_text)
        if m:
            var_text = m.group(1).strip().rstrip(",")
            rest = m.group(2).strip() if m.group(2) else ""
            # Use a sanitized version of the quantified variable
            var_slug = var_text.split()[0] if var_text.split() else "x"
            variables.append(f"{var_slug} : _")
            conclusion = rest if rest else claim_text
            return {
                "variables": variables,
                "hypotheses": hypotheses,
                "conclusion": conclusion,
            }

        # Fallback: use the entire claim as conclusion
        conclusion = claim_text.strip()
        return {
            "variables": variables,
            "hypotheses": hypotheses,
            "conclusion": conclusion,
        }
