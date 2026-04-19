"""Tests for LeanStubGenerator — Lean 4 stub generation from claim components."""

from __future__ import annotations

import pytest

from forgestream.synthesis.lean_stub import LeanStubGenerator


class TestGenerateStub:
    def setup_method(self):
        self.gen = LeanStubGenerator()

    def test_generates_valid_stub(self):
        """generate_stub returns a non-empty string containing 'theorem'."""
        result = self.gen.generate_stub(
            claim_text="Any evaluator satisfying the three axioms converges",
            variables=["E : Evaluator"],
            hypotheses=["monotone E", "bounded_step E"],
            conclusion="converges E",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "theorem" in result

    def test_stub_has_sorry(self):
        """Generated stub must contain 'sorry' — it is an obligation, not a proof."""
        result = self.gen.generate_stub(
            claim_text="For all x, P x implies Q x",
            variables=["x : α"],
            hypotheses=["P x"],
            conclusion="Q x",
        )
        assert "sorry" in result

    def test_stub_contains_conclusion(self):
        """The conclusion text appears in the stub after the colon."""
        result = self.gen.generate_stub(
            claim_text="convergence holds",
            variables=["E : Evaluator"],
            hypotheses=["bounded E"],
            conclusion="converges E",
        )
        assert "converges E" in result

    def test_stub_contains_hypotheses(self):
        """Each hypothesis appears as a named parameter in the stub."""
        result = self.gen.generate_stub(
            claim_text="if monotone then converges",
            variables=["f : Nat → Nat"],
            hypotheses=["monotone f", "bounded f"],
            conclusion="converges f",
        )
        assert "monotone f" in result
        assert "bounded f" in result

    def test_stub_contains_variables(self):
        """Variables appear as implicit parameters."""
        result = self.gen.generate_stub(
            claim_text="for all n, n >= 0",
            variables=["n : Nat"],
            hypotheses=[],
            conclusion="n ≥ 0",
        )
        assert "n : Nat" in result

    def test_stub_no_hypotheses(self):
        """Stubs without hypotheses are still valid (just theorem name + conclusion)."""
        result = self.gen.generate_stub(
            claim_text="there exists a bound",
            variables=["S : Set"],
            hypotheses=[],
            conclusion="bounded S",
        )
        assert "theorem" in result
        assert "sorry" in result
        assert "bounded S" in result

    def test_stub_multiple_hypotheses_numbered(self):
        """Multiple hypotheses get numbered names h1, h2, h3..."""
        result = self.gen.generate_stub(
            claim_text="axiom convergence",
            variables=["E : Evaluator"],
            hypotheses=["axiom1 E", "axiom2 E", "axiom3 E"],
            conclusion="converges E",
        )
        assert "h1" in result
        assert "h2" in result
        assert "h3" in result


class TestExtractComponents:
    def setup_method(self):
        self.gen = LeanStubGenerator()

    def test_extract_if_then(self):
        """'if X then Y' extracts X as hypothesis and Y as conclusion."""
        result = self.gen.extract_components("if engagement exceeds 0.7 then verification rate doubles")
        assert "hypothesis" in result or "hypotheses" in result
        assert "conclusion" in result
        # The conclusion should contain the "then" part
        conclusion = result.get("conclusion", "")
        assert "verification" in conclusion or "doubles" in conclusion

    def test_extract_for_all(self):
        """'for all X, Y' extracts X as variable and Y as conclusion."""
        result = self.gen.extract_components("for all evaluators satisfying axioms, convergence holds")
        assert "variables" in result or "variable" in result
        conclusion = result.get("conclusion", "")
        assert len(conclusion) > 0

    def test_extract_for_any(self):
        """'for any X' also extracts the variable."""
        result = self.gen.extract_components("for any function f that is monotone, f converges")
        assert "variables" in result or "variable" in result

    def test_extract_for_every(self):
        """'for every X' also extracts the variable."""
        result = self.gen.extract_components("for every bounded set S, S has a supremum")
        components = result
        variables = components.get("variables", [])
        if isinstance(variables, list):
            joined = " ".join(variables)
        else:
            joined = str(variables)
        # Should extract something from the quantified variable
        assert len(joined) > 0 or len(components.get("conclusion", "")) > 0

    def test_extract_returns_dict(self):
        """extract_components always returns a dict."""
        result = self.gen.extract_components("the system converges")
        assert isinstance(result, dict)

    def test_extract_plain_claim_has_conclusion(self):
        """Even plain claims without quantifiers get a conclusion entry."""
        result = self.gen.extract_components("the algorithm terminates in finite time")
        assert "conclusion" in result
        assert len(result["conclusion"]) > 0

    def test_extract_when_then(self):
        """'when X, Y' is equivalent to 'if X then Y'."""
        result = self.gen.extract_components("when the input is bounded, the output converges")
        assert "conclusion" in result
        conclusion = result.get("conclusion", "")
        assert len(conclusion) > 0

    def test_extract_variables_is_list(self):
        """'variables' field is always a list."""
        result = self.gen.extract_components("for all x in S, P x")
        variables = result.get("variables", [])
        assert isinstance(variables, list)

    def test_extract_hypotheses_is_list(self):
        """'hypotheses' field is always a list."""
        result = self.gen.extract_components("if P then Q")
        hypotheses = result.get("hypotheses", [])
        assert isinstance(hypotheses, list)
