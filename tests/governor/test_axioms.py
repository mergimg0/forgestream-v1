from forgestream.governor.axioms import AxiomChecker, AxiomResult


class TestAxiomChecker:
    def test_monotone_improvement_holds(self):
        checker = AxiomChecker()
        trajectory = [0.3, 0.35, 0.4, 0.45, 0.5]
        result = checker.check_monotone(trajectory, window_size=3)
        assert result.holds is True

    def test_monotone_improvement_violated(self):
        checker = AxiomChecker()
        trajectory = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
        result = checker.check_monotone(trajectory, window_size=2)
        assert result.holds is False

    def test_monotone_allows_individual_dips(self):
        checker = AxiomChecker()
        trajectory = [0.3, 0.4, 0.35, 0.45, 0.5]
        result = checker.check_monotone(trajectory, window_size=3)
        assert result.holds is True

    def test_monotone_insufficient_data(self):
        checker = AxiomChecker()
        result = checker.check_monotone([0.5], window_size=3)
        assert result.holds is True

    def test_bounded_step_holds(self):
        checker = AxiomChecker(epsilon=0.5)
        result = checker.check_bounded_step(
            semantic_drift=0.3, resource_delta=1, scope_delta=5
        )
        assert result.holds is True

    def test_bounded_step_violated_drift(self):
        checker = AxiomChecker(epsilon=0.3)
        result = checker.check_bounded_step(
            semantic_drift=0.8, resource_delta=0, scope_delta=0
        )
        assert result.holds is False
        assert "semantic" in result.reason.lower()

    def test_constraint_preservation_holds(self):
        checker = AxiomChecker()
        result = checker.check_constraint(
            verified_claims_intact=True,
            compilation_preserved=True,
            source_chain_valid=True,
        )
        assert result.holds is True

    def test_constraint_preservation_violated(self):
        checker = AxiomChecker()
        result = checker.check_constraint(
            verified_claims_intact=False,
            compilation_preserved=True,
            source_chain_valid=True,
        )
        assert result.holds is False
        assert "verified" in result.reason.lower()
