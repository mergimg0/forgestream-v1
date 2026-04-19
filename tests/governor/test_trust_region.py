from forgestream.governor.trust_region import TrustRegion


class TestTrustRegion:
    def test_initial_epsilon(self):
        tr = TrustRegion()
        assert tr.epsilon == TrustRegion.EPSILON_BASE * TrustRegion._initial_competence() * 1.0

    def test_expand_on_good_meeting(self):
        tr = TrustRegion()
        initial = tr.epsilon
        tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        assert tr.epsilon > initial

    def test_contract_on_axiom_violation(self):
        tr = TrustRegion()
        tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        expanded = tr.epsilon
        tr.record_axiom_violation()
        assert tr.epsilon < expanded

    def test_epsilon_has_floor(self):
        tr = TrustRegion()
        for _ in range(100):
            tr.record_axiom_violation()
        assert tr.epsilon >= TrustRegion.EPSILON_FLOOR

    def test_epsilon_has_ceiling(self):
        tr = TrustRegion()
        for _ in range(100):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        assert tr.epsilon <= TrustRegion.EPSILON_CEILING

    def test_resource_limits_scale_with_epsilon(self):
        tr = TrustRegion()
        limits_conservative = tr.get_resource_limits()

        for _ in range(10):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        limits_earned = tr.get_resource_limits()
        assert limits_earned["max_concurrent_research"] >= limits_conservative["max_concurrent_research"]
        assert limits_earned["max_concurrent_scaffold"] >= limits_conservative["max_concurrent_scaffold"]

    def test_auto_spawn_only_when_earned(self):
        tr = TrustRegion()
        assert tr.get_resource_limits()["auto_spawn"] is False

        for _ in range(8):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        assert tr.get_resource_limits()["auto_spawn"] is True
