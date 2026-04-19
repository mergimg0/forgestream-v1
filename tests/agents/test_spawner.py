from forgestream.agents.registry import AgentRegistry, AgentType
from forgestream.agents.spawner import SpawnDecision, SpawnPolicy
from forgestream.governor.trust_region import TrustRegion


class TestSpawnPolicy:
    def test_can_spawn_when_under_limits(self):
        registry = AgentRegistry()
        trust = TrustRegion()
        policy = SpawnPolicy(registry=registry, trust_region=trust)

        decision = policy.can_spawn(AgentType.RESEARCH)
        assert decision.allowed is True

    def test_cannot_spawn_over_limit(self):
        registry = AgentRegistry()
        trust = TrustRegion()
        policy = SpawnPolicy(registry=registry, trust_region=trust)

        # Initial limit is 3 (epsilon=0.525 at startup)
        registry.register(AgentType.RESEARCH, "r1")
        registry.register(AgentType.RESEARCH, "r2")
        registry.register(AgentType.RESEARCH, "r3")

        decision = policy.can_spawn(AgentType.RESEARCH)
        assert decision.allowed is False
        assert "limit" in decision.reason.lower()

    def test_spawn_limit_increases_with_trust(self):
        registry = AgentRegistry()
        trust = TrustRegion()

        for _ in range(10):
            trust.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        policy = SpawnPolicy(registry=registry, trust_region=trust)
        limits = trust.get_resource_limits()

        assert limits["max_concurrent_research"] > 2
