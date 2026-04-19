"""Agent spawn policy and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .registry import AgentRegistry, AgentType
from forgestream.governor.trust_region import TrustRegion


@dataclass
class SpawnDecision:
    allowed: bool
    reason: str = ""


class SpawnPolicy:
    """Determines whether a new agent can be spawned based on trust region limits."""

    def __init__(
        self,
        registry: AgentRegistry,
        trust_region: TrustRegion,
    ) -> None:
        self.registry = registry
        self.trust_region = trust_region
        self._last_spawn: datetime | None = None

    def can_spawn(self, agent_type: AgentType) -> SpawnDecision:
        """Check if spawning a new agent is allowed."""
        limits = self.trust_region.get_resource_limits()

        counts = self.registry.count_active_by_type()
        limit_key = (
            "max_concurrent_research"
            if agent_type == AgentType.RESEARCH
            else "max_concurrent_scaffold"
        )
        current_count = counts.get(agent_type, 0)
        max_count = limits[limit_key]

        if current_count >= max_count:
            return SpawnDecision(
                allowed=False,
                reason=f"{agent_type.value} agent limit reached ({current_count}/{max_count})",
            )

        cooldown = limits["spawn_cooldown_seconds"]
        if self._last_spawn is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_spawn).total_seconds()
            if elapsed < cooldown:
                return SpawnDecision(
                    allowed=False,
                    reason=f"cooldown: {cooldown - elapsed:.0f}s remaining",
                )

        return SpawnDecision(allowed=True)

    def record_spawn(self) -> None:
        """Record that an agent was spawned (for cooldown tracking)."""
        self._last_spawn = datetime.now(timezone.utc)
