"""Agent registry -- tracks active Claude Code instances."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class AgentType(str, Enum):
    RESEARCH = "research"
    SCAFFOLD = "scaffold"


class AgentStatus(str, Enum):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentInfo:
    id: str
    agent_type: AgentType
    task_description: str
    status: AgentStatus = AgentStatus.PROVISIONING
    tmux_session: str = ""
    worktree_path: str = ""
    branch_name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class AgentRegistry:
    """In-memory registry of all agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    def register(
        self,
        agent_type: AgentType,
        task_description: str,
    ) -> AgentInfo:
        agent_id = f"{agent_type.value[:1]}-{str(uuid4())[:8]}"
        agent = AgentInfo(
            id=agent_id,
            agent_type=agent_type,
            task_description=task_description,
        )
        self._agents[agent_id] = agent
        return agent

    def get(self, agent_id: str) -> AgentInfo | None:
        return self._agents.get(agent_id)

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                agent.completed_at = datetime.now(timezone.utc)

    def get_active(self) -> list[AgentInfo]:
        return [
            a
            for a in self._agents.values()
            if a.status in (AgentStatus.PROVISIONING, AgentStatus.RUNNING)
        ]

    def get_by_type(self, agent_type: AgentType) -> list[AgentInfo]:
        return [
            a
            for a in self._agents.values()
            if a.agent_type == agent_type
            and a.status in (AgentStatus.PROVISIONING, AgentStatus.RUNNING)
        ]

    def count_active_by_type(self) -> Counter[AgentType]:
        active = self.get_active()
        return Counter(a.agent_type for a in active)
