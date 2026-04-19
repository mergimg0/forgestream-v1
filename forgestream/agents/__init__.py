"""Agent orchestration -- Claude Code CLI lifecycle management."""
from .monitor import AgentMonitor, AgentOutput
from .registry import AgentInfo, AgentRegistry, AgentStatus, AgentType
from .spawner import SpawnDecision, SpawnPolicy
from .worktrees import WorktreeManager

__all__ = [
    "AgentInfo",
    "AgentMonitor",
    "AgentOutput",
    "AgentRegistry",
    "AgentStatus",
    "AgentType",
    "SpawnDecision",
    "SpawnPolicy",
    "WorktreeManager",
]
