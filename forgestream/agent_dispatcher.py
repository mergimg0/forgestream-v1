"""AgentDispatcher -- spawns Claude Code agents from requirement events."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .agents.registry import AgentInfo, AgentRegistry, AgentStatus, AgentType
from .agents.spawner import SpawnPolicy
from .agents.templates.research import ResearchTemplate
from .agents.templates.scaffold import ScaffoldTemplate
from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .governor.trust_region import TrustRegion
from .orchestrator import Orchestrator


class AgentDispatcher:
    """Spawns Claude Code CLI agents in tmux when requirements are detected.

    Subscribes to the orchestrator EventBus. When a requirement event arrives,
    checks SpawnPolicy, builds a prompt, and launches claude -p in tmux.
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        orchestrator: Orchestrator,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.registry = AgentRegistry()
        trust_region_path = Path(config.data_dir) / "trust_region.json"
        self.trust_region = TrustRegion.load(trust_region_path)
        self._trust_region_path = trust_region_path
        self.spawn_policy = SpawnPolicy(
            registry=self.registry, trust_region=self.trust_region
        )
        self.research_template = ResearchTemplate()
        self.scaffold_template = ScaffoldTemplate()

        Path("/tmp/forgestream").mkdir(parents=True, exist_ok=True)

    def should_spawn(self, event: Event) -> bool:
        """Check if we should spawn an agent for this event."""
        if event.event_type != EventType.REQUIREMENT:
            return False

        decision = self.spawn_policy.can_spawn(AgentType.RESEARCH)
        return decision.allowed

    def build_prompt(
        self,
        agent_type: AgentType,
        description: str,
        context_claims: list[str],
    ) -> str:
        """Build a prompt for the agent using the appropriate template."""
        if agent_type == AgentType.RESEARCH:
            return self.research_template.build_prompt(
                query=description,
                context_claims=context_claims,
            )
        else:
            return self.scaffold_template.build_prompt(
                requirement=description,
                domain="",
                verified_findings=context_claims,
            )

    def spawn_agent(
        self,
        agent_type: AgentType,
        description: str,
        prompt: str,
    ) -> AgentInfo:
        """Spawn a Claude Code agent in a tmux session."""
        agent = self.registry.register(
            agent_type=agent_type,
            task_description=description,
        )

        prompt_file = f"/tmp/forgestream/prompt-{agent.id}.md"
        Path(prompt_file).write_text(prompt)

        agent.tmux_session = f"{agent_type.value}-{agent.id}"

        headless_env = "CLAUDE_PROJECT_DIR=/usr/local/etc/claude-headless"
        if agent_type == AgentType.RESEARCH:
            cmd = (
                f"{headless_env} claude -p \"$(cat {prompt_file})\" "
                f"--allowedTools 'WebSearch,WebFetch,Read,Grep,Glob' "
                f"2>&1 | tee /tmp/forgestream/agent-{agent.id}.out"
            )
        else:
            cmd = (
                f"{headless_env} claude -p \"$(cat {prompt_file})\" "
                f"2>&1 | tee /tmp/forgestream/agent-{agent.id}.out"
            )

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", agent.tmux_session, cmd],
            capture_output=True,
        )

        self.registry.update_status(agent.id, AgentStatus.RUNNING)
        self.spawn_policy.record_spawn()
        return agent

    async def on_event(self, event: Event) -> None:
        """EventBus handler -- check if we should spawn an agent."""
        if not self.should_spawn(event):
            return

        description = event.payload.get("description", "")
        recent_claims = [
            e.payload.get("text", "")
            for e in self.orchestrator._event_buffer[-10:]
            if e.event_type == EventType.CLAIM
        ]

        prompt = self.build_prompt(
            agent_type=AgentType.RESEARCH,
            description=description,
            context_claims=recent_claims,
        )

        self.spawn_agent(
            agent_type=AgentType.RESEARCH,
            description=description,
            prompt=prompt,
        )
