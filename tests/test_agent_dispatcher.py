"""AgentDispatcher tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from forgestream.agent_dispatcher import AgentDispatcher
from forgestream.agents.registry import AgentStatus, AgentType
from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator


class TestAgentDispatcher:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)
        assert dispatcher.registry is not None
        assert dispatcher.spawn_policy is not None

    async def test_handles_requirement_event(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.5,
            payload={
                "description": "Build a data pipeline",
                "domain": "data-engineering",
                "complexity_estimate": 0.5,
                "linked_claims": [],
            },
        )

        result = dispatcher.should_spawn(event)
        assert isinstance(result, bool)

    def test_build_research_prompt(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        prompt = dispatcher.build_prompt(
            agent_type=AgentType.RESEARCH,
            description="Research Kafka best practices",
            context_claims=["Expert said Kafka is fast"],
        )
        assert "Kafka" in prompt
        assert len(prompt) > 50

    def test_build_scaffold_prompt(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        prompt = dispatcher.build_prompt(
            agent_type=AgentType.SCAFFOLD,
            description="Build a data pipeline",
            context_claims=["Need sub-100ms latency"],
        )
        assert "pipeline" in prompt
        assert len(prompt) > 50

    @patch("forgestream.agent_dispatcher.subprocess")
    def test_spawn_creates_tmux_session(self, mock_subprocess):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        agent = dispatcher.spawn_agent(
            agent_type=AgentType.RESEARCH,
            description="Research test",
            prompt="Test prompt",
        )

        assert agent.status == AgentStatus.RUNNING
        assert agent.agent_type == AgentType.RESEARCH
        mock_subprocess.run.assert_called()

    def test_non_requirement_event_not_spawned(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "just a claim"},
        )
        assert dispatcher.should_spawn(event) is False
