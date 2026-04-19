from forgestream.agents.registry import AgentInfo, AgentRegistry, AgentStatus, AgentType


class TestAgentRegistry:
    def test_register_agent(self):
        reg = AgentRegistry()
        agent = reg.register(
            agent_type=AgentType.RESEARCH,
            task_description="Research Kafka Streams",
        )
        assert agent.status == AgentStatus.PROVISIONING
        assert agent.agent_type == AgentType.RESEARCH

    def test_get_active_agents(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "task 1")
        a2 = reg.register(AgentType.SCAFFOLD, "task 2")
        reg.update_status(a2.id, AgentStatus.RUNNING)

        active = reg.get_active()
        assert len(active) == 2

    def test_get_by_type(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "r1")
        reg.register(AgentType.RESEARCH, "r2")
        reg.register(AgentType.SCAFFOLD, "s1")

        research = reg.get_by_type(AgentType.RESEARCH)
        assert len(research) == 2

    def test_update_status(self):
        reg = AgentRegistry()
        agent = reg.register(AgentType.SCAFFOLD, "test")
        reg.update_status(agent.id, AgentStatus.RUNNING)
        assert reg.get(agent.id).status == AgentStatus.RUNNING

    def test_count_by_type(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "r1")
        reg.register(AgentType.RESEARCH, "r2")
        reg.register(AgentType.SCAFFOLD, "s1")

        counts = reg.count_active_by_type()
        assert counts[AgentType.RESEARCH] == 2
        assert counts[AgentType.SCAFFOLD] == 1

    def test_completed_agents_not_active(self):
        reg = AgentRegistry()
        agent = reg.register(AgentType.RESEARCH, "done")
        reg.update_status(agent.id, AgentStatus.COMPLETED)

        active = reg.get_active()
        assert len(active) == 0
