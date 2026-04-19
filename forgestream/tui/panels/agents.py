"""Agents panel -- status of running Claude Code instances."""

from textual.widgets import Static

from forgestream.agents.registry import AgentRegistry, AgentStatus


STATUS_ICONS = {
    AgentStatus.PROVISIONING: "░",
    AgentStatus.RUNNING: "▶",
    AgentStatus.COMPLETED: "■",
    AgentStatus.FAILED: "✗",
}


class AgentsPanel(Static):
    """Displays status of all running agent instances."""

    DEFAULT_CSS = """
    AgentsPanel {
        height: auto;
        max-height: 8;
        border: solid yellow;
    }
    """

    def __init__(self, registry: AgentRegistry | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.registry = registry or AgentRegistry()

    def on_mount(self) -> None:
        self.refresh_display()

    def refresh_display(self) -> None:
        lines = ["[bold]AGENTS[/bold]"]
        all_agents = list(self.registry._agents.values())
        if not all_agents:
            lines.append("  [dim]No agents running[/dim]")
        for agent in all_agents[-8:]:
            icon = STATUS_ICONS.get(agent.status, "?")
            status_color = (
                "green" if agent.status == AgentStatus.COMPLETED
                else "red" if agent.status == AgentStatus.FAILED
                else "yellow"
            )
            lines.append(
                f"  {icon} [{status_color}]{agent.id}[/{status_color}] "
                f"{agent.task_description[:40]}"
            )
        self.update("\n".join(lines))
