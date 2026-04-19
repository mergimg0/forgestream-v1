"""Monitor running Claude Code agents via tmux."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class AgentOutput:
    last_lines: list[str]
    is_running: bool


class AgentMonitor:
    """Monitors agent progress by reading tmux pane output."""

    @staticmethod
    def capture(tmux_session: str, lines: int = 5) -> AgentOutput:
        """Capture the last N lines from an agent's tmux session."""
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", tmux_session, "-p", "-l", str(lines)],
            capture_output=True,
            text=True,
        )
        output_lines = result.stdout.strip().splitlines() if result.stdout else []

        check = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            capture_output=True,
        )
        is_running = check.returncode == 0

        return AgentOutput(last_lines=output_lines, is_running=is_running)

    @staticmethod
    def kill(tmux_session: str) -> None:
        """Kill an agent's tmux session."""
        subprocess.run(
            ["tmux", "kill-session", "-t", tmux_session],
            capture_output=True,
        )
