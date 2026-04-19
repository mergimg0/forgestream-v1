"""ForgeStream TUI -- primary meeting interface."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from forgestream.events.schema import Event
from forgestream.orchestrator import Orchestrator

from .panels.agents import AgentsPanel
from .panels.branches import BranchesPanel
from .panels.feed import FeedPanel
from .panels.suggestions import SuggestionsPanel


class EvaluatorBar(Static):
    """Compact evaluator display in the header area."""

    DEFAULT_CSS = """
    EvaluatorBar {
        height: 1;
        dock: top;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def update_evaluator(self, value: float, event_count: int, mode: str) -> None:
        bar_len = 20
        filled = int(value * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        self.update(
            f"E(π)={value:.3f} [{bar}] events:{event_count} mode:{mode.upper()}"
        )


class ForgeStreamApp(App):
    """ForgeStream terminal UI for live meetings."""

    TITLE = "ForgeStream"
    SUB_TITLE = "Live Meeting Intelligence"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-content {
        layout: horizontal;
        height: 1fr;
    }
    #left-column {
        width: 60%;
    }
    #right-column {
        width: 40%;
    }
    #bottom-bar {
        height: auto;
        max-height: 12;
        layout: horizontal;
    }
    #branches-container {
        width: 60%;
    }
    #agents-container {
        width: 40%;
    }
    """

    BINDINGS = [
        Binding("m", "cycle_mode", "Mode"),
        Binding("s", "dismiss_suggestion", "Dismiss"),
        Binding("p", "pause_agents", "Pause"),
        Binding("r", "resume_agents", "Resume"),
        Binding("q", "toggle_quiet", "Quiet"),
        Binding("space", "bookmark", "Bookmark"),
        Binding("escape", "reset_view", "Reset"),
        Binding("e", "end_meeting", "End"),
    ]

    def __init__(self, orchestrator: Orchestrator | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self._mode = "extract"
        self._quiet = False
        self._event_count = 0
        self._meeting_ended = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield EvaluatorBar(id="evaluator-bar")
        with Horizontal(id="main-content"):
            with Vertical(id="left-column"):
                yield FeedPanel(id="feed")
            with Vertical(id="right-column"):
                yield SuggestionsPanel(id="suggestions")
        with Horizontal(id="bottom-bar"):
            with Vertical(id="branches-container"):
                yield BranchesPanel(id="branches")
            with Vertical(id="agents-container"):
                yield AgentsPanel(id="agents")
        yield Footer()

    async def on_mount(self) -> None:
        """Subscribe to the orchestrator's event bus when mounted."""
        if self.orchestrator:
            self.orchestrator.event_bus.subscribe(self._on_event)

    async def _on_event(self, event: Event) -> None:
        """Route events from the EventBus to the appropriate panels."""
        self._event_count += 1

        feed = self.query_one("#feed", FeedPanel)
        feed.on_event_received(event)

        suggestions = self.query_one("#suggestions", SuggestionsPanel)
        suggestions.on_event_received(event)

        branches = self.query_one("#branches", BranchesPanel)
        branches.on_event_received(event)

        evaluator_bar = self.query_one("#evaluator-bar", EvaluatorBar)
        evaluator_bar.update_evaluator(
            event.evaluator, self._event_count, self._mode
        )

    def action_cycle_mode(self) -> None:
        modes = ["extract", "collaborative", "knowledge"]
        idx = modes.index(self._mode)
        self._mode = modes[(idx + 1) % len(modes)]
        self.sub_title = f"Mode: {self._mode.upper()}"

    def action_dismiss_suggestion(self) -> None:
        panel = self.query_one("#suggestions", SuggestionsPanel)
        panel.queue.dismiss()
        panel.refresh_display()

    def action_pause_agents(self) -> None:
        self.sub_title = "AGENTS PAUSED"

    def action_resume_agents(self) -> None:
        self.sub_title = f"Mode: {self._mode.upper()}"

    def action_toggle_quiet(self) -> None:
        self._quiet = not self._quiet

    def action_bookmark(self) -> None:
        feed = self.query_one("#feed", FeedPanel)
        feed.write("[bold yellow]>>> BOOKMARK <<<[/bold yellow]")

    def action_reset_view(self) -> None:
        self._quiet = False

    def action_end_meeting(self) -> None:
        """End the current meeting and trigger post-meeting synthesis."""
        self._meeting_ended = True
        feed = self.query_one("#feed", FeedPanel)
        feed.write("[bold cyan]>>> MEETING ENDED[/bold cyan]")
        feed.write("[cyan]Rate this meeting (1-10, or Enter to skip):[/cyan]")
        self.sub_title = "Meeting Ended — Rate 1-10"

    @staticmethod
    def _parse_feedback(text: str) -> float | None:
        """Parse human feedback score (1-10) to float (0.1-1.0)."""
        text = text.strip()
        if not text or text.lower() in ("skip", "s"):
            return None
        try:
            score = int(text)
            if 1 <= score <= 10:
                return score / 10.0
        except ValueError:
            pass
        return None
