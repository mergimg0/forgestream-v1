"""Suggestions panel -- priority-sorted queue display."""

from textual.widgets import Static

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.suggestions import Priority, Suggestion, SuggestionQueue

PRIORITY_COLORS = {
    Priority.CRITICAL: "red",
    Priority.STRATEGIC: "yellow",
    Priority.DELVE_DEEPER: "blue",
    Priority.GOOD_TO_PROBE: "green",
    Priority.NICE_TO_KNOW: "dim",
}

PRIORITY_ICONS = {
    Priority.CRITICAL: "!!",
    Priority.STRATEGIC: "▲",
    Priority.DELVE_DEEPER: "◆",
    Priority.GOOD_TO_PROBE: "○",
    Priority.NICE_TO_KNOW: "·",
}


class SuggestionsPanel(Static):
    """Displays the suggestion priority queue."""

    DEFAULT_CSS = """
    SuggestionsPanel {
        height: 1fr;
        border: solid cyan;
    }
    """

    def __init__(self, queue: SuggestionQueue | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.queue = queue or SuggestionQueue()

    def on_event_received(self, event: Event) -> None:
        """Handle suggestion-related events."""
        if event.event_type == EventType.SUGGESTION:
            self.queue.add(Suggestion(
                text=event.payload.get("text", ""),
                priority_score=event.payload.get("priority", 0.5),
            ))
            self.refresh_display()

    def refresh_display(self) -> None:
        lines = ["[bold]SUGGESTION QUEUE[/bold]\n"]
        for suggestion in self.queue.get_all()[:10]:
            color = PRIORITY_COLORS.get(suggestion.category, "white")
            icon = PRIORITY_ICONS.get(suggestion.category, " ")
            lines.append(
                f"[{color}]{icon} {suggestion.category.value.upper()}[/{color}]\n"
                f"  {suggestion.text}\n"
            )
        if not self.queue.get_all():
            lines.append("[dim]No suggestions yet[/dim]")
        self.update("\n".join(lines))
