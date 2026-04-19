"""Live feed panel -- scrolling log of claims."""

from textual.widgets import RichLog

from forgestream.events.schema import Event, EventType


class FeedPanel(RichLog):
    """Scrolling log of claim events with linkage annotations."""

    DEFAULT_CSS = """
    FeedPanel {
        height: 1fr;
        border: solid green;
    }
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[override]
        super().__init__(markup=True, **kwargs)

    def on_event_received(self, event: Event) -> None:
        """Handle an event from the EventBus."""
        if event.event_type == EventType.CLAIM:
            confidence = event.payload.get("confidence", 0.5)
            speaker = event.payload.get("speaker", "unknown")
            text = event.payload.get("text", "")
            conf_color = (
                "green" if confidence >= 0.7
                else "yellow" if confidence >= 0.4
                else "red"
            )
            self.write(
                f"[dim]{event.timestamp.strftime('%H:%M:%S')}[/dim] "
                f"[{conf_color}][{speaker}][/{conf_color}] {text} "
                f"[dim]conf:{confidence:.2f}[/dim]"
            )
            if event.payload.get("is_requirement"):
                self.write("         [yellow]>>> REQUIREMENT DETECTED[/yellow]")
            if event.payload.get("is_question"):
                self.write("         [blue]??? QUESTION[/blue]")
        elif event.event_type == EventType.ARTIFACT:
            compiles = event.payload.get("compiles", False)
            status = "[green]✓[/green]" if compiles else "[red]✗[/red]"
            self.write(f"  {status} Scaffold: {event.author}")
