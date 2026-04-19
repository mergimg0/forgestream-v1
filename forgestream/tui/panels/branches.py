"""Branches panel -- conversation tree with metrics."""

from textual.widgets import Static

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.branches import BranchTracker


class BranchesPanel(Static):
    """Displays conversation branches with potential/momentum/ROI."""

    DEFAULT_CSS = """
    BranchesPanel {
        height: auto;
        max-height: 8;
        border: solid magenta;
    }
    """

    def __init__(self, tracker: BranchTracker | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.tracker = tracker or BranchTracker()

    def on_event_received(self, event: Event) -> None:
        """Update branch tracking on claim events."""
        if event.event_type == EventType.CLAIM:
            keywords = event.payload.get("topic_keywords", [])
            if keywords:
                self.tracker.add_keywords(self.tracker.main_branch_id, keywords)
                self.refresh_display()

    def refresh_display(self) -> None:
        lines = ["[bold]BRANCHES[/bold]"]
        for branch in self.tracker.all_branches[:12]:
            metrics = self.tracker.get_metrics(branch.id)
            pot = metrics["potential"]
            claims = metrics["claim_count"]
            prefix = "━" * min(18, max(1, claims * 2))
            indent = "" if branch.parent_branch_id is None else "├─ "
            lines.append(
                f"  {indent}{branch.name} {prefix} pot:{pot:.2f} [{claims} claims]"
            )
        self.update("\n".join(lines))
