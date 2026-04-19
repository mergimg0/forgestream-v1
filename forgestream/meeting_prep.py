"""Meeting Preparation Mode — generates a prep document before a meeting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .events.schema import Event, EventType
from .graph.materializer import GraphMaterializer
from .graph.model import EdgeType, KnowledgeGraph


class MeetingPrep:
    """Generate a markdown preparation document for the next meeting.

    Loads the most recent events export from data_dir, materializes the
    knowledge graph, then identifies knowledge gaps, unresolved contradictions,
    active seeds, and generates suggested questions.
    """

    #: Concepts with confidence below this threshold are treated as gaps.
    GAP_THRESHOLD: float = 0.5

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self._materializer = GraphMaterializer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, topic: str = "") -> str:
        """Generate a prep document for the next meeting.

        Loads latest events from data/, identifies knowledge gaps,
        unresolved contradictions, active seeds, and generates questions.

        Args:
            topic: Optional topic hint to focus the questions.

        Returns:
            Markdown string ready to display or print.
        """
        events = self._load_latest_events()
        graph = self._materializer.materialize(events)
        gaps = self._find_knowledge_gaps(graph)
        contradictions = self._find_unresolved_contradictions(events)
        seeds = self._find_active_seeds(events)
        questions = self._generate_questions(gaps, contradictions, seeds, topic)
        return self._format_prep_doc(gaps, contradictions, seeds, questions, topic)

    # ------------------------------------------------------------------
    # Internal helpers — also exposed for direct testing
    # ------------------------------------------------------------------

    def _find_knowledge_gaps_from_events(self, events: list[Event]) -> list[dict[str, Any]]:
        """Return low-confidence concepts from an event list (test helper)."""
        graph = self._materializer.materialize(events)
        return self._find_knowledge_gaps(graph)

    def _load_latest_events(self) -> list[Event]:
        """Load the most recent events export from data_dir.

        Looks for ``events_export.json`` first, then the most recently
        modified ``*.json`` file whose name contains "events". Falls back
        to an empty list if nothing is found.
        """
        # Preferred file name
        preferred = self.data_dir / "events_export.json"
        if preferred.exists():
            return self._parse_events_file(preferred)

        # Fallback: any JSON file with "events" in the name
        candidates = sorted(
            self.data_dir.glob("*events*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return self._parse_events_file(candidates[0])

        return []

    def _parse_events_file(self, path: Path) -> list[Event]:
        """Parse a JSON events file into a list of Event objects."""
        try:
            raw: list[dict] = json.loads(path.read_text())
            return [Event.from_dict(item) for item in raw]
        except (json.JSONDecodeError, KeyError, ValueError):
            return []

    def _find_knowledge_gaps(self, graph: KnowledgeGraph) -> list[dict[str, Any]]:
        """Return concepts with confidence < GAP_THRESHOLD."""
        gaps: list[dict[str, Any]] = []
        for concept in graph.concepts:
            # Skip speaker nodes (domain == "speaker")
            if concept.domain == "speaker":
                continue
            if concept.confidence < self.GAP_THRESHOLD:
                gaps.append(
                    {
                        "name": concept.name,
                        "confidence": concept.confidence,
                        "domain": concept.domain,
                    }
                )
        # Sort lowest confidence first
        gaps.sort(key=lambda g: g["confidence"])
        return gaps

    def _find_unresolved_contradictions(
        self, events: list[Event]
    ) -> list[dict[str, Any]]:
        """Return CONTRADICTION events that are not marked as resolved."""
        contradictions: list[dict[str, Any]] = []
        for event in events:
            if event.event_type != EventType.CONTRADICTION:
                continue
            payload = event.payload
            if payload.get("resolved", False):
                continue
            contradictions.append(
                {
                    "concept_a": payload.get("concept_a", ""),
                    "concept_b": payload.get("concept_b", ""),
                    "description": payload.get("description", ""),
                }
            )
        return contradictions

    def _find_active_seeds(self, events: list[Event]) -> list[dict[str, Any]]:
        """Return SEED events with status == 'active'."""
        seeds: list[dict[str, Any]] = []
        for event in events:
            if event.event_type != EventType.SEED:
                continue
            payload = event.payload
            if payload.get("status", "") == "active":
                seeds.append(
                    {
                        "hypothesis": payload.get("hypothesis", ""),
                        "confidence": payload.get("confidence", 0.5),
                    }
                )
        return seeds

    def _generate_questions(
        self,
        gaps: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
        seeds: list[dict[str, Any]],
        topic: str,
    ) -> list[str]:
        """Generate suggested questions for the upcoming meeting."""
        questions: list[str] = []

        # Topic-focused question
        if topic:
            questions.append(f"What are the key open questions around **{topic}**?")

        # Questions from knowledge gaps
        for gap in gaps[:3]:
            name = gap["name"]
            conf = gap["confidence"]
            questions.append(
                f"Can you elaborate on **{name}**? "
                f"(current confidence: {conf:.0%})"
            )

        # Questions from contradictions
        for c in contradictions[:3]:
            a, b = c["concept_a"], c["concept_b"]
            if a and b:
                questions.append(
                    f"How do you reconcile the tension between **{a}** and **{b}**?"
                )

        # Questions from active seeds
        for seed in seeds[:2]:
            hyp = seed["hypothesis"]
            if hyp:
                questions.append(f"Can we make progress on: *{hyp}*?")

        # Fallback question
        if not questions:
            questions.append(
                "What are the key open questions you'd like to address today?"
            )

        return questions

    def _format_prep_doc(
        self,
        gaps: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
        seeds: list[dict[str, Any]],
        questions: list[str],
        topic: str,
    ) -> str:
        """Render the prep document as a markdown string."""
        heading = f"Meeting Preparation{f': {topic}' if topic else ''}"
        lines: list[str] = [
            f"# {heading}",
            "",
        ]

        # Knowledge Gaps
        lines.append("## Knowledge Gaps")
        if gaps:
            for g in gaps:
                lines.append(
                    f"- **{g['name']}** — confidence {g['confidence']:.0%}"
                )
        else:
            lines.append("- No low-confidence concepts detected.")
        lines.append("")

        # Unresolved Contradictions
        lines.append("## Unresolved Contradictions")
        if contradictions:
            for c in contradictions:
                desc = c["description"] or f"{c['concept_a']} vs {c['concept_b']}"
                lines.append(f"- {desc}")
        else:
            lines.append("- No unresolved contradictions.")
        lines.append("")

        # Active Seeds
        lines.append("## Active Seeds")
        if seeds:
            for s in seeds:
                lines.append(f"- {s['hypothesis']}")
        else:
            lines.append("- No active seeds.")
        lines.append("")

        # Suggested Questions
        lines.append("## Suggested Questions")
        for q in questions:
            lines.append(f"- {q}")
        lines.append("")

        return "\n".join(lines)
