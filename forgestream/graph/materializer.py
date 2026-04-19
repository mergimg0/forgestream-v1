"""Materializes the knowledge graph from the event log (event sourcing)."""

from __future__ import annotations

from forgestream.events.schema import Event, EventType

from .model import (
    Artifact,
    Concept,
    EdgeType,
    KnowledgeGraph,
    Requirement,
)


class GraphMaterializer:
    """Replays events to build the knowledge graph.

    Stateless: call materialize() with a full event list to get a fresh graph.
    """

    def materialize(self, events: list[Event]) -> KnowledgeGraph:
        """Build a knowledge graph from a list of events."""
        graph = KnowledgeGraph()

        for event in events:
            handler = self._handlers.get(event.event_type)
            if handler:
                handler(self, graph, event)

        return graph

    def _handle_claim(self, graph: KnowledgeGraph, event: Event) -> None:
        keywords = event.payload.get("topic_keywords", [])
        confidence = event.payload.get("confidence", 0.5)

        for keyword in keywords:
            existing = graph.get_concept(keyword)
            if existing is None:
                concept = Concept(
                    name=keyword,
                    domain="",
                    confidence=confidence,
                    source_events=[str(event.id)],
                )
                graph.add_concept(concept)
            else:
                existing.confidence = max(existing.confidence, confidence)
                existing.source_events.append(str(event.id))

        # Add edges between co-occurring keywords
        for i, kw_a in enumerate(keywords):
            for kw_b in keywords[i + 1:]:
                graph.add_edge(kw_a, kw_b, EdgeType.RELATES_TO, weight=confidence)

    def _handle_requirement(self, graph: KnowledgeGraph, event: Event) -> None:
        req = Requirement(
            description=event.payload["description"],
            domain=event.payload.get("domain", ""),
            complexity_estimate=event.payload.get("complexity_estimate", 0.5),
            linked_claims=event.payload.get("linked_claims", []),
        )
        graph.add_requirement(req)

    def _handle_artifact(self, graph: KnowledgeGraph, event: Event) -> None:
        artifact = Artifact(
            path=event.payload.get("worktree_path", ""),
            branch=event.payload.get("branch_name", ""),
            compiles=event.payload.get("compiles", False),
            tests_pass=event.payload.get("tests_pass", False),
        )
        graph.add_artifact(artifact)

    def _handle_contradiction(self, graph: KnowledgeGraph, event: Event) -> None:
        concept_a = event.payload.get("concept_a", "")
        concept_b = event.payload.get("concept_b", "")
        if concept_a and concept_b:
            graph.add_edge(concept_a, concept_b, EdgeType.CONTRADICTS)

    def _handle_prosodic_feature(self, graph: KnowledgeGraph, event: Event) -> None:
        speaker = event.payload.get("speaker_id", "unknown")
        emotion = event.payload.get("emotion_tag") or "neutral"
        arousal = event.payload.get("arousal", 0.5)
        # Create or update speaker concept node
        existing = graph.get_concept(speaker)
        if existing is None:
            concept = Concept(
                name=speaker,
                domain="speaker",
                confidence=arousal,
                source_events=[str(event.id)],
            )
            graph.add_concept(concept)
        else:
            existing.confidence = arousal
            existing.source_events.append(str(event.id))

    _handlers = {
        EventType.CLAIM: _handle_claim,
        EventType.REQUIREMENT: _handle_requirement,
        EventType.ARTIFACT: _handle_artifact,
        EventType.CONTRADICTION: _handle_contradiction,
        EventType.PROSODIC_FEATURE: _handle_prosodic_feature,
    }
