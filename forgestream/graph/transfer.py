"""Cross-meeting transfer -- merged graphs, shared concepts, seed garden."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .model import KnowledgeGraph


class CrossMeetingTransfer:
    """Detects knowledge transfer opportunities across meetings."""

    def merge_graphs(self, graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        """Merge multiple session graphs into one.

        Shared concepts get their confidence boosted (knowledge reuse).
        """
        merged = KnowledgeGraph()

        for graph in graphs:
            for concept in graph.concepts:
                existing = merged.get_concept(concept.name)
                if existing is None:
                    merged.add_concept(concept)
                else:
                    # Boost confidence for concepts seen across meetings
                    existing.confidence = max(existing.confidence, concept.confidence)
                    existing.source_events.extend(concept.source_events)

            for req in graph.requirements:
                merged.add_requirement(req)

            for art in graph.artifacts:
                merged.add_artifact(art)

            # Merge edges
            for node in graph.get_all_nodes():
                for edge in graph.get_edges(node):
                    merged.add_edge(
                        edge.source, edge.target, edge.edge_type, edge.weight
                    )

        return merged

    def find_shared_concepts(
        self,
        graph_a: KnowledgeGraph,
        graph_b: KnowledgeGraph,
    ) -> set[str]:
        """Find concepts that appear in both graphs (transfer candidates)."""
        names_a = {c.name for c in graph_a.concepts}
        names_b = {c.name for c in graph_b.concepts}
        return names_a & names_b


class SeedGarden:
    """Cross-meeting seed tracking and lifecycle management."""

    def __init__(self) -> None:
        self._seeds: dict[str, dict[str, Any]] = {}

    def add_seed(
        self,
        session_id: str,
        cluster_nodes: list[str],
        novelty_score: float,
    ) -> str:
        """Add a seed to the garden. Returns seed ID."""
        seed_id = str(uuid4())[:8]
        self._seeds[seed_id] = {
            "id": seed_id,
            "session_id": session_id,
            "cluster_nodes": cluster_nodes,
            "novelty_score": novelty_score,
            "status": "dormant",
        }
        return seed_id

    def get_seed(self, seed_id: str) -> dict[str, Any] | None:
        return self._seeds.get(seed_id)

    def list_seeds(self, status: str | None = None) -> list[dict[str, Any]]:
        seeds = list(self._seeds.values())
        if status:
            seeds = [s for s in seeds if s["status"] == status]
        return seeds

    def promote(self, seed_id: str) -> None:
        """Promote a seed to active branch status."""
        seed = self._seeds.get(seed_id)
        if seed:
            seed["status"] = "promoted"

    def archive(self, seed_id: str) -> None:
        """Archive a seed (not worth pursuing)."""
        seed = self._seeds.get(seed_id)
        if seed:
            seed["status"] = "archived"
