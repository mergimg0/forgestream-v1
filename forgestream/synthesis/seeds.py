"""Seed detection -- find disconnected concept clusters in the knowledge graph."""

from __future__ import annotations

from typing import Any

from forgestream.graph.model import KnowledgeGraph
from forgestream.graph.query import GraphQuery


class SeedDetector:
    """Detects new root topics as disconnected clusters in the knowledge graph."""

    def __init__(self, min_cluster_size: int = 3) -> None:
        self.min_cluster_size = min_cluster_size

    def detect(self, graph: KnowledgeGraph) -> list[dict[str, Any]]:
        """Find disconnected clusters that qualify as seeds.

        A seed is a cluster of concepts that:
        - Has at least min_cluster_size nodes
        - Is disconnected from the largest cluster (assumed to be "main")
        """
        query = GraphQuery(graph)
        clusters = query.find_isolated_clusters(min_size=self.min_cluster_size)

        if len(clusters) <= 1:
            return []

        # The largest cluster is assumed to be the main conversation
        clusters.sort(key=len, reverse=True)
        main_cluster = clusters[0]

        seeds = []
        for cluster in clusters[1:]:
            # Calculate novelty from cluster properties
            concepts_raw = [graph.get_concept(name) for name in cluster]
            concepts = [c for c in concepts_raw if c is not None]
            avg_confidence = (
                sum(c.confidence for c in concepts) / len(concepts)
                if concepts
                else 0.0
            )
            novelty = len(cluster) / (len(main_cluster) + len(cluster))

            seeds.append({
                "cluster_nodes": list(cluster),
                "novelty_score": novelty,
                "avg_confidence": avg_confidence,
                "domain_guess": concepts[0].domain if concepts else "",
                "description": f"Disconnected cluster: {', '.join(list(cluster)[:5])}",
            })

        return seeds
