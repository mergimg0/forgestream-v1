"""Query utilities for the knowledge graph."""

from __future__ import annotations

from .model import KnowledgeGraph


class GraphQuery:
    """Read-only queries over the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def find_related(self, node: str, depth: int = 1) -> set[str]:
        """BFS to find all nodes within N hops."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(node, 0)]

        while queue:
            current, d = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if d < depth:
                for neighbor in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))

        visited.discard(node)
        return visited

    def find_isolated_clusters(self, min_size: int = 1) -> list[set[str]]:
        """Find disconnected clusters above minimum size (for seed detection)."""
        clusters = self.graph.get_disconnected_clusters()
        return [c for c in clusters if len(c) >= min_size]

    def concept_density(self) -> float:
        """Graph density: 2 * edges / (nodes * (nodes-1))."""
        nodes = self.graph.get_all_nodes()
        n = len(nodes)
        if n < 2:
            return 0.0

        edge_count = sum(
            len(self.graph.get_edges(node)) for node in nodes
        )
        return (2 * edge_count) / (n * (n - 1))

    def verified_ratio(self) -> float:
        """Fraction of concepts that are verified."""
        concepts = self.graph.concepts
        if not concepts:
            return 0.0
        verified = sum(1 for c in concepts if c.verified)
        return verified / len(concepts)
