"""Knowledge graph model -- materialized view over the event log."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class RequirementStatus(str, Enum):
    DETECTED = "detected"
    SCAFFOLDING = "scaffolding"
    BUILT = "built"
    VERIFIED = "verified"


class EdgeType(str, Enum):
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    FULFILLED_BY = "fulfilled_by"
    CONTRADICTS = "contradicts"


@dataclass
class Concept:
    name: str
    domain: str
    confidence: float
    verified: bool = False
    source_events: list[str] = field(default_factory=list)


@dataclass
class Requirement:
    description: str
    domain: str
    complexity_estimate: float
    id: str = field(default_factory=lambda: str(uuid4()))
    status: RequirementStatus = RequirementStatus.DETECTED
    linked_claims: list[str] = field(default_factory=list)


@dataclass
class Artifact:
    path: str
    branch: str
    compiles: bool
    tests_pass: bool
    id: str = field(default_factory=lambda: str(uuid4()))
    linked_requirements: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0


class KnowledgeGraph:
    """In-memory graph rebuilt from event log on startup."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._requirements: dict[str, Requirement] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._edges: dict[str, list[Edge]] = {}

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.name] = concept

    def get_concept(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    @property
    def concepts(self) -> list[Concept]:
        return list(self._concepts.values())

    def add_requirement(self, req: Requirement) -> None:
        self._requirements[req.id] = req

    def get_requirement(self, req_id: str) -> Requirement | None:
        return self._requirements.get(req_id)

    @property
    def requirements(self) -> list[Requirement]:
        return list(self._requirements.values())

    def add_artifact(self, artifact: Artifact) -> None:
        self._artifacts[artifact.id] = artifact

    @property
    def artifacts(self) -> list[Artifact]:
        return list(self._artifacts.values())

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        edge = Edge(source=source, target=target, edge_type=edge_type, weight=weight)
        self._edges.setdefault(source, []).append(edge)

    def get_edges(
        self, source: str, edge_type: EdgeType | None = None
    ) -> list[Edge]:
        edges = self._edges.get(source, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_all_nodes(self) -> set[str]:
        nodes: set[str] = set()
        nodes.update(self._concepts.keys())
        nodes.update(self._requirements.keys())
        nodes.update(self._artifacts.keys())
        return nodes

    def get_neighbors(self, node: str) -> set[str]:
        neighbors: set[str] = set()
        for edge in self._edges.get(node, []):
            neighbors.add(edge.target)
        for source, edges in self._edges.items():
            for edge in edges:
                if edge.target == node:
                    neighbors.add(source)
        return neighbors

    def get_disconnected_clusters(self) -> list[set[str]]:
        """Find connected components in the graph (for seed detection)."""
        all_nodes = self.get_all_nodes()
        visited: set[str] = set()
        clusters: list[set[str]] = []

        for node in all_nodes:
            if node in visited:
                continue
            cluster: set[str] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                for neighbor in self.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if cluster:
                clusters.append(cluster)

        return clusters
