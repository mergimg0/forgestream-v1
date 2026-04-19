from forgestream.graph.model import (
    Artifact,
    Concept,
    EdgeType,
    KnowledgeGraph,
    Requirement,
    RequirementStatus,
)


class TestConcept:
    def test_create_concept(self):
        c = Concept(name="Kafka Streams", domain="data-engineering", confidence=0.85)
        assert c.name == "Kafka Streams"
        assert c.verified is False
        assert c.source_events == []

    def test_concept_mark_verified(self):
        c = Concept(name="test", domain="test", confidence=0.9)
        c.verified = True
        assert c.verified is True


class TestRequirement:
    def test_create_requirement(self):
        r = Requirement(
            description="Sub-100ms ingestion pipeline",
            domain="data-engineering",
            complexity_estimate=0.7,
        )
        assert r.status == RequirementStatus.DETECTED
        assert r.linked_claims == []

    def test_requirement_status_transitions(self):
        r = Requirement(description="x", domain="x", complexity_estimate=0.5)
        r.status = RequirementStatus.SCAFFOLDING
        assert r.status == RequirementStatus.SCAFFOLDING
        r.status = RequirementStatus.BUILT
        assert r.status == RequirementStatus.BUILT


class TestKnowledgeGraph:
    def test_add_concept(self):
        g = KnowledgeGraph()
        c = Concept(name="Kafka", domain="data", confidence=0.8)
        g.add_concept(c)
        assert g.get_concept("Kafka") == c

    def test_add_edge(self):
        g = KnowledgeGraph()
        c1 = Concept(name="Kafka", domain="data", confidence=0.8)
        c2 = Concept(name="Flink", domain="data", confidence=0.7)
        g.add_concept(c1)
        g.add_concept(c2)
        g.add_edge(c1.name, c2.name, EdgeType.RELATES_TO, weight=0.6)

        edges = g.get_edges(c1.name)
        assert len(edges) == 1
        assert edges[0].target == c2.name
        assert edges[0].edge_type == EdgeType.RELATES_TO

    def test_add_requirement_with_supporting_concept(self):
        g = KnowledgeGraph()
        c = Concept(name="latency", domain="perf", confidence=0.9)
        r = Requirement(description="Sub-100ms", domain="perf", complexity_estimate=0.5)
        g.add_concept(c)
        g.add_requirement(r)
        g.add_edge(c.name, r.id, EdgeType.SUPPORTS)

        edges = g.get_edges(c.name)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SUPPORTS

    def test_get_disconnected_clusters(self):
        g = KnowledgeGraph()
        # Cluster 1
        c1 = Concept(name="A", domain="x", confidence=0.5)
        c2 = Concept(name="B", domain="x", confidence=0.5)
        g.add_concept(c1)
        g.add_concept(c2)
        g.add_edge("A", "B", EdgeType.RELATES_TO)

        # Cluster 2 (disconnected)
        c3 = Concept(name="C", domain="y", confidence=0.5)
        c4 = Concept(name="D", domain="y", confidence=0.5)
        g.add_concept(c3)
        g.add_concept(c4)
        g.add_edge("C", "D", EdgeType.RELATES_TO)

        clusters = g.get_disconnected_clusters()
        assert len(clusters) == 2

    def test_get_neighbors(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="A", domain="x", confidence=0.5))
        g.add_concept(Concept(name="B", domain="x", confidence=0.5))
        g.add_concept(Concept(name="C", domain="x", confidence=0.5))
        g.add_edge("A", "B", EdgeType.RELATES_TO)
        g.add_edge("A", "C", EdgeType.RELATES_TO)

        neighbors = g.get_neighbors("A")
        assert "B" in neighbors
        assert "C" in neighbors
        assert len(neighbors) == 2
