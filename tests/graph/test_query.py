from forgestream.graph.model import Concept, EdgeType, KnowledgeGraph
from forgestream.graph.query import GraphQuery


class TestGraphQuery:
    def _build_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        for name in ["A", "B", "C", "D", "E"]:
            g.add_concept(Concept(name=name, domain="test", confidence=0.5))
        g.add_edge("A", "B", EdgeType.RELATES_TO, weight=0.8)
        g.add_edge("B", "C", EdgeType.RELATES_TO, weight=0.6)
        g.add_edge("D", "E", EdgeType.RELATES_TO, weight=0.7)
        return g

    def test_find_related(self):
        g = self._build_graph()
        q = GraphQuery(g)
        related = q.find_related("A", depth=2)
        assert "B" in related
        assert "C" in related
        assert "D" not in related

    def test_find_isolated_clusters(self):
        g = self._build_graph()
        q = GraphQuery(g)
        clusters = q.find_isolated_clusters(min_size=2)
        assert len(clusters) == 2

    def test_concept_density(self):
        g = self._build_graph()
        q = GraphQuery(g)
        density = q.concept_density()
        assert 0.0 < density <= 1.0

    def test_verified_ratio(self):
        g = KnowledgeGraph()
        c1 = Concept(name="A", domain="x", confidence=0.9, verified=True)
        c2 = Concept(name="B", domain="x", confidence=0.5)
        g.add_concept(c1)
        g.add_concept(c2)
        q = GraphQuery(g)
        assert q.verified_ratio() == 0.5

    def test_verified_ratio_empty(self):
        g = KnowledgeGraph()
        q = GraphQuery(g)
        assert q.verified_ratio() == 0.0
