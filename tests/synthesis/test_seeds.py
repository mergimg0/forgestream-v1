from forgestream.graph.model import Concept, EdgeType, KnowledgeGraph
from forgestream.synthesis.seeds import SeedDetector


class TestSeedDetector:
    def test_detects_disconnected_cluster_as_seed(self):
        g = KnowledgeGraph()
        # Connected cluster (main) — must be >= min_cluster_size
        g.add_concept(Concept(name="A", domain="x", confidence=0.8))
        g.add_concept(Concept(name="B", domain="x", confidence=0.7))
        g.add_concept(Concept(name="C", domain="x", confidence=0.6))
        g.add_edge("A", "B", EdgeType.RELATES_TO)
        g.add_edge("B", "C", EdgeType.RELATES_TO)

        # Disconnected cluster (seed candidate)
        g.add_concept(Concept(name="X", domain="y", confidence=0.6))
        g.add_concept(Concept(name="Y", domain="y", confidence=0.5))
        g.add_concept(Concept(name="Z", domain="y", confidence=0.7))
        g.add_edge("X", "Y", EdgeType.RELATES_TO)
        g.add_edge("Y", "Z", EdgeType.RELATES_TO)

        detector = SeedDetector(min_cluster_size=3)
        seeds = detector.detect(g)
        assert len(seeds) == 1
        assert seeds[0]["novelty_score"] > 0

    def test_no_seed_for_small_clusters(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="lone", domain="x", confidence=0.5))

        detector = SeedDetector(min_cluster_size=3)
        seeds = detector.detect(g)
        assert len(seeds) == 0

    def test_no_seed_when_all_connected(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="A", domain="x", confidence=0.8))
        g.add_concept(Concept(name="B", domain="x", confidence=0.7))
        g.add_concept(Concept(name="C", domain="x", confidence=0.6))
        g.add_edge("A", "B", EdgeType.RELATES_TO)
        g.add_edge("B", "C", EdgeType.RELATES_TO)

        detector = SeedDetector(min_cluster_size=3)
        seeds = detector.detect(g)
        assert len(seeds) == 0  # only one cluster, it's the main graph
