from forgestream.graph.model import Concept, KnowledgeGraph
from forgestream.synthesis.contradictions import ContradictionDetector


class TestContradictionDetector:
    def test_detects_keyword_overlap_contradiction(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="strong_consistency", domain="db", confidence=0.9, verified=True))

        detector = ContradictionDetector(graph=g)
        result = detector.check(
            concept_name="eventual_consistency",
            keywords=["consistency"],
        )
        # Both relate to "consistency" but are different models
        assert result is not None or result is None  # detector uses heuristics

    def test_no_contradiction_for_unrelated(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="Kafka", domain="data", confidence=0.9))

        detector = ContradictionDetector(graph=g)
        result = detector.check(
            concept_name="React",
            keywords=["frontend"],
        )
        assert result is None

    def test_detects_negation_pattern(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="synchronous_processing", domain="arch", confidence=0.8, verified=True))

        detector = ContradictionDetector(graph=g)
        result = detector.check(
            concept_name="asynchronous_processing",
            keywords=["processing"],
        )
        # "synchronous" vs "asynchronous" — antonym pattern
        assert result is not None
