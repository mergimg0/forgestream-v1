from forgestream.graph.model import Concept, KnowledgeGraph, Requirement
from forgestream.gemini.context import ContextBuilder


class TestContextBuilder:
    def test_build_summary(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="Kafka", domain="data", confidence=0.9, verified=True))
        g.add_concept(Concept(name="latency", domain="perf", confidence=0.8))
        g.add_requirement(Requirement(
            description="Sub-100ms ingestion", domain="data",
            complexity_estimate=0.7,
        ))

        builder = ContextBuilder()
        summary = builder.build_injection(g, active_branches=["main", "burst-handling"])

        assert "Kafka" in summary
        assert "Sub-100ms" in summary
        assert "burst-handling" in summary
        assert isinstance(summary, str)
        assert len(summary) < 2000

    def test_empty_graph(self):
        g = KnowledgeGraph()
        builder = ContextBuilder()
        summary = builder.build_injection(g, active_branches=[])
        assert isinstance(summary, str)
