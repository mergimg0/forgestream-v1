import json
from uuid import uuid4

from forgestream.agents.templates.research import ResearchTemplate
from forgestream.agents.templates.scaffold import ScaffoldTemplate
from forgestream.events.schema import EventType


class TestResearchTemplate:
    def test_build_prompt(self):
        tmpl = ResearchTemplate()
        prompt = tmpl.build_prompt(
            query="What are best practices for sub-100ms data ingestion?",
            context_claims=["Expert said Kafka is fast", "Latency requirement is critical"],
        )
        assert "sub-100ms" in prompt
        assert "Kafka" in prompt
        assert len(prompt) > 50

    def test_parse_output_valid(self):
        tmpl = ResearchTemplate()
        output = json.dumps({
            "query": "ingestion best practices",
            "finding": "Kafka Streams achieves 10ms p99",
            "sources": [{"url": "https://example.com", "title": "Kafka Docs"}],
            "verification_chain": "Verified via official docs",
            "confidence": 0.9,
            "connections": ["Kafka", "latency"],
            "growth_vectors": ["Flink comparison"],
        })
        event = tmpl.parse_output(output, session_id=uuid4(), branch_id=uuid4())
        assert event is not None
        assert event.event_type == EventType.VERIFIED_FINDING
        assert event.payload["confidence"] == 0.9

    def test_parse_output_invalid_json(self):
        tmpl = ResearchTemplate()
        event = tmpl.parse_output("not json", session_id=uuid4(), branch_id=uuid4())
        assert event is None


class TestScaffoldTemplate:
    def test_build_prompt(self):
        tmpl = ScaffoldTemplate()
        prompt = tmpl.build_prompt(
            requirement="Build a real-time ingestion pipeline",
            domain="data-engineering",
            verified_findings=["Kafka Streams achieves 10ms p99"],
        )
        assert "ingestion pipeline" in prompt
        assert "Kafka" in prompt

    def test_parse_output_valid(self):
        tmpl = ScaffoldTemplate()
        output = json.dumps({
            "files_created": ["src/pipeline.py", "tests/test_pipeline.py"],
            "compiles": True,
            "tests_pass": True,
            "design_decisions": ["Used Kafka Streams for ingestion"],
            "open_questions": ["Exactly-once semantics needed?"],
            "estimated_completeness": 0.7,
        })
        result = tmpl.parse_output(output, session_id=uuid4(), branch_id=uuid4())
        assert result is not None
        artifact, suggestions = result
        assert artifact.event_type == EventType.ARTIFACT
        assert artifact.payload["compiles"] is True
        assert len(suggestions) == 1  # one open question
        assert suggestions[0].event_type == EventType.SUGGESTION
