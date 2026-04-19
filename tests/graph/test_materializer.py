from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.model import EdgeType


class TestGraphMaterializer:
    def test_materialize_claim_creates_concepts(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "We should use Kafka for ingestion",
                "confidence": 0.85,
                "topic_keywords": ["Kafka", "ingestion"],
            },
        )
        graph = m.materialize([event])
        assert graph.get_concept("Kafka") is not None
        assert graph.get_concept("ingestion") is not None

    def test_materialize_requirement_event(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.6,
            payload={
                "description": "Sub-100ms ingestion",
                "domain": "data-engineering",
                "complexity_estimate": 0.7,
                "linked_claims": [],
            },
        )
        graph = m.materialize([event])
        reqs = graph.requirements
        assert len(reqs) == 1
        assert reqs[0].description == "Sub-100ms ingestion"

    def test_materialize_artifact_event(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.ARTIFACT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="scaffold-001",
            evaluator=0.7,
            payload={
                "worktree_path": "../forgestream-sc-001",
                "branch_name": "sc/pipeline",
                "files_created": ["src/main.py"],
                "compiles": True,
                "tests_pass": True,
            },
        )
        graph = m.materialize([event])
        arts = graph.artifacts
        assert len(arts) == 1
        assert arts[0].compiles is True

    def test_materialize_contradiction_creates_edge(self):
        m = GraphMaterializer()
        events = [
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "Use strong consistency",
                    "confidence": 0.8,
                    "topic_keywords": ["strong_consistency"],
                },
            ),
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "Use eventual consistency",
                    "confidence": 0.7,
                    "topic_keywords": ["eventual_consistency"],
                },
            ),
            Event(
                event_type=EventType.CONTRADICTION,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="synthesis",
                evaluator=0.6,
                payload={
                    "concept_a": "strong_consistency",
                    "concept_b": "eventual_consistency",
                    "explanation": "Contradictory consistency models",
                },
            ),
        ]
        graph = m.materialize(events)
        edges = graph.get_edges("strong_consistency", EdgeType.CONTRADICTS)
        assert len(edges) == 1
        assert edges[0].target == "eventual_consistency"

    def test_rebuild_from_events_is_idempotent(self):
        m = GraphMaterializer()
        events = [
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "test",
                    "confidence": 0.8,
                    "topic_keywords": ["Kafka"],
                },
            ),
        ]
        graph1 = m.materialize(events)
        graph2 = m.materialize(events)
        assert len(graph1.concepts) == len(graph2.concepts)
