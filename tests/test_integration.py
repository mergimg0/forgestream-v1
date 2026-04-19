"""Integration tests -- verify the full event pipeline end-to-end."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.gemini.extraction import ClaimExtractor
from forgestream.governor.axioms import AxiomChecker
from forgestream.governor.evaluator import Evaluator
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.query import GraphQuery
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.branches import BranchTracker
from forgestream.synthesis.contradictions import ContradictionDetector
from forgestream.synthesis.requirements import RequirementDetector
from forgestream.synthesis.seeds import SeedDetector
from forgestream.synthesis.suggestions import SuggestionQueue


class TestEndToEndPipeline:
    """Simulates a meeting: Gemini output → claim extraction → knowledge graph
    → requirement detection → evaluator computation → axiom checking."""

    async def test_claim_to_knowledge_graph(self):
        """Claims flow through extraction to knowledge graph."""
        session_id = uuid4()
        branch_id = uuid4()

        # Simulate Gemini output
        extractor = ClaimExtractor(session_id=session_id, branch_id=branch_id)
        gemini_outputs = [
            {"text": "We need Kafka for ingestion", "confidence": 0.85,
             "topic_keywords": ["Kafka", "ingestion"], "is_requirement": True},
            {"text": "Latency must be under 100ms", "confidence": 0.9,
             "topic_keywords": ["latency", "performance"], "is_requirement": True},
            {"text": "We use Python for the pipeline", "confidence": 0.7,
             "topic_keywords": ["Python", "pipeline"], "is_requirement": False},
        ]

        events = [extractor.parse_claim(out) for out in gemini_outputs]
        assert len(events) == 3
        assert all(e.event_type == EventType.CLAIM for e in events)

        # Materialize into knowledge graph
        materializer = GraphMaterializer()
        graph = materializer.materialize(events)

        assert graph.get_concept("Kafka") is not None
        assert graph.get_concept("latency") is not None
        assert graph.get_concept("Python") is not None
        assert len(graph.concepts) == 6  # Kafka, ingestion, latency, performance, Python, pipeline

    async def test_requirement_detection_from_claims(self):
        """Requirements are detected from claim payloads."""
        detector = RequirementDetector()

        claim_with_req = Event(
            event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
            author="gemini", evaluator=0.5,
            payload={"text": "We need a real-time dashboard", "is_requirement": True,
                     "topic_keywords": ["dashboard", "realtime"]},
        )
        claim_without = Event(
            event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
            author="gemini", evaluator=0.5,
            payload={"text": "The weather is nice", "is_requirement": False,
                     "topic_keywords": ["weather"]},
        )

        assert detector.check(claim_with_req) is not None
        assert detector.check(claim_without) is None

    async def test_evaluator_improves_with_more_data(self):
        """Evaluator score increases as more verified findings arrive."""
        evaluator = Evaluator()

        events_baseline = [
            Event(event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
                  author="g", evaluator=0.0, payload={"topic_keywords": ["A"]}),
        ]
        events_enriched = events_baseline + [
            Event(event_type=EventType.VERIFIED_FINDING, session_id=uuid4(),
                  branch_id=uuid4(), author="r", evaluator=0.0,
                  payload={"confidence": 0.9, "sources": ["x"]}),
            Event(event_type=EventType.ARTIFACT, session_id=uuid4(),
                  branch_id=uuid4(), author="s", evaluator=0.0,
                  payload={"compiles": True, "tests_pass": True}),
        ]

        score_baseline = evaluator.compute(events_baseline)
        score_enriched = evaluator.compute(events_enriched)
        assert score_enriched > score_baseline

    async def test_axioms_hold_during_normal_meeting(self):
        """SOS axioms are satisfied during a normal productive meeting."""
        checker = AxiomChecker()

        # Simulate improving evaluator trajectory
        trajectory = [0.2, 0.25, 0.3, 0.32, 0.38, 0.42, 0.45]
        mono = checker.check_monotone(trajectory)
        assert mono.holds is True

        bounded = checker.check_bounded_step(
            semantic_drift=0.3, resource_delta=1, scope_delta=5
        )
        assert bounded.holds is True

        constraint = checker.check_constraint(
            verified_claims_intact=True,
            compilation_preserved=True,
            source_chain_valid=True,
        )
        assert constraint.holds is True

    async def test_branch_detection_and_seed_lifecycle(self):
        """Topic drift creates branches; disconnected clusters become seeds."""
        tracker = BranchTracker()

        # Main topic
        for _ in range(5):
            tracker.add_keywords(tracker.main_branch_id,
                                 ["Kafka", "ingestion", "pipeline", "data"])

        # Drift into unrelated topic
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["quantum", "entanglement", "physics"],
        )
        assert drift is not None
        assert drift["potential_score"] > 0

    async def test_orchestrator_full_pipeline(self):
        """Events flow through the orchestrator pipeline correctly."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)

        processed_events = []

        async def capture(event: Event):
            processed_events.append(event)

        orch.event_bus.subscribe(capture)

        # Simulate a mini meeting
        session_id = orch.session_id
        branch_id = uuid4()

        claims = [
            Event(event_type=EventType.CLAIM, session_id=session_id,
                  branch_id=branch_id, author="gemini", evaluator=0.0,
                  payload={"text": "Use Kafka", "topic_keywords": ["Kafka"], "confidence": 0.9}),
            Event(event_type=EventType.CLAIM, session_id=session_id,
                  branch_id=branch_id, author="gemini", evaluator=0.0,
                  payload={"text": "Need sub-100ms", "topic_keywords": ["latency"], "confidence": 0.85}),
        ]

        for claim in claims:
            result = await orch.process_event(claim)
            assert result is True

        # Verified finding
        finding = Event(
            event_type=EventType.VERIFIED_FINDING, session_id=session_id,
            branch_id=branch_id, author="research_agent", evaluator=0.0,
            payload={"finding": "Kafka achieves 10ms p99",
                     "sources": [{"url": "https://kafka.apache.org"}],
                     "confidence": 0.95},
        )
        result = await orch.process_event(finding)
        assert result is True

        # Scaffold artifact
        artifact = Event(
            event_type=EventType.ARTIFACT, session_id=session_id,
            branch_id=branch_id, author="scaffold_agent", evaluator=0.0,
            payload={"files_created": ["pipeline.py"], "compiles": True, "tests_pass": True},
        )
        result = await orch.process_event(artifact)
        assert result is True

        # Verify all events were published
        assert len(processed_events) == 4

        # Verify evaluator is increasing
        evaluators = [e.evaluator for e in processed_events]
        assert evaluators[-1] > evaluators[0]

        # Verify the graph can be materialized from processed events
        materializer = GraphMaterializer()
        graph = materializer.materialize(processed_events)
        assert graph.get_concept("Kafka") is not None
        assert len(graph.artifacts) == 1
        assert graph.artifacts[0].compiles is True


class TestCrossCuttingConcerns:
    """Test that modules integrate correctly across boundaries."""

    async def test_suggestion_queue_fed_by_requirements(self):
        """Requirements detected from claims feed the suggestion queue."""
        detector = RequirementDetector()
        queue = SuggestionQueue()

        claim = Event(
            event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
            author="gemini", evaluator=0.5,
            payload={"text": "The system must handle 10k events/sec",
                     "is_requirement": False, "topic_keywords": ["throughput"]},
        )

        req = detector.check(claim)
        if req:
            from forgestream.synthesis.suggestions import Suggestion
            queue.add(Suggestion(
                text=f"Requirement detected: {req['description'][:50]}",
                priority_score=0.6,
            ))

        assert len(queue) == 1

    def test_knowledge_graph_query_after_materialization(self):
        """Graph queries work on materialized graphs."""
        materializer = GraphMaterializer()
        events = [
            Event(event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
                  author="g", evaluator=0.5,
                  payload={"topic_keywords": ["A", "B"], "confidence": 0.8}),
            Event(event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
                  author="g", evaluator=0.5,
                  payload={"topic_keywords": ["B", "C"], "confidence": 0.7}),
        ]

        graph = materializer.materialize(events)
        query = GraphQuery(graph)

        # B connects to both A and C
        related = query.find_related("B", depth=1)
        assert "A" in related
        assert "C" in related

        density = query.concept_density()
        assert density > 0
