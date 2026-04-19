"""Milestone A integration test -- full pipeline: claims -> orchestrator -> persistence -> TUI."""

from unittest.mock import MagicMock
from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore
from forgestream.gemini.extraction import ClaimExtractor
from forgestream.graph.materializer import GraphMaterializer
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.branches import BranchTracker
from forgestream.synthesis.requirements import RequirementDetector
from forgestream.tui.app import ForgeStreamApp


class TestMilestoneAIntegration:
    async def test_full_pipeline_with_persistence(self, db_conn):
        """Claims flow through orchestrator to PostgreSQL and event bus."""
        config = ForgeStreamConfig()
        store = EventStore(conn=db_conn)
        mock_firestore = MagicMock()

        orch = Orchestrator(config, store=store, firestore_sync=mock_firestore)

        extractor = ClaimExtractor(session_id=orch.session_id, branch_id=uuid4())
        claims_data = [
            {"text": "We need Kafka", "confidence": 0.9, "topic_keywords": ["Kafka"],
             "is_requirement": True},
            {"text": "Latency under 100ms", "confidence": 0.85, "topic_keywords": ["latency"],
             "is_requirement": True},
            {"text": "Using Python", "confidence": 0.7, "topic_keywords": ["Python"],
             "is_requirement": False},
        ]

        bus_events = []

        async def capture(event: Event):
            bus_events.append(event)

        orch.event_bus.subscribe(capture)

        for claim_data in claims_data:
            event = extractor.parse_claim(claim_data)
            result = await orch.process_event(event)
            assert result is True

        # Verify PostgreSQL persistence
        stored = await store.get_events(session_id=orch.session_id)
        assert len(stored) == 3

        # Verify Firestore sync called
        assert mock_firestore.sync_event.call_count == 3

        # Verify event bus received all events
        assert len(bus_events) == 3

        # Verify knowledge graph builds from stored events
        graph = GraphMaterializer().materialize(stored)
        assert graph.get_concept("Kafka") is not None
        assert graph.get_concept("latency") is not None

        # Verify evaluator computed
        assert all(e.evaluator > 0 for e in bus_events)

    async def test_branch_threshold_reduces_noise(self):
        """With tuned threshold, related topics don't create branches."""
        tracker = BranchTracker()

        for _ in range(5):
            tracker.add_keywords(tracker.main_branch_id,
                                 ["Kafka", "ingestion", "pipeline", "data", "streaming"])

        # Related sub-topic — should NOT branch
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["Kafka", "consumer", "partition"],
        )
        assert drift is None

        # Unrelated topic — SHOULD branch
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["quantum", "physics", "entanglement"],
        )
        assert drift is not None

    def test_tui_wires_to_orchestrator(self):
        """TUI app accepts orchestrator and can receive events."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch
        assert app.title == "ForgeStream"
