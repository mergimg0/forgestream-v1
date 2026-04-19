"""Orchestrator tests -- verify the event lifecycle and component wiring."""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import (
    EventBus,
    Orchestrator,
    StructuralValidator,
)


class TestStructuralValidator:
    def test_valid_claim_passes(self):
        validator = StructuralValidator()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "test", "confidence": 0.8},
        )
        result = validator.validate(event)
        assert result.valid is True

    def test_missing_author_fails(self):
        validator = StructuralValidator()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="",
            evaluator=0.0,
            payload={"text": "test"},
        )
        result = validator.validate(event)
        assert result.valid is False

    def test_verified_finding_without_sources_downgrades(self):
        validator = StructuralValidator()
        event = Event(
            event_type=EventType.VERIFIED_FINDING,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="research",
            evaluator=0.0,
            payload={"finding": "something", "sources": []},
        )
        result = validator.validate(event)
        # Should downgrade to claim, not reject
        assert result.valid is True
        assert result.downgrade_to == EventType.CLAIM

    def test_verified_finding_with_sources_passes(self):
        validator = StructuralValidator()
        event = Event(
            event_type=EventType.VERIFIED_FINDING,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="research",
            evaluator=0.0,
            payload={"finding": "x", "sources": [{"url": "http://example.com"}]},
        )
        result = validator.validate(event)
        assert result.valid is True
        assert result.downgrade_to is None


class TestEventBus:
    async def test_publish_and_subscribe(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(handler)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="test",
            evaluator=0.5,
            payload={"text": "hello"},
        )
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].id == event.id

    async def test_multiple_subscribers(self):
        bus = EventBus()
        counts = {"a": 0, "b": 0}

        async def handler_a(event: Event):
            counts["a"] += 1

        async def handler_b(event: Event):
            counts["b"] += 1

        bus.subscribe(handler_a)
        bus.subscribe(handler_b)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="test",
            evaluator=0.5,
            payload={},
        )
        await bus.publish(event)

        assert counts["a"] == 1
        assert counts["b"] == 1

    async def test_unsubscribe(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(handler)
        bus.unsubscribe(handler)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="test",
            evaluator=0.5,
            payload={},
        )
        await bus.publish(event)

        assert len(received) == 0


class TestOrchestrator:
    def test_orchestrator_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        assert orch.session_id is not None
        assert orch.event_bus is not None
        assert orch.validator is not None

    async def test_process_event_through_pipeline(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)

        processed = []
        async def capture(event: Event):
            processed.append(event)

        orch.event_bus.subscribe(capture)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "test claim", "confidence": 0.8, "topic_keywords": ["test"]},
        )

        result = await orch.process_event(event)
        assert result is True
        assert len(processed) == 1

    async def test_invalid_event_rejected(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)

        processed = []
        async def capture(event: Event):
            processed.append(event)

        orch.event_bus.subscribe(capture)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="",  # invalid
            evaluator=0.0,
            payload={"text": "bad"},
        )

        result = await orch.process_event(event)
        assert result is False
        assert len(processed) == 0

    async def test_verified_finding_downgrade(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)

        processed = []
        async def capture(event: Event):
            processed.append(event)

        orch.event_bus.subscribe(capture)

        event = Event(
            event_type=EventType.VERIFIED_FINDING,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="research",
            evaluator=0.0,
            payload={"finding": "no sources", "sources": []},
        )

        result = await orch.process_event(event)
        assert result is True
        assert processed[0].event_type == EventType.CLAIM  # downgraded


class TestOrchestratorPersistence:
    async def test_process_event_writes_to_store(self, db_conn):
        """Events are persisted to PostgreSQL."""
        from forgestream.events.store import EventStore

        config = ForgeStreamConfig()
        store = EventStore(conn=db_conn)
        orch = Orchestrator(config, store=store)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "persisted claim", "topic_keywords": ["test"]},
        )
        result = await orch.process_event(event)
        assert result is True

        events = await store.get_events(session_id=orch.session_id)
        assert len(events) == 1
        assert events[0].payload["text"] == "persisted claim"

    async def test_process_event_syncs_to_firestore(self):
        """Events are synced to Firestore when enabled."""
        config = ForgeStreamConfig()
        mock_sync = MagicMock()
        orch = Orchestrator(config, firestore_sync=mock_sync)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "synced to firestore"},
        )
        await orch.process_event(event)

        mock_sync.sync_event.assert_called_once()
