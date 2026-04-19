from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore

TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


class TestEventStore:
    @pytest.fixture
    async def store(self, db_conn):
        s = EventStore(conn=db_conn)
        yield s

    async def test_append_and_retrieve(self, store, session_id, branch_id):
        event = Event(
            event_type=EventType.CLAIM,
            session_id=session_id,
            branch_id=branch_id,
            author="gemini",
            evaluator=0.5,
            payload={"text": "Test claim", "confidence": 0.9},
        )
        stored = await store.append(event)
        assert stored.id == event.id

        events = await store.get_events(session_id=session_id)
        assert len(events) == 1
        assert events[0].id == event.id
        assert events[0].payload["text"] == "Test claim"

    async def test_append_multiple_events(self, store, session_id, branch_id):
        for i in range(5):
            event = Event(
                event_type=EventType.CLAIM,
                session_id=session_id,
                branch_id=branch_id,
                author="gemini",
                evaluator=0.5 + i * 0.05,
                payload={"text": f"Claim {i}"},
            )
            await store.append(event)

        events = await store.get_events(session_id=session_id)
        assert len(events) == 5
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    async def test_filter_by_event_type(self, store, session_id, branch_id):
        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_id, author="gemini", evaluator=0.5,
            payload={"text": "a claim"},
        ))
        await store.append(Event(
            event_type=EventType.SUGGESTION, session_id=session_id,
            branch_id=branch_id, author="synthesis", evaluator=0.6,
            payload={"text": "a suggestion", "priority": 0.8},
        ))

        claims = await store.get_events(
            session_id=session_id, event_type=EventType.CLAIM
        )
        assert len(claims) == 1
        assert claims[0].event_type == EventType.CLAIM

    async def test_filter_by_branch(self, store, session_id):
        branch_a = uuid4()
        branch_b = uuid4()

        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_a, author="gemini", evaluator=0.5,
            payload={"text": "branch a"},
        ))
        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_b, author="gemini", evaluator=0.5,
            payload={"text": "branch b"},
        ))

        events = await store.get_events(
            session_id=session_id, branch_id=branch_a
        )
        assert len(events) == 1
        assert events[0].payload["text"] == "branch a"

    async def test_get_latest_evaluator(self, store, session_id, branch_id):
        for val in [0.3, 0.5, 0.7]:
            await store.append(Event(
                event_type=EventType.CLAIM, session_id=session_id,
                branch_id=branch_id, author="gemini", evaluator=val,
                payload={"text": "x"},
            ))

        latest = await store.get_latest_evaluator(session_id)
        assert latest == 0.7

    async def test_count_by_type(self, store, session_id, branch_id):
        for _ in range(3):
            await store.append(Event(
                event_type=EventType.CLAIM, session_id=session_id,
                branch_id=branch_id, author="g", evaluator=0.5,
                payload={},
            ))
        await store.append(Event(
            event_type=EventType.ARTIFACT, session_id=session_id,
            branch_id=branch_id, author="scaffold", evaluator=0.6,
            payload={},
        ))

        counts = await store.count_by_type(session_id)
        assert counts[EventType.CLAIM] == 3
        assert counts[EventType.ARTIFACT] == 1
