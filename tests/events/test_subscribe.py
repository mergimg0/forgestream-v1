import asyncio
import json
from uuid import uuid4

import psycopg
import pytest

from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore
from forgestream.events.subscribe import EventSubscriber

TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


class TestEventSubscriber:
    async def test_subscribe_receives_events(self):
        received = []

        async def on_event(event_type: str, payload: dict):
            received.append((event_type, payload))

        subscriber = EventSubscriber(TEST_DSN)
        await subscriber.start(channels=["claim"])
        subscriber.on_event = on_event

        # Write an event from a separate connection to trigger NOTIFY
        conn = await psycopg.AsyncConnection.connect(TEST_DSN)
        async with conn.cursor() as cur:
            event_data = json.dumps({
                "id": str(uuid4()),
                "event_type": "claim",
                "session_id": str(uuid4()),
            })
            await cur.execute(
                "SELECT pg_notify(%s, %s)", ("event_claim", event_data)
            )
        await conn.commit()
        await conn.close()

        # Give LISTEN/NOTIFY time to deliver
        await asyncio.sleep(0.5)
        await subscriber.poll()

        assert len(received) >= 1
        assert received[0][0] == "claim"

        await subscriber.stop()

    async def test_store_append_triggers_subscriber(self):
        """Integration test: EventStore.append -> NOTIFY -> EventSubscriber."""
        received = []

        async def on_event(event_type: str, payload: dict):
            received.append(event_type)

        subscriber = EventSubscriber(TEST_DSN)
        await subscriber.start(channels=["claim"])
        subscriber.on_event = on_event

        # Append via EventStore (separate connection)
        conn = await psycopg.AsyncConnection.connect(TEST_DSN)
        store = EventStore(conn=conn)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "integration test"},
        )
        await store.append(event)
        await conn.commit()

        await asyncio.sleep(0.5)
        await subscriber.poll()

        assert "claim" in received

        await subscriber.stop()
        await conn.close()
