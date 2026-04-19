"""Async PostgreSQL event store -- append-only by design."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from .schema import Event, EventType

if TYPE_CHECKING:
    import psycopg


class EventStore:
    """Append-only event store backed by PostgreSQL."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self.conn = conn

    async def append(self, event: Event) -> Event:
        """Append an event to the log. The only write operation."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """INSERT INTO events
                   (id, session_id, timestamp, event_type, parent_id,
                    branch_id, author, evaluator, payload,
                    degradation_flag, trust_region_ok)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(event.id),
                    str(event.session_id),
                    event.timestamp,
                    event.event_type.value,
                    str(event.parent_id) if event.parent_id else None,
                    str(event.branch_id),
                    event.author,
                    event.evaluator,
                    json.dumps(event.payload),
                    event.degradation_flag,
                    event.trust_region_ok,
                ),
            )
            # Notify subscribers
            notification = json.dumps({
                "id": str(event.id),
                "event_type": event.event_type.value,
                "session_id": str(event.session_id),
                "author": event.author,
            })
            channel = f"event_{event.event_type.value}"
            await cur.execute(
                "SELECT pg_notify(%s, %s)", (channel, notification)
            )
        return event

    async def get_events(
        self,
        session_id: UUID,
        event_type: EventType | None = None,
        branch_id: UUID | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Query events with optional filters. Read-only."""
        conditions = ["session_id = %s"]
        params: list = [str(session_id)]

        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(event_type.value)
        if branch_id is not None:
            conditions.append("branch_id = %s")
            params.append(str(branch_id))
        if after is not None:
            conditions.append("timestamp > %s")
            params.append(after)

        from psycopg import sql

        where = sql.SQL(" AND ").join(sql.SQL(c) for c in conditions)
        query = sql.SQL("SELECT * FROM events WHERE {} ORDER BY timestamp ASC").format(where)
        if limit is not None:
            query = sql.SQL("{} LIMIT {}").format(query, sql.Literal(limit))

        async with self.conn.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            desc = cur.description or ()
            cols = [d.name for d in desc]

        return [self._row_to_event(dict(zip(cols, row))) for row in rows]

    async def get_latest_evaluator(self, session_id: UUID) -> float | None:
        """Get the most recent evaluator value for a session."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT evaluator FROM events
                   WHERE session_id = %s
                   ORDER BY timestamp DESC LIMIT 1""",
                (str(session_id),),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def count_by_type(self, session_id: UUID) -> Counter[EventType]:
        """Count events by type for a session."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT event_type, COUNT(*) FROM events
                   WHERE session_id = %s GROUP BY event_type""",
                (str(session_id),),
            )
            rows = await cur.fetchall()
        return Counter({EventType(row[0]): row[1] for row in rows})

    @staticmethod
    def _row_to_event(row: dict) -> Event:
        """Convert a database row to an Event."""
        return Event(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            event_type=EventType(row["event_type"]),
            session_id=(
                row["session_id"]
                if isinstance(row["session_id"], UUID)
                else UUID(row["session_id"])
            ),
            branch_id=(
                row["branch_id"]
                if isinstance(row["branch_id"], UUID)
                else UUID(row["branch_id"])
            ),
            author=row["author"],
            evaluator=row["evaluator"],
            payload=(
                row["payload"]
                if isinstance(row["payload"], dict)
                else json.loads(row["payload"])
            ),
            timestamp=row["timestamp"],
            parent_id=(
                (
                    row["parent_id"]
                    if isinstance(row["parent_id"], UUID)
                    else UUID(row["parent_id"])
                )
                if row.get("parent_id")
                else None
            ),
            degradation_flag=row.get("degradation_flag", False),
            trust_region_ok=row.get("trust_region_ok", True),
        )
