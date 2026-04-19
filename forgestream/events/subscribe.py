"""Event subscription via PostgreSQL LISTEN/NOTIFY."""

from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

import psycopg
from psycopg import sql


class EventSubscriber:
    """Subscribes to event channels via PostgreSQL LISTEN/NOTIFY."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn: psycopg.AsyncConnection | None = None
        self.on_event: (
            Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None
        ) = None
        self._channels: list[str] = []

    async def start(self, channels: list[str]) -> None:
        """Connect and LISTEN on the given channels."""
        self._channels = channels
        self.conn = await psycopg.AsyncConnection.connect(
            self.dsn, autocommit=True
        )
        for channel in channels:
            await self.conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(f"event_{channel}")))

    async def poll(self) -> None:
        """Check for pending notifications and dispatch them."""
        if self.conn is None:
            return

        async for notify in self.conn.notifies():
            if self.on_event is not None:
                payload = json.loads(notify.payload) if notify.payload else {}
                event_type = payload.get(
                    "event_type",
                    notify.channel.removeprefix("event_"),
                )
                await self.on_event(event_type, payload)
            break  # Process available notifications then return

    async def poll_continuous(self) -> None:
        """Continuously poll for notifications. Blocking."""
        if self.conn is None:
            return

        async for notify in self.conn.notifies():
            if self.on_event is not None:
                payload = json.loads(notify.payload) if notify.payload else {}
                event_type = payload.get(
                    "event_type",
                    notify.channel.removeprefix("event_"),
                )
                await self.on_event(event_type, payload)

    async def stop(self) -> None:
        """Disconnect and stop listening."""
        if self.conn:
            for channel in self._channels:
                await self.conn.execute(sql.SQL("UNLISTEN {}").format(sql.Identifier(f"event_{channel}")))
            await self.conn.close()
            self.conn = None
