"""Sentinel bridge — forwards ForgeStream ECEF events to the Sentinel daemon.

Subscribes to ForgeStream's EventBus and POSTs high-signal events to
Sentinel's /api/forgestream_event endpoint.  Low-frequency events only
(CLAIM, REQUIREMENT, SEED, CONTRADICTION, PROOF_OBLIGATION, RAPPORT_SCORE,
MEETING_SUMMARY, VERIFIED_FINDING, ARTIFACT, SUGGESTION, BRANCH_POINT)
— the high-frequency prosodic stream is excluded to stay within Sentinel's
backpressure limits.

On meeting end, posts E(π) and metadata to /api/forgestream_meeting_end
so Sentinel can attribute credit to modes that were active during prep.

Degrades silently if Sentinel daemon is not running.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .events.schema import Event, EventType

logger = logging.getLogger(__name__)

# Events worth forwarding — high signal, low frequency
_FORWARD_EVENT_TYPES = {
    EventType.CLAIM,
    EventType.REQUIREMENT,
    EventType.SEED,
    EventType.CONTRADICTION,
    EventType.PROOF_OBLIGATION,
    EventType.RAPPORT_SCORE,
    EventType.MEETING_SUMMARY,
    EventType.VERIFIED_FINDING,
    EventType.ARTIFACT,
    EventType.SUGGESTION,
    EventType.BRANCH_POINT,
}


class SentinelForwarder:
    """EventBus subscriber that forwards events to the Sentinel daemon."""

    def __init__(
        self,
        port: int | None = None,
        token: str | None = None,
        project_path: str | None = None,
    ) -> None:
        self._port = port or int(os.environ.get("SENTINEL_PORT", "9100"))
        self._token = token or self._read_token()
        self._project_path = project_path or os.getcwd()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._batch_id = str(uuid4())[:8]
        self._available: bool | None = None  # None = unchecked
        self._event_count = 0
        self._session: Any = None  # aiohttp.ClientSession, lazy

    @staticmethod
    def _read_token() -> str:
        token_path = Path.home() / ".claude" / "state" / "sentinel" / "daemon-token"
        try:
            return token_path.read_text().strip()
        except (FileNotFoundError, PermissionError):
            return ""

    async def _get_session(self) -> Any:
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=2),
                    headers={"Authorization": f"Bearer {self._token}"},
                )
            except ImportError:
                # aiohttp not available — use urllib fallback
                self._session = False
        return self._session

    async def _post(self, path: str, data: dict) -> bool:
        """POST JSON to Sentinel daemon.  Returns True on success."""
        session = await self._get_session()
        if session is False:
            # No aiohttp — use synchronous urllib as fallback
            return self._post_sync(path, data)
        try:
            async with session.post(f"{self._base_url}{path}", json=data) as resp:
                if resp.status == 200:
                    return True
                logger.debug("Sentinel %s returned %d", path, resp.status)
                return False
        except Exception:
            return False

    def _post_sync(self, path: str, data: dict) -> bool:
        """Synchronous fallback when aiohttp is unavailable."""
        import json
        import urllib.request
        url = f"{self._base_url}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _check_available(self) -> bool:
        """Check if Sentinel daemon is reachable (once)."""
        if self._available is not None:
            return self._available  # type: ignore[return-value]
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"{self._base_url}/health", timeout=1
            ) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        if self._available:
            logger.info("Sentinel bridge: connected to daemon on port %d", self._port)
        else:
            logger.debug("Sentinel bridge: daemon not reachable, forwarding disabled")
        return self._available or False

    async def on_event(self, event: Event) -> None:
        """EventBus handler — forwards eligible events to Sentinel."""
        if event.event_type not in _FORWARD_EVENT_TYPES:
            return

        if not await self._check_available():
            return

        # ot-fs-002: Preserve speaker_id, meeting_id, segment_id in payload
        # so Sentinel modes can correlate events across speakers and segments.
        # Explicit keys AFTER **event.payload so they take precedence
        # (fixes key-shadowing bug: payload keys like "author" no longer override).
        payload = {
            **event.payload,
            "forgestream_type": event.event_type.value,
            "author": event.author,
            "evaluator": event.evaluator,
            "branch_id": str(event.branch_id),
            "meeting_id": str(event.session_id),
            "speaker_id": event.payload.get("speaker", event.payload.get("speaker_id", "")),
            "segment_id": str(event.id),
        }

        sentinel_event = {
            "event_id": str(event.id),
            "event_type": f"forgestream_{event.event_type.value}",
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "source_system": "forgestream",
            "event_batch_id": self._batch_id,
            "payload": payload,
        }

        success = await self._post(
            "/api/forgestream_event",
            {
                "event": sentinel_event,
                "project_path": self._project_path,
            },
        )
        if success:
            self._event_count += 1

    async def on_meeting_start(self, meeting_name: str = "", session_id: str = "") -> None:
        """Post meeting start to Sentinel — stamps attribution window."""
        if not await self._check_available():
            return

        await self._post(
            "/api/forgestream_meeting_start",
            {
                "project_path": self._project_path,
                "meeting_name": meeting_name,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Sentinel bridge: meeting start posted (%s)", meeting_name)

    async def on_meeting_end(self, result: dict, meeting_name: str = "") -> None:
        """Post meeting results to Sentinel for cross-system scoring."""
        if not await self._check_available():
            return

        await self._post(
            "/api/forgestream_meeting_end",
            {
                "project_path": self._project_path,
                "meeting_name": meeting_name,
                "e_meso": result.get("e_meso", 0),
                "meeting_count": result.get("meeting_count", 0),
                "weights": result.get("weights", {}),
                "rapport_weights": result.get("rapport_weights", {}),
                "claim_count": result.get("raw_claim_count", 0),
                "consensus_claim_count": result.get("consensus_claim_count", 0),
                "bottleneck_count": result.get("bottleneck_count", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_id": self._batch_id,
            },
        )
        logger.info(
            "Sentinel bridge: meeting end posted (E(π)=%.3f, %d events forwarded)",
            result.get("e_meso", 0),
            self._event_count,
        )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and self._session is not False:
            await self._session.close()
            self._session = None

    def subscribe(self, event_bus: Any) -> None:
        """Subscribe to a ForgeStream EventBus."""
        event_bus.subscribe(self.on_event)
