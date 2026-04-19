"""Firestore sync -- async dual-write for cloud access and dashboard.

Events are written to Firestore in the background after PostgreSQL write.
Firestore is the cloud sync layer; PostgreSQL stays the local source of truth.
"""

from __future__ import annotations

import logging
from typing import Any

from .events.schema import Event

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    firebase_admin = None  # type: ignore
    firestore = None  # type: ignore


class FirestoreSync:
    """Syncs events to Firestore for cloud access and real-time dashboard.

    - Append-only: documents are created, never updated or deleted
    - Background: sync failures don't block the main pipeline
    - Optional: disabled gracefully when firebase-admin isn't installed
    """

    def __init__(self, project_id: str, enabled: bool = True) -> None:
        self.project_id = project_id
        self.enabled = enabled and HAS_FIREBASE
        self._db = None
        self._app = None

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Firebase Admin SDK and Firestore client."""
        try:
            self._app = firebase_admin.initialize_app(
                credential=credentials.ApplicationDefault(),
                options={"projectId": self.project_id},
            )
            self._db = firestore.client()
            logger.info("Firestore sync initialized for project %s", self.project_id)
        except Exception as e:
            logger.warning("Firestore sync disabled: %s", e)
            self.enabled = False

    def sync_event(self, event: Event) -> None:
        """Sync an event to Firestore. Fire-and-forget."""
        if not self.enabled or self._db is None:
            return

        try:
            doc = self._event_to_doc(event)
            self._db.collection("events").document(str(event.id)).set(doc)
        except Exception as e:
            logger.warning("Firestore sync failed for event %s: %s", event.id, e)

    def sync_session(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Sync session metadata to Firestore."""
        if not self.enabled or self._db is None:
            return

        try:
            self._db.collection("sessions").document(session_id).set(metadata)
        except Exception as e:
            logger.warning("Firestore session sync failed: %s", e)

    @staticmethod
    def _event_to_doc(event: Event) -> dict[str, Any]:
        """Convert an Event to a Firestore-compatible document."""
        return {
            "id": str(event.id),
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "parent_id": str(event.parent_id) if event.parent_id else None,
            "branch_id": str(event.branch_id),
            "author": event.author,
            "evaluator": event.evaluator,
            "payload": event.payload,
            "degradation_flag": event.degradation_flag,
            "trust_region_ok": event.trust_region_ok,
        }
