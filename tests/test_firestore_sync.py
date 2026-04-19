from unittest.mock import MagicMock, patch
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.firestore_sync import FirestoreSync


class TestFirestoreSync:
    def test_init_disabled(self):
        sync = FirestoreSync(project_id="test", enabled=False)
        assert sync.enabled is False

    def test_event_to_firestore_doc(self):
        sync = FirestoreSync(project_id="test", enabled=False)
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test claim", "confidence": 0.85},
        )
        doc = sync._event_to_doc(event)
        assert doc["event_type"] == "claim"
        assert doc["author"] == "gemini"
        assert doc["payload"]["text"] == "test claim"
        assert isinstance(doc["id"], str)
        assert isinstance(doc["session_id"], str)

    def test_sync_disabled_is_noop(self):
        sync = FirestoreSync(project_id="test", enabled=False)
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test"},
        )
        sync.sync_event(event)  # should not raise

    def test_sync_enabled_writes_to_collection(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        sync = FirestoreSync(project_id="test", enabled=False)
        # Manually enable and inject mock db
        sync.enabled = True
        sync._db = mock_db

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "written to firestore"},
        )
        sync.sync_event(event)

        mock_db.collection.assert_called_with("events")
        mock_collection.document.assert_called_once()
