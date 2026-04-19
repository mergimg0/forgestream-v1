from datetime import datetime, timezone
from uuid import UUID, uuid4

from forgestream.events.schema import Event, EventType


class TestEventType:
    def test_all_event_types_exist(self):
        expected = [
            "claim", "contradiction", "requirement", "verified_finding",
            "artifact", "suggestion", "branch_point", "seed",
            "evaluator_snapshot", "mode_switch", "merge", "meeting_summary",
        ]
        for t in expected:
            assert EventType(t) is not None

    def test_event_type_is_string(self):
        assert EventType.CLAIM == "claim"
        assert EventType.ARTIFACT == "artifact"


class TestEvent:
    def test_create_event_with_required_fields(self):
        session_id = uuid4()
        branch_id = uuid4()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=session_id,
            branch_id=branch_id,
            author="gemini",
            evaluator=0.5,
            payload={"text": "Expert said X", "confidence": 0.9},
        )
        assert isinstance(event.id, UUID)
        assert event.session_id == session_id
        assert event.branch_id == branch_id
        assert event.event_type == EventType.CLAIM
        assert event.author == "gemini"
        assert event.evaluator == 0.5
        assert event.payload["text"] == "Expert said X"
        assert isinstance(event.timestamp, datetime)
        assert event.parent_id is None
        assert event.degradation_flag is False
        assert event.trust_region_ok is True

    def test_create_event_with_parent(self):
        parent_id = uuid4()
        event = Event(
            event_type=EventType.CONTRADICTION,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.6,
            payload={"claim_a_id": str(uuid4()), "claim_b_id": str(uuid4())},
            parent_id=parent_id,
        )
        assert event.parent_id == parent_id

    def test_create_event_with_degradation(self):
        event = Event(
            event_type=EventType.EVALUATOR_SNAPSHOT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="governor",
            evaluator=0.3,
            payload={"E_micro": 0.3},
            degradation_flag=True,
            trust_region_ok=False,
        )
        assert event.degradation_flag is True
        assert event.trust_region_ok is False

    def test_event_to_dict(self):
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test"},
        )
        d = event.to_dict()
        assert d["event_type"] == "claim"
        assert d["author"] == "gemini"
        assert isinstance(d["id"], str)
        assert isinstance(d["session_id"], str)

    def test_event_from_dict(self):
        event_id = uuid4()
        session_id = uuid4()
        branch_id = uuid4()
        now = datetime.now(timezone.utc)
        d = {
            "id": str(event_id),
            "event_type": "claim",
            "session_id": str(session_id),
            "branch_id": str(branch_id),
            "author": "gemini",
            "evaluator": 0.7,
            "payload": {"text": "hello"},
            "timestamp": now.isoformat(),
            "parent_id": None,
            "degradation_flag": False,
            "trust_region_ok": True,
        }
        event = Event.from_dict(d)
        assert event.id == event_id
        assert event.event_type == EventType.CLAIM
        assert event.session_id == session_id
