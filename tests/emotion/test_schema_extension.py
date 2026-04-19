"""Test that new emotion event types exist and serialize correctly."""

from uuid import uuid4

from forgestream.events.schema import Event, EventType


def test_prosodic_feature_event_type_exists():
    assert EventType.PROSODIC_FEATURE == "prosodic_feature"


def test_emotion_state_event_type_exists():
    assert EventType.EMOTION_STATE == "emotion_state"


def test_entrainment_snapshot_event_type_exists():
    assert EventType.ENTRAINMENT_SNAPSHOT == "entrainment_snapshot"


def test_prosodic_feature_event_serializes():
    event = Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="emotion_extractor",
        evaluator=0.0,
        payload={
            "speaker_id": "unknown",
            "timestamp_ms": 1000,
            "chunk_index": 2,
            "arousal": 0.5,
            "valence": 0.5,
            "dominance": 0.5,
        },
    )
    d = event.to_dict()
    assert d["event_type"] == "prosodic_feature"
    assert d["payload"]["arousal"] == 0.5

    roundtrip = Event.from_dict(d)
    assert roundtrip.event_type == EventType.PROSODIC_FEATURE
    assert roundtrip.payload["speaker_id"] == "unknown"
