"""Test RAPPORT_SCORE event type."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType


def test_rapport_score_event_type_exists():
    assert EventType.RAPPORT_SCORE == "rapport_score"


def test_rapport_score_event_serializes():
    event = Event(
        event_type=EventType.RAPPORT_SCORE,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="rapport_engine",
        evaluator=0.0,
        payload={
            "group_composite": 0.68,
            "group_trend": 0.03,
            "pair_scores": [],
            "disengaged_speakers": [],
        },
    )
    d = event.to_dict()
    assert d["event_type"] == "rapport_score"
    roundtrip = Event.from_dict(d)
    assert roundtrip.payload["group_composite"] == 0.68


def test_rapport_config_fields():
    config = ForgeStreamConfig()
    assert config.runpod_crqa_endpoint == ""
    assert config.runpod_timeout_seconds == 4.0
    assert config.rapport_damping_factor == 0.3
    assert config.rapport_enabled is True
