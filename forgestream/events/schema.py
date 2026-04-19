"""ECEF event schema -- the atomic unit of ForgeStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    """All event types in the ECEF event log."""

    CLAIM = "claim"
    CONTRADICTION = "contradiction"
    REQUIREMENT = "requirement"
    VERIFIED_FINDING = "verified_finding"
    ARTIFACT = "artifact"
    SUGGESTION = "suggestion"
    BRANCH_POINT = "branch_point"
    SEED = "seed"
    EVALUATOR_SNAPSHOT = "evaluator_snapshot"
    MODE_SWITCH = "mode_switch"
    MERGE = "merge"
    MEETING_SUMMARY = "meeting_summary"
    PROSODIC_FEATURE = "prosodic_feature"
    EMOTION_STATE = "emotion_state"
    ENTRAINMENT_SNAPSHOT = "entrainment_snapshot"
    RAPPORT_SCORE = "rapport_score"
    PROOF_OBLIGATION = "proof_obligation"


@dataclass
class Event:
    """An immutable event in the ECEF append-only log.

    Once written, events are never updated or deleted.
    """

    event_type: EventType
    session_id: UUID
    branch_id: UUID
    author: str
    evaluator: float
    payload: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    parent_id: UUID | None = None
    degradation_flag: bool = False
    trust_region_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "session_id": str(self.session_id),
            "branch_id": str(self.branch_id),
            "author": self.author,
            "evaluator": self.evaluator,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "degradation_flag": self.degradation_flag,
            "trust_region_ok": self.trust_region_ok,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Deserialize from a dict (e.g., from database row)."""
        return cls(
            id=UUID(d["id"]),
            event_type=EventType(d["event_type"]),
            session_id=UUID(d["session_id"]),
            branch_id=UUID(d["branch_id"]),
            author=d["author"],
            evaluator=d["evaluator"],
            payload=d["payload"],
            timestamp=(
                datetime.fromisoformat(d["timestamp"])
                if isinstance(d["timestamp"], str)
                else d["timestamp"]
            ),
            parent_id=UUID(d["parent_id"]) if d.get("parent_id") else None,
            degradation_flag=d.get("degradation_flag", False),
            trust_region_ok=d.get("trust_region_ok", True),
        )
