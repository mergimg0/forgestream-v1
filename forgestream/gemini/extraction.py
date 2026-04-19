"""Parse Gemini Live API output into ECEF claim events."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from forgestream.events.schema import Event, EventType


HESITATION_PENALTY = 0.15
BACKTRACK_PENALTY = 0.2
EMPHASIS_BOOST = 0.2
EXCITEMENT_BOOST = 0.15


class ClaimExtractor:
    """Transforms structured Gemini output into Event objects."""

    def __init__(self, session_id: UUID, branch_id: UUID) -> None:
        self.session_id = session_id
        self.branch_id = branch_id

    def parse_claim(self, gemini_output: dict[str, Any]) -> Event:
        """Parse a single Gemini output into a claim event."""
        text = gemini_output.get("text", "")
        confidence = gemini_output.get("confidence", 0.5)
        tone_markers = gemini_output.get("tone_markers", [])
        topic_keywords = gemini_output.get("topic_keywords", [])

        priority_boost = 0.0

        if "hesitation" in tone_markers:
            confidence -= HESITATION_PENALTY
        if "backtracking" in tone_markers:
            confidence -= BACKTRACK_PENALTY
        if "emphasis" in tone_markers:
            priority_boost += EMPHASIS_BOOST
        if "excitement" in tone_markers:
            priority_boost += EXCITEMENT_BOOST

        confidence = max(0.0, min(1.0, confidence))

        payload: dict[str, Any] = {
            "text": text,
            "speaker": gemini_output.get("speaker", "unknown"),
            "confidence": confidence,
            "tone_markers": tone_markers,
            "topic_keywords": topic_keywords,
            "is_requirement": gemini_output.get("is_requirement", False),
            "is_question": gemini_output.get("is_question", False),
        }

        if gemini_output.get("audio_timestamp"):
            payload["audio_timestamp"] = gemini_output["audio_timestamp"]

        if priority_boost > 0:
            payload["priority_boost"] = priority_boost

        return Event(
            event_type=EventType.CLAIM,
            session_id=self.session_id,
            branch_id=self.branch_id,
            author="gemini",
            evaluator=0.0,  # filled by governor before write
            payload=payload,
        )
