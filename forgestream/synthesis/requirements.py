"""Requirement detection from claim events."""

from __future__ import annotations

import re
from typing import Any

from forgestream.events.schema import Event

REQUIREMENT_PATTERNS = [
    re.compile(r"\bwe need\b", re.IGNORECASE),
    re.compile(r"\bit should\b", re.IGNORECASE),
    re.compile(r"\bmust have\b", re.IGNORECASE),
    re.compile(r"\bthe system must\b", re.IGNORECASE),
    re.compile(r"\brequire[ds]?\b", re.IGNORECASE),
    re.compile(r"\bshould be able to\b", re.IGNORECASE),
    re.compile(r"\bneeds to\b", re.IGNORECASE),
    re.compile(r"\bhas to\b", re.IGNORECASE),
]


class RequirementDetector:
    """Detects actionable requirements from claim events."""

    def check(self, event: Event) -> dict[str, Any] | None:
        """Check if a claim event contains a requirement.

        Returns a requirement payload dict if detected, None otherwise.
        """
        payload = event.payload
        text = payload.get("text", "")

        # Gemini may have already flagged it
        if payload.get("is_requirement", False):
            return self._build_requirement(event)

        # Check text patterns
        for pattern in REQUIREMENT_PATTERNS:
            if pattern.search(text):
                return self._build_requirement(event)

        return None

    @staticmethod
    def _build_requirement(event: Event) -> dict[str, Any]:
        payload = event.payload
        keywords = payload.get("topic_keywords", [])
        domain = keywords[0] if keywords else ""

        return {
            "description": payload.get("text", ""),
            "domain": domain,
            "complexity_estimate": 0.5,
            "linked_claims": [str(event.id)],
        }
