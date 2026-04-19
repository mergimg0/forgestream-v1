"""ProofObligationDetector — detects formalizable claims and emits Lean 4 obligations.

Subscribes to the EventBus. On each CLAIM event:
  1. Checks if the claim is formalizable (pattern + keyword + confidence).
  2. If yes: generates a Lean 4 stub and emits a PROOF_OBLIGATION event.
  3. Deduplication: skips if the same conclusion was already emitted.
  4. Export: save_obligations(data_dir) writes proof_obligations.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ..events.schema import Event, EventType
from .lean_stub import LeanStubGenerator

if TYPE_CHECKING:
    from ..orchestrator import EventBus


# ---------------------------------------------------------------------------
# Detection signals
# ---------------------------------------------------------------------------

FORMALIZABLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:for all|for every|for any)\s+\w+", re.IGNORECASE),
    re.compile(r"(?:if|when|whenever)\s+.+(?:then|,)\s+", re.IGNORECASE),
    re.compile(r"(?:bounded|bounded by|within|between)\s+", re.IGNORECASE),
    re.compile(r"(?:converges?|diverges?|monotone|increasing|decreasing)", re.IGNORECASE),
    re.compile(r"(?:at most|at least|exactly|no more than)\s+", re.IGNORECASE),
    re.compile(r"(?:implies|entails|guarantees|ensures)\s+", re.IGNORECASE),
    re.compile(r"(?:iff|if and only if)\s+", re.IGNORECASE),
    re.compile(r"(?:there exists|there is)\s+", re.IGNORECASE),
]

MATH_KEYWORDS: frozenset[str] = frozenset({
    "theorem", "lemma", "proof", "axiom", "convergence", "bound",
    "epsilon", "delta", "sigma", "monotone", "continuous", "finite",
    "infinite", "set", "function", "mapping", "injective", "surjective",
    "bijective", "isomorphism", "homomorphism",
    # Additional terms that commonly appear in formalizable claims
    "converge", "converges", "diverge", "diverges", "bounded", "increasing",
    "decreasing",
})

_CONFIDENCE_THRESHOLD = 0.7


class ProofObligationDetector:
    """Subscribes to EventBus and detects formalizable claims.

    Emits PROOF_OBLIGATION events for claims that match the pattern + keyword
    criteria and have sufficient confidence.
    """

    def __init__(self, event_bus: "EventBus") -> None:
        self.event_bus = event_bus
        self._stub_gen = LeanStubGenerator()
        # Dedup: track (normalised) conclusions already emitted
        self._seen_conclusions: set[str] = set()
        # All obligations emitted this session
        self._obligations: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Register on_event with the EventBus."""
        self.event_bus.subscribe(self.on_event)

    def is_formalizable(self, event: Event) -> bool:
        """Return True if a CLAIM event is formalizable.

        Criteria:
        - event_type is CLAIM
        - payload.is_question is False (or absent)
        - payload.confidence >= 0.7
        - text matches >= 1 FORMALIZABLE_PATTERN
        - text contains >= 1 MATH_KEYWORD
        """
        if event.event_type != EventType.CLAIM:
            return False

        payload = event.payload
        if payload.get("is_question", False):
            return False

        confidence = payload.get("confidence", 0.0)
        if confidence < _CONFIDENCE_THRESHOLD:
            return False

        text: str = payload.get("text", "")
        if not text:
            return False

        # At least one pattern must match
        pattern_match = any(p.search(text) for p in FORMALIZABLE_PATTERNS)
        if not pattern_match:
            return False

        # At least one math keyword must be present (case-insensitive)
        text_lower = text.lower()
        keyword_match = any(kw in text_lower for kw in MATH_KEYWORDS)
        return keyword_match

    async def on_event(self, event: Event) -> None:
        """Handle an event from the EventBus."""
        if not self.is_formalizable(event):
            return

        payload = event.payload
        claim_text: str = payload.get("text", "")

        # Extract components from the claim
        components = self._stub_gen.extract_components(claim_text)
        variables: list[str] = components.get("variables", [])
        hypotheses: list[str] = components.get("hypotheses", [])
        conclusion: str = components.get("conclusion", claim_text)

        # Dedup by normalised conclusion
        norm_conclusion = conclusion.lower().strip()
        if norm_conclusion in self._seen_conclusions:
            return
        self._seen_conclusions.add(norm_conclusion)

        # Generate Lean 4 stub
        lean4_stub = self._stub_gen.generate_stub(
            claim_text=claim_text,
            variables=variables,
            hypotheses=hypotheses,
            conclusion=conclusion,
        )

        # Build obligation dict
        claim_id = str(event.id)
        obligation = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "speaker": payload.get("speaker", "unknown"),
            "confidence": payload.get("confidence", 0.0),
            "lean4_stub": lean4_stub,
            "variables": variables,
            "hypotheses": hypotheses,
            "conclusion": conclusion,
            "status": "pending",
            "formalization_confidence": 0.6,
            "proofforge_task_id": None,
        }
        self._obligations.append(obligation)

        # Emit PROOF_OBLIGATION event on the bus
        obligation_event = Event(
            event_type=EventType.PROOF_OBLIGATION,
            session_id=event.session_id,
            branch_id=event.branch_id,
            author="proof_detector",
            evaluator=event.evaluator,
            payload=obligation,
        )
        await self.event_bus.publish(obligation_event)

    def save_obligations(self, data_dir: str) -> str:
        """Write proof_obligations.json to data_dir.

        Returns the path to the written file.
        """
        path = Path(data_dir) / "proof_obligations.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._obligations, indent=2))
        return str(path)
