"""ContradictionResolver — subscribes to EventBus, resolves CONTRADICTION events.

On each CONTRADICTION event the resolver:
1. Generates a resolution payload with both sides + probing questions
2. Emits a SUGGESTION event with priority "high" and category "contradiction_resolution"
"""

from __future__ import annotations

import logging
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

AUTHOR = "contradiction_resolver"


class ContradictionResolver:
    """Subscribes to the EventBus and handles CONTRADICTION events.

    For each CONTRADICTION event it emits a SUGGESTION with:
    - Both concepts (concept_a, concept_b)
    - An explanation
    - Suggested probing questions
    - priority >= 0.8 ("high")
    - category == "contradiction_resolution"
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def subscribe(self) -> None:
        """Register on_event with the orchestrator's EventBus."""
        self.orchestrator.event_bus.subscribe(self.on_event)

    async def on_event(self, event: Event) -> None:
        """EventBus handler — only processes CONTRADICTION events."""
        if event.author == AUTHOR:
            return
        if event.event_type != EventType.CONTRADICTION:
            return

        payload = event.payload
        concept_a = payload.get("concept_a", "")
        concept_b = payload.get("concept_b", "")
        explanation = payload.get("explanation", "")

        probing_questions = self._generate_probing_questions(concept_a, concept_b, explanation)

        suggestion = Event(
            event_type=EventType.SUGGESTION,
            session_id=event.session_id,
            branch_id=event.branch_id,
            author=AUTHOR,
            evaluator=0.0,
            payload={
                "text": (
                    f"Contradiction detected: '{concept_a}' vs '{concept_b}'. "
                    f"{explanation}"
                ),
                "priority": 0.9,
                "category": "contradiction_resolution",
                "concept_a": concept_a,
                "concept_b": concept_b,
                "explanation": explanation,
                "probing_questions": probing_questions,
            },
            parent_id=event.id,
        )
        await self.orchestrator.process_event(suggestion)
        logger.debug(
            "Emitted contradiction_resolution suggestion for '%s' vs '%s'",
            concept_a, concept_b,
        )

    @staticmethod
    def _generate_probing_questions(
        concept_a: str,
        concept_b: str,
        explanation: str,
    ) -> list[str]:
        """Generate probing questions to help resolve the contradiction."""
        return [
            f"In what contexts does '{concept_a}' apply versus '{concept_b}'?",
            f"Are '{concept_a}' and '{concept_b}' mutually exclusive or can they coexist?",
            f"What is the primary trade-off between '{concept_a}' and '{concept_b}' here?",
        ]
