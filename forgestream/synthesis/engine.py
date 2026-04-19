"""SynthesisEngine -- continuous event processing loop.

Subscribes to the EventBus. For each claim event:
- Detects requirements
- Detects contradictions
- Tracks branches
- Periodically detects seeds
Emits derived events back through the Orchestrator.
Ignores self-authored events to prevent infinite loops.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.model import KnowledgeGraph
from forgestream.orchestrator import Orchestrator

from .branches import BranchTracker
from .contradictions import ContradictionDetector
from .requirements import RequirementDetector
from .seeds import SeedDetector
from .suggestions import Suggestion, SuggestionQueue

AUTHOR = "synthesis_engine"


class SynthesisEngine:
    """Continuous event processor -- the brain of ForgeStream.

    Subscribes to the orchestrator EventBus and processes claim events
    through requirement detection, contradiction detection, branch tracking,
    and seed detection. Emits derived events back through the Orchestrator.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.req_detector = RequirementDetector()
        self.branch_tracker = BranchTracker()
        self.seed_detector = SeedDetector(min_cluster_size=3)
        self.materializer = GraphMaterializer()
        self.suggestion_queue = SuggestionQueue()

        self._claim_events: list[Event] = []
        self._claim_count_at_last_seed_check = 0
        self._seed_check_interval = 10

    async def on_event(self, event: Event) -> None:
        """EventBus handler -- process incoming events."""
        # Ignore self-authored events to prevent infinite loops
        if event.author == AUTHOR:
            return

        # Only process claim events
        if event.event_type != EventType.CLAIM:
            return

        self._claim_events.append(event)

        # 1. Requirement detection
        req = self.req_detector.check(event)
        if req:
            req_event = Event(
                event_type=EventType.REQUIREMENT,
                session_id=event.session_id,
                branch_id=event.branch_id,
                author=AUTHOR,
                evaluator=0.0,
                payload=req,
                parent_id=event.id,
            )
            await self.orchestrator.process_event(req_event)

        # 2. Branch tracking
        keywords = event.payload.get("topic_keywords", [])
        if keywords:
            self.branch_tracker.add_keywords(
                self.branch_tracker.main_branch_id, keywords
            )
            drift = self.branch_tracker.check_drift(
                self.branch_tracker.main_branch_id, keywords
            )
            if drift:
                branch_event = Event(
                    event_type=EventType.BRANCH_POINT,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=drift,
                    parent_id=event.id,
                )
                await self.orchestrator.process_event(branch_event)

        # 3. Contradiction detection
        graph = self._build_graph()
        contradiction_detector = ContradictionDetector(graph=graph)
        for kw in keywords:
            contradiction = contradiction_detector.check(
                concept_name=kw, keywords=keywords
            )
            if contradiction:
                contra_event = Event(
                    event_type=EventType.CONTRADICTION,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=contradiction,
                    parent_id=event.id,
                )
                await self.orchestrator.process_event(contra_event)
                break  # one contradiction per claim

        # 4. Periodic seed detection
        if (len(self._claim_events) - self._claim_count_at_last_seed_check
                >= self._seed_check_interval):
            self._claim_count_at_last_seed_check = len(self._claim_events)
            seeds = self.detect_seeds()
            for seed_data in seeds:
                seed_event = Event(
                    event_type=EventType.SEED,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=seed_data,
                )
                await self.orchestrator.process_event(seed_event)

    def _build_graph(self) -> KnowledgeGraph:
        """Build knowledge graph from accumulated claim events."""
        return self.materializer.materialize(self._claim_events)

    def _update_graph(self, event: Event) -> None:
        """Add a claim event to the accumulator (for testing)."""
        self._claim_events.append(event)

    def detect_seeds(self) -> list[dict[str, Any]]:
        """Run seed detection on the current knowledge graph."""
        graph = self._build_graph()
        return self.seed_detector.detect(graph)
