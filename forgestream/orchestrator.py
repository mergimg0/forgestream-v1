"""ForgeStream Orchestrator -- the conductor that wires everything together.

Core process with in-memory event bus. Child workers coordinate through PostgreSQL.
Governor observes post-write, never blocks valid data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine
from uuid import UUID, uuid4

from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .events.store import EventStore
from .firestore_sync import FirestoreSync
from .governor.evaluator import Evaluator


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    downgrade_to: EventType | None = None


class StructuralValidator:
    """Pre-write structural validation. Trivially thin. Never rejects valid data."""

    def validate(self, event: Event) -> ValidationResult:
        """Check structural validity only.

        - Required fields present
        - Source chain for verified_findings
        - Never filters based on semantic content
        """
        if not event.author:
            return ValidationResult(valid=False, reason="missing author")

        if not event.payload and event.event_type != EventType.EVALUATOR_SNAPSHOT:
            return ValidationResult(valid=False, reason="missing payload")

        # Verified findings without sources get downgraded to claims
        if event.event_type == EventType.VERIFIED_FINDING:
            sources = event.payload.get("sources", [])
            if not sources:
                return ValidationResult(
                    valid=True,
                    downgrade_to=EventType.CLAIM,
                    reason="no sources provided, downgrading to claim",
                )

        return ValidationResult(valid=True)


class EventBus:
    """In-memory event fanout to local subscribers.

    Provides instant updates to TUI and other in-process consumers.
    """

    def __init__(self) -> None:
        self._subscribers: list[
            Callable[[Event], Coroutine[Any, Any, None]]
        ] = []

    def subscribe(
        self, handler: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        if handler not in self._subscribers:
            self._subscribers.append(handler)

    def unsubscribe(
        self, handler: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        self._subscribers = [h for h in self._subscribers if h is not handler]

    async def publish(self, event: Event) -> None:
        """Fanout event to all subscribers."""
        for handler in self._subscribers:
            await handler(event)


class Orchestrator:
    """The conductor. Manages the event lifecycle:

    1. Receive event (from worker or direct)
    2. Structural validation (pre-write, < 1ms)
    3. Write to PostgreSQL (append-only)
    4. Publish to in-memory event bus (instant TUI update)
    5. Governor post-write observation (async)
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        store: EventStore | None = None,
        firestore_sync: FirestoreSync | None = None,
    ) -> None:
        self.config = config
        self.session_id = uuid4()
        self.event_bus = EventBus()
        self.validator = StructuralValidator()
        self.evaluator = Evaluator()
        self.store = store
        self.firestore_sync = firestore_sync
        self._event_buffer: list[Event] = []

    async def process_event(self, event: Event) -> bool:
        """Process a single event through the full lifecycle.

        Returns True if the event was accepted, False if structurally invalid.
        """
        # Step 1: Structural validation
        result = self.validator.validate(event)
        if not result.valid:
            return False

        # Step 2: Apply downgrade if needed
        if result.downgrade_to is not None:
            event = Event(
                event_type=result.downgrade_to,
                session_id=event.session_id,
                branch_id=event.branch_id,
                author=event.author,
                evaluator=event.evaluator,
                payload=event.payload,
                id=event.id,
                timestamp=event.timestamp,
                parent_id=event.parent_id,
            )

        # Step 3: Compute evaluator value
        self._event_buffer.append(event)
        event.evaluator = self.evaluator.compute(self._event_buffer[-20:])

        # Step 4: Write to PostgreSQL
        if self.store is not None:
            await self.store.append(event)

        # Step 5: Sync to Firestore (fire-and-forget)
        if self.firestore_sync is not None:
            self.firestore_sync.sync_event(event)

        # Step 6: Publish to in-memory bus (instant TUI update)
        await self.event_bus.publish(event)

        return True

    def attach_synthesis_engine(self) -> "SynthesisEngine":
        """Create and attach a SynthesisEngine to this orchestrator's EventBus."""
        from .synthesis.engine import SynthesisEngine
        engine = SynthesisEngine(orchestrator=self)
        self.event_bus.subscribe(engine.on_event)
        return engine

    def attach_emotion_correlator(self) -> "EmotionCorrelator":
        """Create and attach an EmotionCorrelator to this orchestrator's EventBus."""
        from .emotion.correlator import EmotionCorrelator
        correlator = EmotionCorrelator(orchestrator=self)
        self.event_bus.subscribe(correlator.on_event)
        return correlator

    def attach_dynamics_engine(self) -> "GroupDynamicsEngine":
        """Create and attach a GroupDynamicsEngine to this orchestrator's EventBus."""
        from .emotion.dynamics import GroupDynamicsEngine
        engine = GroupDynamicsEngine(orchestrator=self)
        self.event_bus.subscribe(engine.on_event)
        return engine

    def attach_rapport_engine(
        self, meeting_count: int = 1, damping_factor: float = 0.3,
        runpod_endpoint: str = "", runpod_timeout: float = 4.0,
        rapport_weights: dict[str, float] | None = None,
    ) -> "RapportEngine":
        """Create and attach a RapportEngine to this orchestrator's EventBus."""
        from .emotion.rapport import RapportEngine
        engine = RapportEngine(
            orchestrator=self, meeting_count=meeting_count,
            damping_factor=damping_factor, runpod_endpoint=runpod_endpoint,
            runpod_timeout=runpod_timeout, rapport_weights=rapport_weights,
        )
        self.event_bus.subscribe(engine.on_event)
        return engine

    def attach_contradiction_resolver(self) -> "ContradictionResolver":
        """Create and attach a ContradictionResolver to this orchestrator's EventBus."""
        from .synthesis.contradiction_resolver import ContradictionResolver
        resolver = ContradictionResolver(orchestrator=self)
        resolver.subscribe()
        return resolver

    def attach_proof_detector(self) -> "ProofObligationDetector":
        """Create and attach a ProofObligationDetector to this orchestrator's EventBus."""
        from .synthesis.proof_obligations import ProofObligationDetector
        detector = ProofObligationDetector(event_bus=self.event_bus)
        detector.subscribe()
        return detector

    @property
    def event_count(self) -> int:
        return len(self._event_buffer)
