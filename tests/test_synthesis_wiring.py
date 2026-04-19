"""Test SynthesisEngine wiring to Orchestrator EventBus."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.engine import SynthesisEngine


class TestSynthesisWiring:
    async def test_engine_receives_events_from_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        orch.event_bus.subscribe(engine.on_event)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={
                "text": "The system must handle 10k events per second",
                "is_requirement": False,
                "topic_keywords": ["throughput", "events"],
                "confidence": 0.9,
            },
        )
        await orch.process_event(event)
        assert len(engine._claim_events) >= 1

    async def test_engine_emits_requirement_back_to_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        orch.event_bus.subscribe(engine.on_event)

        all_events = []

        async def capture(event: Event):
            all_events.append(event)

        orch.event_bus.subscribe(capture)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={
                "text": "We need a real-time dashboard for monitoring",
                "is_requirement": True,
                "topic_keywords": ["dashboard", "monitoring"],
                "confidence": 0.85,
            },
        )
        await orch.process_event(event)

        req_events = [e for e in all_events if e.event_type == EventType.REQUIREMENT]
        assert len(req_events) >= 1
        assert req_events[0].author == "synthesis_engine"

    async def test_engine_ignores_own_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        orch.event_bus.subscribe(engine.on_event)

        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis_engine",
            evaluator=0.5,
            payload={"description": "already processed"},
        )
        await orch.process_event(event)
        assert len(engine._claim_events) == 0

    def test_attach_synthesis_engine(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = orch.attach_synthesis_engine()
        assert engine is not None
        assert engine.orchestrator is orch
        assert len(orch.event_bus._subscribers) >= 1
