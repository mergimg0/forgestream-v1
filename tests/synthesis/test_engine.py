"""SynthesisEngine tests -- continuous event processing loop."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.engine import SynthesisEngine


class TestSynthesisEngine:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        assert engine.orchestrator is orch

    async def test_processes_claim_event(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []
        original_process = orch.process_event

        async def capture_process(event: Event) -> bool:
            emitted.append(event)
            return await original_process(event)

        orch.process_event = capture_process

        claim = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "The system must handle 10k events per second",
                "is_requirement": False,
                "topic_keywords": ["throughput", "events"],
                "confidence": 0.9,
            },
        )

        await engine.on_event(claim)

        req_events = [e for e in emitted if e.event_type == EventType.REQUIREMENT]
        assert len(req_events) >= 1

    async def test_ignores_own_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []

        async def capture(event: Event) -> bool:
            emitted.append(event)
            return True

        orch.process_event = capture

        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis_engine",
            evaluator=0.5,
            payload={"description": "already processed"},
        )

        await engine.on_event(event)
        assert len(emitted) == 0

    async def test_ignores_non_claim_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []

        async def capture(event: Event) -> bool:
            emitted.append(event)
            return True

        orch.process_event = capture

        artifact = Event(
            event_type=EventType.ARTIFACT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="scaffold",
            evaluator=0.6,
            payload={"compiles": True},
        )

        await engine.on_event(artifact)
        assert len(emitted) == 0

    def test_branch_tracker_shared(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        assert engine.branch_tracker is not None
        assert engine.branch_tracker.main_branch_id is not None

    async def test_seed_detection(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        for keywords in [["A", "B", "C"], ["X", "Y", "Z"]]:
            for _ in range(3):
                claim = Event(
                    event_type=EventType.CLAIM,
                    session_id=orch.session_id,
                    branch_id=uuid4(),
                    author="gemini",
                    evaluator=0.5,
                    payload={"text": "test", "topic_keywords": keywords, "confidence": 0.8},
                )
                engine._update_graph(claim)

        seeds = engine.detect_seeds()
        assert isinstance(seeds, list)
