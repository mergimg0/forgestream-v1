"""Tests for ContradictionResolver — Task 5."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator


class TestContradictionResolver:
    def _make_event(self, orch: Orchestrator, event_type: EventType, payload: dict) -> Event:
        return Event(
            event_type=event_type,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="test",
            evaluator=0.5,
            payload=payload,
        )

    def test_module_importable(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver  # noqa: F401

    def test_instantiates(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        assert resolver is not None

    def test_subscribes_on_attach(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        before = len(orch.event_bus._subscribers)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()
        after = len(orch.event_bus._subscribers)
        assert after == before + 1

    def test_on_contradiction_emits_suggestion(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()

        emitted: list[Event] = []

        async def capture(event: Event) -> None:
            emitted.append(event)

        orch.event_bus.subscribe(capture)

        contra = self._make_event(orch, EventType.CONTRADICTION, {
            "concept_a": "sync",
            "concept_b": "async",
            "explanation": "conflicting concepts",
        })

        asyncio.get_event_loop().run_until_complete(orch.process_event(contra))

        suggestion_events = [e for e in emitted if e.event_type == EventType.SUGGESTION]
        assert len(suggestion_events) >= 1

    def test_suggestion_has_high_priority(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()

        emitted: list[Event] = []

        async def capture(event: Event) -> None:
            emitted.append(event)

        orch.event_bus.subscribe(capture)

        contra = self._make_event(orch, EventType.CONTRADICTION, {
            "concept_a": "blocking",
            "concept_b": "nonblocking",
            "explanation": "I/O model conflict",
        })

        asyncio.get_event_loop().run_until_complete(orch.process_event(contra))

        suggestions = [e for e in emitted if e.event_type == EventType.SUGGESTION]
        assert len(suggestions) >= 1
        # priority should be "high" or numeric >= 0.8
        p = suggestions[0].payload.get("priority")
        if isinstance(p, str):
            assert p == "high"
        else:
            assert p >= 0.8

    def test_suggestion_category_is_contradiction_resolution(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()

        emitted: list[Event] = []

        async def capture(event: Event) -> None:
            emitted.append(event)

        orch.event_bus.subscribe(capture)

        contra = self._make_event(orch, EventType.CONTRADICTION, {
            "concept_a": "stateful",
            "concept_b": "stateless",
            "explanation": "state model conflict",
        })

        asyncio.get_event_loop().run_until_complete(orch.process_event(contra))

        suggestions = [e for e in emitted if e.event_type == EventType.SUGGESTION]
        assert len(suggestions) >= 1
        assert suggestions[0].payload.get("category") == "contradiction_resolution"

    def test_suggestion_has_probing_questions(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()

        emitted: list[Event] = []

        async def capture(event: Event) -> None:
            emitted.append(event)

        orch.event_bus.subscribe(capture)

        contra = self._make_event(orch, EventType.CONTRADICTION, {
            "concept_a": "mutable",
            "concept_b": "immutable",
            "explanation": "data model conflict",
        })

        asyncio.get_event_loop().run_until_complete(orch.process_event(contra))

        suggestions = [e for e in emitted if e.event_type == EventType.SUGGESTION]
        assert len(suggestions) >= 1
        payload = suggestions[0].payload
        assert "probing_questions" in payload
        assert len(payload["probing_questions"]) >= 1

    def test_non_contradiction_events_ignored(self):
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = ContradictionResolver(orchestrator=orch)
        resolver.subscribe()

        emitted: list[Event] = []

        async def capture(event: Event) -> None:
            emitted.append(event)

        orch.event_bus.subscribe(capture)

        claim = self._make_event(orch, EventType.CLAIM, {
            "text": "PostgreSQL is reliable",
            "confidence": 0.9,
            "topic_keywords": ["db"],
        })

        asyncio.get_event_loop().run_until_complete(orch.process_event(claim))

        # No suggestion emitted from resolver for a plain claim
        resolver_suggestions = [
            e for e in emitted
            if e.event_type == EventType.SUGGESTION
            and e.payload.get("category") == "contradiction_resolution"
        ]
        assert len(resolver_suggestions) == 0

    def test_attach_contradiction_resolver_method_on_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        resolver = orch.attach_contradiction_resolver()
        assert resolver is not None
        # Should be subscribed
        from forgestream.synthesis.contradiction_resolver import ContradictionResolver
        assert isinstance(resolver, ContradictionResolver)


class TestAPIContradictions:
    def test_contradictions_endpoint_exists(self):
        from fastapi.testclient import TestClient
        from forgestream.dashboard.server import create_app
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/contradictions")
        assert response.status_code == 200

    def test_contradictions_endpoint_returns_list(self):
        from fastapi.testclient import TestClient
        from forgestream.dashboard.server import create_app
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/contradictions")
        data = response.json()
        assert "contradictions" in data
        assert isinstance(data["contradictions"], list)
