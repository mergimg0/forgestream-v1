"""Tests for ProofObligationDetector — formalizability detection and stub emission."""

from __future__ import annotations

import json
import os
import tempfile
from uuid import uuid4

import pytest
import asyncio

from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import EventBus
from forgestream.synthesis.proof_obligations import ProofObligationDetector


def _make_claim(text: str, confidence: float = 0.8, is_question: bool = False) -> Event:
    """Helper: create a CLAIM event with the given text."""
    return Event(
        event_type=EventType.CLAIM,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="speaker_a",
        evaluator=0.5,
        payload={
            "text": text,
            "confidence": confidence,
            "is_question": is_question,
            "speaker": "Speaker A",
            "topic_keywords": [],
            "tone_markers": [],
        },
    )


class TestFormalizabilityDetection:
    def setup_method(self):
        self.bus = EventBus()
        self.detector = ProofObligationDetector(event_bus=self.bus)

    def test_detects_formalizable_for_all(self):
        """'for all X' + math keyword → formalizable."""
        claim = _make_claim("for all evaluators satisfying the axioms, convergence holds", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_detects_formalizable_if_then(self):
        """'if X then Y' + math keyword → formalizable."""
        claim = _make_claim("if the function is monotone, then it converges", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_detects_formalizable_bounded(self):
        """'bounded by' + math keyword → formalizable."""
        claim = _make_claim("the epsilon is bounded by 0.15 and forms a finite set", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_detects_formalizable_converges(self):
        """'converges' alone (also math keyword) → formalizable."""
        claim = _make_claim("the theorem shows the sequence converges monotonically", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_detects_formalizable_there_exists(self):
        """'there exists' + math keyword → formalizable."""
        claim = _make_claim("there exists a bound for the bijective mapping", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_detects_formalizable_iff(self):
        """'iff' + math keyword → formalizable."""
        claim = _make_claim("P iff Q where P is the monotone condition", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True

    def test_skips_non_formalizable_plain_statement(self):
        """Plain English statement with no math keywords → not formalizable."""
        claim = _make_claim("the meeting went well today and everyone agreed", confidence=0.9)
        assert self.detector.is_formalizable(claim) is False

    def test_skips_low_confidence(self):
        """Confidence < 0.7 → not formalizable even if pattern matches."""
        claim = _make_claim("for all evaluators, convergence holds", confidence=0.5)
        assert self.detector.is_formalizable(claim) is False

    def test_skips_question(self):
        """Questions (is_question=True) → not formalizable."""
        claim = _make_claim("for all x, does convergence hold?", confidence=0.8, is_question=True)
        assert self.detector.is_formalizable(claim) is False

    def test_skips_pattern_without_math_keyword(self):
        """Pattern match without any math keyword → not formalizable."""
        claim = _make_claim("if the weather is good, then we should go outside", confidence=0.9)
        assert self.detector.is_formalizable(claim) is False

    def test_detects_implies_keyword(self):
        """'implies' + math keyword → formalizable."""
        claim = _make_claim("A implies B where A is the convergence condition", confidence=0.8)
        assert self.detector.is_formalizable(claim) is True


class TestEventEmission:
    def setup_method(self):
        self.bus = EventBus()
        self.detector = ProofObligationDetector(event_bus=self.bus)
        self.emitted: list[Event] = []

        async def capture(event: Event) -> None:
            self.emitted.append(event)

        self.bus.subscribe(capture)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_emits_proof_obligation_for_formalizable(self):
        """A formalizable CLAIM → PROOF_OBLIGATION event emitted on bus."""
        claim = _make_claim("for all evaluators satisfying axioms, convergence holds", confidence=0.8)
        self._run(self.detector.on_event(claim))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 1

    def test_no_emission_for_non_formalizable(self):
        """A non-formalizable CLAIM → no PROOF_OBLIGATION emitted."""
        claim = _make_claim("the meeting went smoothly today", confidence=0.9)
        self._run(self.detector.on_event(claim))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 0

    def test_proof_obligation_payload_structure(self):
        """PROOF_OBLIGATION payload has required fields."""
        claim = _make_claim("for all monotone functions, convergence holds", confidence=0.8)
        self._run(self.detector.on_event(claim))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 1
        payload = proof_events[0].payload
        assert "claim_id" in payload
        assert "claim_text" in payload
        assert "lean4_stub" in payload
        assert "conclusion" in payload
        assert "status" in payload
        assert payload["status"] == "pending"

    def test_proof_obligation_has_lean_stub(self):
        """PROOF_OBLIGATION payload.lean4_stub contains 'sorry'."""
        claim = _make_claim("for all continuous functions, the bound holds", confidence=0.8)
        self._run(self.detector.on_event(claim))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        stub = proof_events[0].payload["lean4_stub"]
        assert "sorry" in stub
        assert "theorem" in stub

    def test_non_claim_event_ignored(self):
        """Non-CLAIM events (e.g. REQUIREMENT) are ignored."""
        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="system",
            evaluator=0.5,
            payload={"description": "for all functions, convergence theorem holds"},
        )
        self._run(self.detector.on_event(event))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 0


class TestDeduplication:
    def setup_method(self):
        self.bus = EventBus()
        self.detector = ProofObligationDetector(event_bus=self.bus)
        self.emitted: list[Event] = []

        async def capture(event: Event) -> None:
            self.emitted.append(event)

        self.bus.subscribe(capture)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_dedup_same_claim_twice(self):
        """Same claim text submitted twice → only one PROOF_OBLIGATION emitted."""
        text = "for all monotone evaluators, convergence holds as a theorem"
        claim1 = _make_claim(text, confidence=0.8)
        claim2 = _make_claim(text, confidence=0.9)
        self._run(self.detector.on_event(claim1))
        self._run(self.detector.on_event(claim2))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 1

    def test_dedup_different_claims_both_emitted(self):
        """Two different formalizable claims → two PROOF_OBLIGATION events."""
        claim1 = _make_claim("for all monotone evaluators, convergence holds", confidence=0.8)
        claim2 = _make_claim("there exists a finite bound for the continuous mapping", confidence=0.8)
        self._run(self.detector.on_event(claim1))
        self._run(self.detector.on_event(claim2))
        proof_events = [e for e in self.emitted if e.event_type == EventType.PROOF_OBLIGATION]
        assert len(proof_events) == 2


class TestExportJson:
    def setup_method(self):
        self.bus = EventBus()
        self.detector = ProofObligationDetector(event_bus=self.bus)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_exports_json(self):
        """save_obligations writes valid JSON with the obligation list."""
        claim = _make_claim("for all continuous functions f, convergence is a theorem", confidence=0.8)
        self._run(self.detector.on_event(claim))

        with tempfile.TemporaryDirectory() as tmpdir:
            self.detector.save_obligations(tmpdir)
            path = os.path.join(tmpdir, "proof_obligations.json")
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert isinstance(data, list)
            assert len(data) == 1
            assert "lean4_stub" in data[0]
            assert "claim_text" in data[0]

    def test_exports_empty_list_when_no_obligations(self):
        """save_obligations with no obligations writes empty JSON array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.detector.save_obligations(tmpdir)
            path = os.path.join(tmpdir, "proof_obligations.json")
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert data == []

    def test_exports_multiple_obligations(self):
        """save_obligations exports all queued obligations."""
        claim1 = _make_claim("for all monotone functions, convergence holds", confidence=0.8)
        claim2 = _make_claim("there exists a finite bound on the continuous mapping", confidence=0.8)
        self._run(self.detector.on_event(claim1))
        self._run(self.detector.on_event(claim2))

        with tempfile.TemporaryDirectory() as tmpdir:
            self.detector.save_obligations(tmpdir)
            path = os.path.join(tmpdir, "proof_obligations.json")
            data = json.loads(open(path).read())
            assert len(data) == 2
