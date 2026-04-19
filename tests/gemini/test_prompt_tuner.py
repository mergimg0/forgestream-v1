"""Tests for PromptTuner — GRPO-style prompt parameter tuning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.gemini.prompt_tuner import PromptParams, PromptTuner


def _make_events(n_claims: int = 3) -> list[Event]:
    sid = uuid4()
    bid = uuid4()
    events = []
    for i in range(n_claims):
        events.append(
            Event(
                event_type=EventType.CLAIM,
                session_id=sid,
                branch_id=bid,
                author="gemini",
                evaluator=0.5 + i * 0.05,
                payload={"text": f"claim {i}", "topic_keywords": [f"kw{i}"]},
            )
        )
    events.append(
        Event(
            event_type=EventType.REQUIREMENT,
            session_id=sid,
            branch_id=bid,
            author="synthesis",
            evaluator=0.7,
            payload={"description": "Build X"},
        )
    )
    return events


class TestPromptParams:
    """Tests for the PromptParams dataclass."""

    def test_default_construction(self):
        params = PromptParams()
        assert isinstance(params.extraction_granularity, float)
        assert isinstance(params.tone_sensitivity, float)
        assert isinstance(params.context_injection_minutes, float)

    def test_to_dict(self):
        params = PromptParams(
            extraction_granularity=0.7,
            tone_sensitivity=0.5,
            context_injection_minutes=3.0,
        )
        d = params.to_dict()
        assert d["extraction_granularity"] == 0.7
        assert d["tone_sensitivity"] == 0.5
        assert d["context_injection_minutes"] == 3.0

    def test_from_dict_roundtrip(self):
        params = PromptParams(
            extraction_granularity=0.8,
            tone_sensitivity=0.3,
            context_injection_minutes=5.0,
        )
        restored = PromptParams.from_dict(params.to_dict())
        assert restored.extraction_granularity == params.extraction_granularity
        assert restored.tone_sensitivity == params.tone_sensitivity
        assert restored.context_injection_minutes == params.context_injection_minutes

    def test_save_and_load(self):
        params = PromptParams(
            extraction_granularity=0.6,
            tone_sensitivity=0.4,
            context_injection_minutes=2.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompt_params.json"
            params.save(str(path))

            assert path.exists()
            loaded = PromptParams.load(str(path))

        assert loaded.extraction_granularity == pytest.approx(0.6)
        assert loaded.tone_sensitivity == pytest.approx(0.4)
        assert loaded.context_injection_minutes == pytest.approx(2.5)

    def test_save_creates_valid_json(self):
        params = PromptParams()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "params.json"
            params.save(str(path))
            data = json.loads(path.read_text())
            assert "extraction_granularity" in data


class TestPromptTuner:
    """Tests for the PromptTuner class."""

    def test_apply_params_returns_string(self):
        tuner = PromptTuner()
        params = PromptParams()
        result = tuner.apply_params("Extract claims from speech.", params)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_params_includes_base_instruction(self):
        tuner = PromptTuner()
        base = "Extract claims from speech."
        params = PromptParams()
        result = tuner.apply_params(base, params)
        # The base instruction should be incorporated
        assert "claim" in result.lower() or "extract" in result.lower()

    def test_apply_params_granularity_high(self):
        tuner = PromptTuner()
        base = "Extract claims."
        params_high = PromptParams(extraction_granularity=0.9, tone_sensitivity=0.5, context_injection_minutes=0.0)
        params_low = PromptParams(extraction_granularity=0.1, tone_sensitivity=0.5, context_injection_minutes=0.0)
        high_result = tuner.apply_params(base, params_high)
        low_result = tuner.apply_params(base, params_low)
        # High and low granularity should produce different instructions
        assert high_result != low_result

    def test_tune_returns_prompt_params(self):
        tuner = PromptTuner()
        events = _make_events()
        current = PromptParams()
        result = tuner.tune(current, events, human_score=0.8)
        assert isinstance(result, PromptParams)

    def test_tune_returns_params_in_valid_range(self):
        tuner = PromptTuner()
        events = _make_events()
        current = PromptParams()
        result = tuner.tune(current, events, human_score=0.8)
        assert 0.0 <= result.extraction_granularity <= 1.0
        assert 0.0 <= result.tone_sensitivity <= 1.0
        assert result.context_injection_minutes >= 0.0

    def test_tune_with_empty_events(self):
        tuner = PromptTuner()
        current = PromptParams()
        # Should not raise even with no events
        result = tuner.tune(current, [], human_score=0.5)
        assert isinstance(result, PromptParams)

    def test_tune_perturbs_params(self):
        """tune() with non-zero perturbation should (usually) change params."""
        tuner = PromptTuner(perturbation_scale=0.2)
        events = _make_events()
        current = PromptParams(
            extraction_granularity=0.5,
            tone_sensitivity=0.5,
            context_injection_minutes=3.0,
        )
        # Run several times — at least one should differ from current
        results = [tuner.tune(current, events, human_score=0.7) for _ in range(5)]
        changed = any(
            r.extraction_granularity != current.extraction_granularity
            or r.tone_sensitivity != current.tone_sensitivity
            for r in results
        )
        assert changed, "tune() never changed params across 5 calls"
