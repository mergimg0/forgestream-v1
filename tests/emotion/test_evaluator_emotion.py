"""Tests for the emotional_engagement evaluator extension."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.governor.evaluator import Evaluator, EvaluatorMetrics


def _make_event(event_type: EventType, payload: dict | None = None) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.0,
        payload=payload or {},
    )


class TestEmotionalEngagement:
    def test_default_weights_sum_to_one(self):
        total = sum(Evaluator.DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_default_weights_include_engagement(self):
        assert "engagement" in Evaluator.DEFAULT_WEIGHTS
        assert Evaluator.DEFAULT_WEIGHTS["engagement"] == 0.15

    def test_engagement_from_prosodic_events(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["test"], "confidence": 0.8}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.8, "f0_std": 40.0, "energy_rms": 0.12,
            }),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.7, "f0_std": 35.0, "energy_rms": 0.10,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        # With high arousal prosodic events, engagement should be above default
        assert metrics.emotional_engagement > 0.5

    def test_engagement_defaults_to_half_without_prosodic(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["test"], "confidence": 0.8}),
        ]
        metrics = evaluator.compute_metrics(events)
        assert metrics.emotional_engagement == 0.5

    def test_composite_includes_engagement(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["a", "b"], "confidence": 0.8}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.9, "f0_std": 60.0, "energy_rms": 0.15,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        # Composite should be influenced by engagement weight
        expected_engagement_contribution = (
            Evaluator.DEFAULT_WEIGHTS["engagement"] * metrics.emotional_engagement
        )
        assert metrics.composite > 0.0
        # Verify the weight is actually being applied
        manual_composite = (
            Evaluator.DEFAULT_WEIGHTS["knowledge"] * metrics.knowledge_density
            + Evaluator.DEFAULT_WEIGHTS["verification"] * metrics.verification_rate
            + Evaluator.DEFAULT_WEIGHTS["scaffold"] * metrics.scaffold_success
            + Evaluator.DEFAULT_WEIGHTS["uptake"] * metrics.suggestion_uptake
            + Evaluator.DEFAULT_WEIGHTS["engagement"] * metrics.emotional_engagement
        )
        assert metrics.composite == pytest.approx(
            max(0.0, min(1.0, manual_composite)), abs=0.001
        )

    def test_evaluator_metrics_has_emotional_engagement_field(self):
        m = EvaluatorMetrics(
            knowledge_density=0.5,
            verification_rate=0.3,
            scaffold_success=0.4,
            suggestion_uptake=0.5,
            emotional_engagement=0.7,
            composite=0.5,
        )
        assert m.emotional_engagement == 0.7

    def test_backward_compatible_with_4_weight_constructor(self):
        """Old 4-weight dicts should still work (engagement defaults)."""
        old_weights = {
            "knowledge": 0.3, "verification": 0.3,
            "scaffold": 0.25, "uptake": 0.15,
        }
        evaluator = Evaluator(weights=old_weights)
        events = [_make_event(EventType.CLAIM, {"topic_keywords": ["x"]})]
        # Should not crash — engagement weight defaults to 0.0 if missing
        score = evaluator.compute(events)
        assert 0.0 <= score <= 1.0
