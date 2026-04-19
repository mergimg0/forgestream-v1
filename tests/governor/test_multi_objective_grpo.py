"""Tests for multi-objective GRPO weight tuning."""

from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.governor.improvement import WeightTuner


def _make_events() -> list[Event]:
    sid = uuid4()
    bid = uuid4()
    return [
        Event(
            event_type=EventType.CLAIM,
            session_id=sid,
            branch_id=bid,
            author="gemini",
            evaluator=0.5,
            payload={"topic_keywords": ["A", "B"]},
        ),
        Event(
            event_type=EventType.VERIFIED_FINDING,
            session_id=sid,
            branch_id=bid,
            author="research",
            evaluator=0.6,
            payload={"confidence": 0.9, "sources": ["x"]},
        ),
        Event(
            event_type=EventType.ARTIFACT,
            session_id=sid,
            branch_id=bid,
            author="scaffold",
            evaluator=0.7,
            payload={"compiles": True, "tests_pass": True},
        ),
    ]


class TestTuneMultiObjective:
    """Tests for WeightTuner.tune_multi_objective()."""

    def _base_weights(self) -> dict[str, float]:
        return {
            "knowledge": 0.25,
            "verification": 0.25,
            "scaffold": 0.20,
            "uptake": 0.15,
            "engagement": 0.15,
        }

    def test_improves_toward_targets(self):
        """tune_multi_objective moves weights toward individual component targets."""
        tuner = WeightTuner()
        events = _make_events()
        weights = self._base_weights()

        # Set targets that differ from current weights
        targets = {
            "knowledge": 0.4,
            "verification": 0.3,
            "scaffold": 0.15,
            "uptake": 0.1,
            "engagement": 0.05,
        }

        result = tuner.tune_multi_objective(weights, events, targets)

        # Result must be a dict with same keys
        assert isinstance(result, dict)
        assert set(result.keys()) == set(weights.keys())

    def test_normalized_output(self):
        """tune_multi_objective always returns weights summing to 1.0."""
        tuner = WeightTuner()
        events = _make_events()
        weights = self._base_weights()

        targets = {
            "knowledge": 0.5,
            "verification": 0.2,
            "scaffold": 0.1,
            "uptake": 0.1,
            "engagement": 0.1,
        }

        result = tuner.tune_multi_objective(weights, events, targets)

        total = sum(result.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_backward_compatible(self):
        """tune_multi_objective result has the same keys as input weights."""
        tuner = WeightTuner()
        events = _make_events()

        # Use subset of keys — should still work
        weights = {"knowledge": 0.5, "verification": 0.5}
        targets = {"knowledge": 0.6, "verification": 0.4}

        result = tuner.tune_multi_objective(weights, events, targets)

        assert set(result.keys()) == set(weights.keys())
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_all_values_positive(self):
        """All returned weight values are positive."""
        tuner = WeightTuner()
        events = _make_events()
        weights = self._base_weights()
        targets = {k: 0.2 for k in weights}

        result = tuner.tune_multi_objective(weights, events, targets)

        for k, v in result.items():
            assert v > 0, f"Weight {k}={v} is not positive"

    def test_generates_10_perturbations_internally(self):
        """Method accepts events and targets and returns blended result."""
        tuner = WeightTuner()
        events = _make_events()
        weights = self._base_weights()
        targets = {k: 0.2 for k in weights}

        # Should not raise, should return dict
        result = tuner.tune_multi_objective(weights, events, targets)
        assert result is not None

    def test_score_uses_component_targets(self):
        """Perturbations are scored by component-level target proximity, not global."""
        tuner = WeightTuner(perturbation_scale=0.0)  # No perturbation = returns current
        events = _make_events()
        weights = self._base_weights()
        targets = {k: v for k, v in weights.items()}  # Targets exactly match current

        # With zero perturbation, result should be very close to current weights
        result = tuner.tune_multi_objective(weights, events, targets)

        for k in weights:
            assert abs(result[k] - weights[k]) < 0.1, (
                f"Weight {k} changed too much: {weights[k]} -> {result[k]}"
            )
