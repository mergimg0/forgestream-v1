"""Tests for WeightSensitivityAnalyzer."""

from __future__ import annotations

from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.governor.sensitivity import WeightSensitivityAnalyzer


def make_events(types_and_payloads: list) -> list[Event]:
    """Helper: build a list of Event objects from (type, payload) pairs."""
    sid = uuid4()
    bid = uuid4()
    return [
        Event(
            event_type=t,
            session_id=sid,
            branch_id=bid,
            author="test",
            evaluator=0.0,
            payload=p,
        )
        for t, p in types_and_payloads
    ]


TYPICAL_EVENTS = [
    (EventType.CLAIM, {"topic_keywords": ["A", "B"]}),
    (EventType.CLAIM, {"topic_keywords": ["C"]}),
    (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["x"]}),
    (EventType.ARTIFACT, {"compiles": True, "tests_pass": True, "files_created": ["f.py"]}),
]

DEFAULT_WEIGHTS = {
    "knowledge": 0.25,
    "verification": 0.25,
    "scaffold": 0.20,
    "uptake": 0.15,
    "engagement": 0.15,
}


class TestWeightSensitivityAnalyzerBasic:
    def test_returns_expected_keys(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        assert "sensitivities" in result
        assert "most_impactful" in result

    def test_sensitivities_covers_all_weight_keys(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        assert set(result["sensitivities"].keys()) == set(DEFAULT_WEIGHTS.keys())

    def test_most_impactful_is_a_known_key(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        assert result["most_impactful"] in DEFAULT_WEIGHTS

    def test_each_entry_has_variance_mean_range(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        for key, stats in result["sensitivities"].items():
            assert "variance" in stats, f"Missing 'variance' for {key}"
            assert "mean" in stats, f"Missing 'mean' for {key}"
            assert "range" in stats, f"Missing 'range' for {key}"

    def test_variance_is_non_negative(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=10)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        for key, stats in result["sensitivities"].items():
            assert stats["variance"] >= 0.0, f"Variance for {key} is negative"

    def test_mean_is_between_0_and_1(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=10)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        for key, stats in result["sensitivities"].items():
            assert 0.0 <= stats["mean"] <= 1.0, f"Mean for {key} out of [0,1]"

    def test_range_is_non_negative(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=10)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        for key, stats in result["sensitivities"].items():
            assert stats["range"] >= 0.0, f"Range for {key} is negative"

    def test_sensitivities_sorted_by_variance_descending(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=20)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        variances = [v["variance"] for v in result["sensitivities"].values()]
        assert variances == sorted(variances, reverse=True)

    def test_most_impactful_matches_first_sorted_entry(self):
        analyzer = WeightSensitivityAnalyzer(n_perturbations=20)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, DEFAULT_WEIGHTS.copy())
        first_key = next(iter(result["sensitivities"]))
        assert result["most_impactful"] == first_key


class TestWeightSensitivityEdgeCases:
    def test_empty_events_returns_valid_structure(self):
        """With no events, E(pi) should still be computable (returns 0.0 or default)."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        result = analyzer.analyze([], DEFAULT_WEIGHTS.copy())
        assert "sensitivities" in result
        assert "most_impactful" in result
        assert set(result["sensitivities"].keys()) == set(DEFAULT_WEIGHTS.keys())

    def test_empty_events_finite_variance(self):
        """With no events, engagement and uptake fallbacks (0.5) still vary when
        weights are perturbed, so variance is non-negative but not necessarily zero."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=10)
        result = analyzer.analyze([], DEFAULT_WEIGHTS.copy())
        for key, stats in result["sensitivities"].items():
            assert stats["variance"] >= 0.0, (
                f"Variance for {key} must be non-negative with empty events"
            )
            assert 0.0 <= stats["mean"] <= 1.0, (
                f"Mean for {key} must be in [0,1] with empty events"
            )

    def test_single_weight_key(self):
        """Analyzer should handle a weight dict with a single key."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        result = analyzer.analyze(events, {"knowledge": 1.0})
        assert result["most_impactful"] == "knowledge"
        assert len(result["sensitivities"]) == 1

    def test_all_equal_weights(self):
        """Equal weights should still produce a valid ranked result."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=20)
        events = make_events(TYPICAL_EVENTS)
        equal_weights = {k: 0.2 for k in DEFAULT_WEIGHTS}
        result = analyzer.analyze(events, equal_weights)
        assert result["most_impactful"] in equal_weights
        assert set(result["sensitivities"].keys()) == set(equal_weights.keys())

    def test_one_dominant_weight(self):
        """A weight set to ~1.0 (others near zero) should still work."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=10)
        events = make_events(TYPICAL_EVENTS)
        # One weight dominates — renormalization ensures all perturbations are valid
        dominant = {"knowledge": 0.96, "verification": 0.01, "scaffold": 0.01,
                    "uptake": 0.01, "engagement": 0.01}
        result = analyzer.analyze(events, dominant)
        assert result["most_impactful"] in dominant
        for stats in result["sensitivities"].values():
            assert stats["variance"] >= 0.0

    def test_weights_are_not_mutated(self):
        """The original weights dict must not be modified by analyze()."""
        analyzer = WeightSensitivityAnalyzer(n_perturbations=5)
        events = make_events(TYPICAL_EVENTS)
        original = DEFAULT_WEIGHTS.copy()
        weights_copy = DEFAULT_WEIGHTS.copy()
        analyzer.analyze(events, weights_copy)
        assert weights_copy == original


class TestWeightSensitivityImport:
    def test_importable_from_governor_package(self):
        from forgestream.governor import WeightSensitivityAnalyzer as WSA  # noqa: F401
        assert WSA is WeightSensitivityAnalyzer
