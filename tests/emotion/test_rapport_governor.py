"""Tests for rapport integration with SOS governor."""

from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.governor.axioms import AxiomChecker
from forgestream.governor.evaluator import Evaluator
from forgestream.governor.trust_region import TrustRegion


def _make_event(event_type, payload=None):
    return Event(
        event_type=event_type, session_id=uuid4(), branch_id=uuid4(),
        author="test", evaluator=0.0, payload=payload or {},
    )


class TestEvaluatorRapportEnrichment:
    def test_engagement_uses_rapport_when_available(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"]}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.6, "f0_std": 30.0, "energy_rms": 0.1,
            }),
            _make_event(EventType.RAPPORT_SCORE, {
                "group_composite": 0.85,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        assert metrics.emotional_engagement > 0.6

    def test_engagement_falls_back_without_rapport(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"]}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.6, "f0_std": 30.0, "energy_rms": 0.1,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        assert metrics.emotional_engagement > 0.0


class TestAxiomRapportAdvisory:
    def test_check_rapport_degradation(self):
        checker = AxiomChecker()
        trajectory = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = checker.check_rapport_trend(trajectory, disengaged=True)
        assert result.axiom == "rapport_advisory"
        assert result.holds is False

    def test_no_advisory_without_disengagement(self):
        checker = AxiomChecker()
        trajectory = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = checker.check_rapport_trend(trajectory, disengaged=False)
        assert result.holds is True

    def test_short_trajectory_holds(self):
        checker = AxiomChecker()
        result = checker.check_rapport_trend([0.5, 0.4], disengaged=True)
        assert result.holds is True


class TestTrustRegionRapportBoost:
    def test_rapport_trend_boosts_improvements(self):
        tr = TrustRegion()
        initial = tr._consecutive_improvements
        tr.record_meeting_result(
            e_macro_improved=True, axiom_violations=0, rapport_trend=0.2
        )
        assert tr._consecutive_improvements == initial + 1.5

    def test_no_boost_without_rapport_trend(self):
        tr = TrustRegion()
        initial = tr._consecutive_improvements
        tr.record_meeting_result(
            e_macro_improved=True, axiom_violations=0, rapport_trend=0.0
        )
        assert tr._consecutive_improvements == initial + 1

    def test_backward_compatible_without_rapport_trend(self):
        tr = TrustRegion()
        initial = tr._consecutive_improvements
        tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        assert tr._consecutive_improvements == initial + 1
