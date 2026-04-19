from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.governor.evaluator import Evaluator, EvaluatorMetrics


class TestEvaluator:
    def _make_events(self, types_and_payloads: list) -> list[Event]:
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

    def test_compute_returns_float_between_0_and_1(self):
        evaluator = Evaluator()
        events = self._make_events([
            (EventType.CLAIM, {"confidence": 0.8, "topic_keywords": ["A"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["x"]}),
        ])
        score = evaluator.compute(events)
        assert 0.0 <= score <= 1.0

    def test_more_verified_findings_increases_score(self):
        evaluator = Evaluator()
        events_few = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
            (EventType.CLAIM, {"topic_keywords": ["B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.8, "sources": ["x"]}),
        ])
        events_many = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
            (EventType.CLAIM, {"topic_keywords": ["B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.8, "sources": ["x"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["y"]}),
        ])
        score_few = evaluator.compute(events_few)
        score_many = evaluator.compute(events_many)
        assert score_many >= score_few

    def test_metrics_breakdown(self):
        evaluator = Evaluator()
        events = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A", "B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["x"]}),
            (EventType.ARTIFACT, {"compiles": True, "tests_pass": True}),
        ])
        metrics = evaluator.compute_metrics(events)
        assert isinstance(metrics, EvaluatorMetrics)
        assert metrics.knowledge_density >= 0.0
        assert metrics.verification_rate >= 0.0
        assert metrics.scaffold_success >= 0.0

    def test_custom_weights(self):
        evaluator = Evaluator(weights={"knowledge": 1.0, "verification": 0.0,
                                        "scaffold": 0.0, "uptake": 0.0})
        events = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
        ])
        score = evaluator.compute(events)
        assert score > 0.0

    def test_empty_events(self):
        evaluator = Evaluator()
        score = evaluator.compute([])
        # Default uptake=0.0 and engagement=0.5, so empty = 0.15*0.0 + 0.15*0.5 = 0.075
        assert 0.0 <= score <= 0.2
