"""Self-improvement mechanism tests."""

from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.governor.improvement import (
    PromptEvolution,
    WeightTuner,
    MeetingSynthesizer,
)


class TestWeightTuner:
    def _make_events(self) -> list[Event]:
        sid = uuid4()
        bid = uuid4()
        return [
            Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                  author="gemini", evaluator=0.5,
                  payload={"topic_keywords": ["A", "B"]}),
            Event(event_type=EventType.VERIFIED_FINDING, session_id=sid, branch_id=bid,
                  author="research", evaluator=0.6,
                  payload={"confidence": 0.9, "sources": ["x"]}),
            Event(event_type=EventType.ARTIFACT, session_id=sid, branch_id=bid,
                  author="scaffold", evaluator=0.7,
                  payload={"compiles": True, "tests_pass": True}),
        ]

    def test_generate_perturbations(self):
        tuner = WeightTuner()
        weights = {"knowledge": 0.3, "verification": 0.3, "scaffold": 0.25, "uptake": 0.15}
        perturbations = tuner.generate_perturbations(weights, n=5)
        assert len(perturbations) == 5
        for p in perturbations:
            assert set(p.keys()) == set(weights.keys())
            # Weights should sum to approximately 1.0
            assert abs(sum(p.values()) - 1.0) < 0.01

    def test_tune_returns_updated_weights(self):
        tuner = WeightTuner()
        events = self._make_events()
        initial = {"knowledge": 0.3, "verification": 0.3, "scaffold": 0.25, "uptake": 0.15}
        updated = tuner.tune(initial, events, human_score=0.8)
        assert set(updated.keys()) == set(initial.keys())
        assert abs(sum(updated.values()) - 1.0) < 0.01


class TestPromptEvolution:
    def test_score_prompt(self):
        evo = PromptEvolution()
        score = evo.score_prompt(
            prompt="Research Kafka best practices",
            output_useful=True,
            human_action="used",
        )
        assert 0.0 <= score <= 1.0

    def test_used_scores_higher_than_discarded(self):
        evo = PromptEvolution()
        used = evo.score_prompt("p", output_useful=True, human_action="used")
        discarded = evo.score_prompt("p", output_useful=False, human_action="discarded")
        assert used > discarded


class TestMeetingSynthesizer:
    def test_generate_summary(self):
        synth = MeetingSynthesizer()
        events = [
            Event(event_type=EventType.CLAIM, session_id=uuid4(), branch_id=uuid4(),
                  author="gemini", evaluator=0.5,
                  payload={"text": "claim 1", "topic_keywords": ["A"]}),
            Event(event_type=EventType.REQUIREMENT, session_id=uuid4(), branch_id=uuid4(),
                  author="synthesis", evaluator=0.6,
                  payload={"description": "Build X", "domain": "eng"}),
        ]
        summary = synth.generate_summary(events)
        assert "claims" in summary.lower() or "claim" in summary.lower()
        assert isinstance(summary, str)
