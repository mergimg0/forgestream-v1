"""Tests for GRPO emotion tuning — auto_score engagement signal + tone tuner."""

from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.governor.tone_tuner import ToneAdjustmentTuner
from forgestream.post_meeting import PostMeetingSynthesis


def _make_event(event_type: EventType, payload: dict | None = None) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.0,
        payload=payload or {},
    )


class TestAutoScoreEngagement:
    def test_auto_score_boosted_by_high_arousal(self):
        config = ForgeStreamConfig()
        pms = PostMeetingSynthesis(config=config, data_dir="/tmp/test_grpo")
        events_no_emotion = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"], "confidence": 0.7}),
        ]
        events_with_emotion = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"], "confidence": 0.7}),
            _make_event(EventType.PROSODIC_FEATURE, {"arousal": 0.9}),
            _make_event(EventType.PROSODIC_FEATURE, {"arousal": 0.8}),
        ]
        score_no = pms.compute_auto_score(events_no_emotion)
        score_with = pms.compute_auto_score(events_with_emotion)
        assert score_with > score_no

    def test_auto_score_stays_in_range(self):
        config = ForgeStreamConfig()
        pms = PostMeetingSynthesis(config=config, data_dir="/tmp/test_grpo2")
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"]}),
            _make_event(EventType.PROSODIC_FEATURE, {"arousal": 1.0}),
        ]
        score = pms.compute_auto_score(events)
        assert 0.0 <= score <= 1.0


class TestToneAdjustmentTuner:
    def test_default_adjustments(self):
        tuner = ToneAdjustmentTuner()
        assert "hesitation_penalty" in tuner.DEFAULT_ADJUSTMENTS
        assert "excitement_boost" in tuner.DEFAULT_ADJUSTMENTS

    def test_generate_perturbations(self):
        tuner = ToneAdjustmentTuner()
        perturbed = tuner.generate_perturbations(
            tuner.DEFAULT_ADJUSTMENTS, n=5
        )
        assert len(perturbed) == 5
        for p in perturbed:
            # All values should be non-negative
            for v in p.values():
                assert v >= 0.0

    def test_tune_returns_valid_adjustments(self):
        tuner = ToneAdjustmentTuner()
        events = [
            _make_event(EventType.CLAIM, {
                "confidence": 0.7, "tone_markers": ["hesitation"],
                "topic_keywords": ["x"],
            }),
            _make_event(EventType.PROSODIC_FEATURE, {
                "jitter_local": 0.04, "shimmer_local": 0.06,
                "hnr": 10.0, "arousal": 0.3,
            }),
        ]
        result = tuner.tune(tuner.DEFAULT_ADJUSTMENTS, events, human_score=0.6)
        assert "hesitation_penalty" in result
        assert "excitement_boost" in result
        for v in result.values():
            assert v >= 0.0
