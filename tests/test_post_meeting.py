"""PostMeetingSynthesis tests."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.post_meeting import PostMeetingSynthesis


class TestPostMeetingSynthesis:
    def _make_meeting_events(self) -> list[Event]:
        sid = uuid4()
        bid = uuid4()
        return [
            Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                  author="gemini", evaluator=0.4,
                  payload={"text": "Use Kafka", "topic_keywords": ["Kafka"], "confidence": 0.9}),
            Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                  author="gemini", evaluator=0.42,
                  payload={"text": "Sub-100ms latency", "topic_keywords": ["latency"],
                           "confidence": 0.85, "is_requirement": True}),
            Event(event_type=EventType.VERIFIED_FINDING, session_id=sid, branch_id=bid,
                  author="research", evaluator=0.5,
                  payload={"finding": "Kafka achieves 10ms p99", "confidence": 0.9,
                           "sources": [{"url": "https://kafka.apache.org"}]}),
            Event(event_type=EventType.ARTIFACT, session_id=sid, branch_id=bid,
                  author="scaffold", evaluator=0.55,
                  payload={"compiles": True, "tests_pass": True,
                           "files_created": ["pipeline.py"]}),
        ]

    def test_generate_report(self):
        config = ForgeStreamConfig()
        synthesis = PostMeetingSynthesis(config)
        events = self._make_meeting_events()

        report = synthesis.generate_report(events, meeting_name="Test Meeting")
        assert "Test Meeting" in report
        assert "claims" in report.lower()

    def test_save_report(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.meetings_dir = tmpdir
            synthesis = PostMeetingSynthesis(config)
            events = self._make_meeting_events()

            path = synthesis.save_report(events, meeting_name="test-meeting")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "test-meeting" in content.lower() or "Test" in content

    def test_tune_weights(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)
            events = self._make_meeting_events()

            new_weights = synthesis.tune_weights(events, human_score=0.8)

            assert set(new_weights.keys()) >= {"knowledge", "verification", "scaffold", "uptake"}
            assert abs(sum(new_weights.values()) - 1.0) < 0.01

    def test_save_and_load_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig()
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)

            weights = {"knowledge": 0.25, "verification": 0.35, "scaffold": 0.25, "uptake": 0.15}
            synthesis.save_weights(weights, meeting_count=1)

            loaded = synthesis.load_weights()
            assert loaded["knowledge"] == 0.25
            assert loaded["verification"] == 0.35

    def test_compute_auto_score(self):
        config = ForgeStreamConfig()
        synthesis = PostMeetingSynthesis(config)
        events = self._make_meeting_events()

        score = synthesis.compute_auto_score(events)
        assert 0.0 <= score <= 1.0
