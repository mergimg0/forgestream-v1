from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.requirements import RequirementDetector


class TestRequirementDetector:
    def test_detects_requirement_language(self):
        detector = RequirementDetector()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "We need a sub-100ms ingestion pipeline",
                "is_requirement": True,
                "topic_keywords": ["ingestion", "pipeline"],
                "confidence": 0.85,
            },
        )
        result = detector.check(event)
        assert result is not None
        assert "ingestion" in result["description"].lower() or "pipeline" in result["description"].lower()

    def test_skips_non_requirement(self):
        detector = RequirementDetector()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "The weather is nice today",
                "is_requirement": False,
                "topic_keywords": ["weather"],
                "confidence": 0.3,
            },
        )
        result = detector.check(event)
        assert result is None

    def test_detects_from_text_patterns(self):
        detector = RequirementDetector()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "The system must handle 10k events per second",
                "is_requirement": False,
                "topic_keywords": ["events", "throughput"],
                "confidence": 0.9,
            },
        )
        result = detector.check(event)
        assert result is not None
