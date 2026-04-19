from uuid import uuid4

from forgestream.events.schema import EventType
from forgestream.gemini.extraction import ClaimExtractor


class TestClaimExtractor:
    def test_parse_structured_claim(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        gemini_output = {
            "text": "We need sub-100ms latency for ingestion",
            "speaker": "Expert A",
            "confidence": 0.85,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["latency", "ingestion"],
            "is_requirement": True,
            "is_question": False,
        }
        event = extractor.parse_claim(gemini_output)
        assert event.event_type == EventType.CLAIM
        assert event.payload["confidence"] == 0.85
        assert "latency" in event.payload["topic_keywords"]
        assert event.author == "gemini"

    def test_parse_adjusts_confidence_for_hesitation(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        gemini_output = {
            "text": "Maybe we should use Kafka",
            "speaker": "Expert A",
            "confidence": 0.8,
            "tone_markers": ["hesitation"],
            "topic_keywords": ["Kafka"],
            "is_requirement": False,
            "is_question": False,
        }
        event = extractor.parse_claim(gemini_output)
        assert event.payload["confidence"] < 0.8

    def test_parse_boosts_priority_for_emphasis(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        output = {
            "text": "This is CRITICAL",
            "speaker": "Expert",
            "confidence": 0.9,
            "tone_markers": ["emphasis", "excitement"],
            "topic_keywords": ["critical_requirement"],
            "is_requirement": True,
            "is_question": False,
        }
        event = extractor.parse_claim(output)
        assert event.payload.get("priority_boost", 0) > 0

    def test_parse_handles_missing_fields(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        gemini_output = {"text": "Something interesting"}
        event = extractor.parse_claim(gemini_output)
        assert event.event_type == EventType.CLAIM
        assert event.payload["text"] == "Something interesting"
        assert event.payload["confidence"] == 0.5
