"""Tests for SenseVoiceClassifier.

funasr-dependent tests use pytest.importorskip so they skip automatically
when funasr is not installed. The parse and interface tests always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from forgestream.emotion.classifier import SenseVoiceClassifier


class TestSenseVoiceClassifier:
    def test_classify_returns_tag_and_confidence(self):
        pytest.importorskip("funasr")
        classifier = SenseVoiceClassifier()
        # Generate 2 seconds of 200Hz sine (simulates speech)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        signal = (10000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        tag, confidence = classifier.classify(signal)
        assert isinstance(tag, str)
        assert tag in (
            "angry", "disgusted", "fearful", "happy", "neutral",
            "other", "sad", "surprised", "unknown",
        )
        assert 0.0 <= confidence <= 1.0

    def test_classify_silence_returns_neutral(self):
        pytest.importorskip("funasr")
        classifier = SenseVoiceClassifier()
        signal = np.zeros(32000, dtype=np.int16)
        tag, confidence = classifier.classify(signal)
        assert tag in ("neutral", "unknown", "other")

    def test_available_without_funasr(self):
        """SenseVoiceClassifier.is_available() interface always exists."""
        assert hasattr(SenseVoiceClassifier, "is_available")
        result = SenseVoiceClassifier.is_available()
        assert isinstance(result, bool)

    def test_parse_emotion_happy(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|HAPPY|> hello world")
        assert tag == "happy"
        assert conf == 0.8

    def test_parse_emotion_neutral(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|NEUTRAL|>")
        assert tag == "neutral"
        assert conf == 0.8

    def test_parse_emotion_angry(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("some text <|ANGRY|> more")
        assert tag == "angry"
        assert conf == 0.8

    def test_parse_emotion_sad(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|SAD|>")
        assert tag == "sad"

    def test_parse_emotion_fearful(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|FEARFUL|>")
        assert tag == "fearful"

    def test_parse_emotion_disgusted(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|DISGUSTED|>")
        assert tag == "disgusted"

    def test_parse_emotion_surprised(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|SURPRISED|>")
        assert tag == "surprised"

    def test_parse_emotion_other(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("<|OTHER|>")
        assert tag == "other"

    def test_parse_emotion_no_tag_falls_back_to_neutral(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("no tags here at all")
        assert tag == "neutral"
        assert conf == 0.5

    def test_parse_emotion_empty_string_falls_back(self):
        tag, conf = SenseVoiceClassifier._parse_emotion("")
        assert tag == "neutral"
        assert conf == 0.5

    def test_parse_emotion_case_insensitive(self):
        # The implementation calls text.upper() so lowercase input should work
        tag, conf = SenseVoiceClassifier._parse_emotion("<|happy|>")
        assert tag == "happy"
