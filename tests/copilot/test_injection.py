"""Audio copilot injection rules tests."""

import time

from forgestream.copilot.injection import InjectionRules
from forgestream.synthesis.suggestions import Priority, Suggestion


class TestInjectionRules:
    def test_allows_critical_suggestion(self):
        rules = InjectionRules()
        suggestion = Suggestion(text="Contradiction detected", priority_score=0.95)
        assert rules.should_inject(suggestion, silence_duration=3.0) is True

    def test_blocks_low_priority(self):
        rules = InjectionRules()
        suggestion = Suggestion(text="Nice to know", priority_score=0.2)
        assert rules.should_inject(suggestion, silence_duration=3.0) is False

    def test_blocks_during_speech(self):
        rules = InjectionRules()
        suggestion = Suggestion(text="Critical", priority_score=0.95)
        assert rules.should_inject(suggestion, silence_duration=0.5) is False

    def test_cooldown_blocks_rapid_fire(self):
        rules = InjectionRules(cooldown_seconds=30)
        suggestion = Suggestion(text="Strategic", priority_score=0.8)
        # First injection allowed
        assert rules.should_inject(suggestion, silence_duration=3.0) is True
        rules.record_injection()
        # Second blocked by cooldown
        assert rules.should_inject(suggestion, silence_duration=3.0) is False

    def test_truncates_to_word_limit(self):
        rules = InjectionRules(max_words=10)
        text = "This is a very long suggestion that should be truncated to ten words maximum"
        truncated = rules.truncate(text)
        assert len(truncated.split()) <= 10

    def test_minimum_silence_threshold(self):
        rules = InjectionRules(min_silence_seconds=2.0)
        suggestion = Suggestion(text="Critical", priority_score=0.95)
        assert rules.should_inject(suggestion, silence_duration=1.5) is False
        assert rules.should_inject(suggestion, silence_duration=2.5) is True
