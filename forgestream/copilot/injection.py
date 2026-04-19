"""Injection rules for the audio copilot."""

from __future__ import annotations

import time

from forgestream.synthesis.suggestions import Priority, Suggestion


class InjectionRules:
    """Controls when and how the audio copilot speaks.

    Conservative by design: interrupting a meeting is worse than
    a missed suggestion.
    """

    ALLOWED_PRIORITIES = {Priority.CRITICAL, Priority.STRATEGIC}

    def __init__(
        self,
        min_silence_seconds: float = 2.0,
        cooldown_seconds: float = 30.0,
        max_words: int = 10,
    ) -> None:
        self.min_silence_seconds = min_silence_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_words = max_words
        self._last_injection_time: float | None = None

    def should_inject(
        self,
        suggestion: Suggestion,
        silence_duration: float,
    ) -> bool:
        """Determine if a suggestion should be spoken aloud."""
        # Only high-priority suggestions
        if suggestion.category not in self.ALLOWED_PRIORITIES:
            return False

        # Must have enough silence
        if silence_duration < self.min_silence_seconds:
            return False

        # Respect cooldown
        if self._last_injection_time is not None:
            elapsed = time.monotonic() - self._last_injection_time
            if elapsed < self.cooldown_seconds:
                return False

        return True

    def record_injection(self) -> None:
        """Record that an injection was made (for cooldown tracking)."""
        self._last_injection_time = time.monotonic()

    def truncate(self, text: str) -> str:
        """Truncate text to max_words."""
        words = text.split()
        if len(words) <= self.max_words:
            return text
        return " ".join(words[: self.max_words])
