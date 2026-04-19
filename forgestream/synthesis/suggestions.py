"""Priority-scored suggestion queue."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class Priority(str, Enum):
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    DELVE_DEEPER = "delve_deeper"
    GOOD_TO_PROBE = "good_to_probe"
    NICE_TO_KNOW = "nice_to_know"

    @classmethod
    def from_score(cls, score: float) -> Priority:
        if score >= 0.9:
            return cls.CRITICAL
        if score >= 0.7:
            return cls.STRATEGIC
        if score >= 0.5:
            return cls.DELVE_DEEPER
        if score >= 0.3:
            return cls.GOOD_TO_PROBE
        return cls.NICE_TO_KNOW


@dataclass
class Suggestion:
    text: str
    priority_score: float
    id: str = field(default_factory=lambda: str(uuid4()))
    category: Priority = field(default=Priority.NICE_TO_KNOW)
    linked_events: list[str] = field(default_factory=list)
    decay_rate: float = 0.02

    def __post_init__(self) -> None:
        self.category = Priority.from_score(self.priority_score)


class SuggestionQueue:
    """Max-priority queue for meeting suggestions."""

    def __init__(self) -> None:
        self._items: list[Suggestion] = []

    def add(self, suggestion: Suggestion) -> None:
        suggestion.category = Priority.from_score(suggestion.priority_score)
        self._items.append(suggestion)
        self._items.sort(key=lambda s: s.priority_score, reverse=True)

    def peek(self) -> Suggestion | None:
        return self._items[0] if self._items else None

    def dismiss(self) -> Suggestion | None:
        if self._items:
            return self._items.pop(0)
        return None

    def apply_decay(self, steps: int = 1) -> None:
        for s in self._items:
            s.priority_score = max(0.0, s.priority_score - s.decay_rate * steps)
            s.category = Priority.from_score(s.priority_score)
        self._items.sort(key=lambda s: s.priority_score, reverse=True)

    def get_by_priority(self, priority: Priority) -> list[Suggestion]:
        return [s for s in self._items if s.category == priority]

    def get_all(self) -> list[Suggestion]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
