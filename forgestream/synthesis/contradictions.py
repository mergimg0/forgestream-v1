"""Contradiction detection between claims and verified concepts."""

from __future__ import annotations

from typing import Any

from forgestream.graph.model import KnowledgeGraph

ANTONYM_PREFIXES = [
    ("synchronous", "asynchronous"),
    ("sync", "async"),
    ("strong", "eventual"),
    ("mutable", "immutable"),
    ("stateful", "stateless"),
    ("blocking", "nonblocking"),
    ("centralized", "decentralized"),
]


class ContradictionDetector:
    """Detects contradictions between new concepts and existing verified concepts."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def check(
        self,
        concept_name: str,
        keywords: list[str],
    ) -> dict[str, Any] | None:
        """Check if a new concept contradicts existing verified concepts.

        Returns a contradiction payload dict if detected, None otherwise.
        """
        for existing in self.graph.concepts:
            if not existing.verified:
                continue

            # Check antonym prefix patterns
            if self._is_antonym_pair(concept_name, existing.name):
                return {
                    "concept_a": existing.name,
                    "concept_b": concept_name,
                    "explanation": (
                        f"'{concept_name}' appears to contradict "
                        f"verified concept '{existing.name}'"
                    ),
                }

            # Check keyword overlap with different concept names
            existing_keywords = set(
                w.lower() for w in existing.name.replace("_", " ").split()
            )
            new_keywords = set(w.lower() for w in keywords)
            overlap = existing_keywords & new_keywords

            if overlap and existing.name != concept_name:
                # Same domain keywords but different concept — potential conflict
                if self._is_antonym_pair(concept_name, existing.name):
                    return {
                        "concept_a": existing.name,
                        "concept_b": concept_name,
                        "explanation": (
                            f"'{concept_name}' shares keywords {overlap} with "
                            f"verified '{existing.name}' but may conflict"
                        ),
                    }

        return None

    @staticmethod
    def _is_antonym_pair(name_a: str, name_b: str) -> bool:
        """Check if two concept names form an antonym pair."""
        a_lower = name_a.lower().replace("_", "")
        b_lower = name_b.lower().replace("_", "")

        for prefix_a, prefix_b in ANTONYM_PREFIXES:
            if (
                (prefix_a in a_lower and prefix_b in b_lower)
                or (prefix_b in a_lower and prefix_a in b_lower)
            ):
                return True

        return False
