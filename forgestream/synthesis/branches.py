"""Branch management and conversation drift detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class BranchInfo:
    id: str
    name: str
    keyword_centroid: Counter[str] = field(default_factory=Counter)
    claim_count: int = 0
    parent_branch_id: str | None = None


class BranchTracker:
    """Tracks conversation branches and detects topic drift."""

    DRIFT_THRESHOLD = 0.9  # Jaccard distance — only truly unrelated topics branch

    def __init__(self) -> None:
        self.main_branch_id = str(uuid4())
        self._branches: dict[str, BranchInfo] = {
            self.main_branch_id: BranchInfo(
                id=self.main_branch_id, name="main"
            ),
        }

    def get_branch(self, branch_id: str) -> BranchInfo | None:
        return self._branches.get(branch_id)

    def add_keywords(self, branch_id: str, keywords: list[str]) -> None:
        """Record keywords from a claim on a branch."""
        branch = self._branches.get(branch_id)
        if branch:
            branch.keyword_centroid.update(keywords)
            branch.claim_count += 1

    def check_drift(
        self,
        branch_id: str,
        new_keywords: list[str],
    ) -> dict[str, Any] | None:
        """Check if new keywords represent a topic drift from the branch.

        Returns a branch_point payload if drift detected, None otherwise.
        """
        branch = self._branches.get(branch_id)
        if not branch or not branch.keyword_centroid:
            return None

        existing_set = set(branch.keyword_centroid.keys())
        new_set = set(new_keywords)

        if not existing_set or not new_set:
            return None

        # Jaccard distance: 1 - |intersection| / |union|
        intersection = existing_set & new_set
        union = existing_set | new_set
        jaccard_distance = 1.0 - len(intersection) / len(union)

        if jaccard_distance >= self.DRIFT_THRESHOLD:
            new_branch_id = str(uuid4())
            new_branch = BranchInfo(
                id=new_branch_id,
                name=f"branch-{len(self._branches)}",
                parent_branch_id=branch_id,
            )
            new_branch.keyword_centroid.update(new_keywords)
            new_branch.claim_count = 1
            self._branches[new_branch_id] = new_branch

            # Estimate potential from novelty
            novelty = jaccard_distance
            potential = novelty * (len(new_keywords) / max(len(union), 1))

            return {
                "parent_branch_id": branch_id,
                "new_branch_id": new_branch_id,
                "potential_score": potential,
                "description": f"Topic drift detected: {', '.join(new_keywords[:3])}",
            }

        return None

    def get_metrics(self, branch_id: str) -> dict[str, Any]:
        """Get branch metrics for display."""
        branch = self._branches.get(branch_id)
        if not branch:
            return {"claim_count": 0, "potential": 0.0, "keyword_count": 0}

        keyword_count = len(branch.keyword_centroid)
        potential = min(1.0, keyword_count * 0.1) if keyword_count > 0 else 0.0

        return {
            "claim_count": branch.claim_count,
            "potential": potential,
            "keyword_count": keyword_count,
        }

    @property
    def all_branches(self) -> list[BranchInfo]:
        return list(self._branches.values())
