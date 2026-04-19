from uuid import uuid4

from forgestream.synthesis.branches import BranchTracker, BranchInfo


class TestBranchTracker:
    def test_create_main_branch(self):
        tracker = BranchTracker()
        main = tracker.get_branch(tracker.main_branch_id)
        assert main is not None
        assert main.name == "main"

    def test_add_claim_to_branch(self):
        tracker = BranchTracker()
        tracker.add_keywords(tracker.main_branch_id, ["Kafka", "ingestion"])
        main = tracker.get_branch(tracker.main_branch_id)
        assert "Kafka" in main.keyword_centroid
        assert "ingestion" in main.keyword_centroid

    def test_detect_drift(self):
        tracker = BranchTracker()
        # Establish main topic
        for _ in range(5):
            tracker.add_keywords(tracker.main_branch_id, ["Kafka", "ingestion", "pipeline"])

        # Introduce a drifting topic
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["quantum", "entanglement", "physics"],
        )
        assert drift is not None
        assert drift["potential_score"] > 0

    def test_no_drift_for_related_topic(self):
        tracker = BranchTracker()
        tracker.add_keywords(tracker.main_branch_id, ["Kafka", "ingestion", "consumer", "pipeline"])

        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["Kafka", "consumer", "offset"],
        )
        # Significant keyword overlap — shouldn't trigger a branch
        assert drift is None

    def test_partially_related_no_branch(self):
        """Topics with some overlap should NOT branch."""
        tracker = BranchTracker()
        for _ in range(3):
            tracker.add_keywords(tracker.main_branch_id,
                                 ["rust", "agents", "lean_4", "proof", "verification"])
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["rust", "performance", "memory_safety"],
        )
        assert drift is None

    def test_branch_metrics(self):
        tracker = BranchTracker()
        tracker.add_keywords(tracker.main_branch_id, ["A", "B"])
        tracker.add_keywords(tracker.main_branch_id, ["C", "D"])

        metrics = tracker.get_metrics(tracker.main_branch_id)
        assert metrics["claim_count"] == 2
        assert metrics["potential"] >= 0
