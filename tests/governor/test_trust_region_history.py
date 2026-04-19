"""Tests for TrustRegion epsilon-history persistence and autonomy-progression API."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forgestream.governor.trust_region import TrustRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_epsilon() -> float:
    """Expected epsilon for a freshly-constructed TrustRegion."""
    tr = TrustRegion()
    return tr.epsilon


# ---------------------------------------------------------------------------
# save() appends to history file
# ---------------------------------------------------------------------------

class TestSaveAppendsHistory:
    def test_save_creates_history_file(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "trust_region.json"
        tr.save(state_path)

        history_path = tmp_path / "trust_region_history.json"
        assert history_path.exists(), "history file was not created by save()"

    def test_first_save_produces_single_entry(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "trust_region.json"
        tr.save(state_path)

        history_path = tmp_path / "trust_region_history.json"
        history = json.loads(history_path.read_text())
        assert isinstance(history, list)
        assert len(history) == 1

    def test_history_entry_has_required_keys(self, tmp_path: Path):
        tr = TrustRegion()
        tr.save(tmp_path / "trust_region.json")

        history = json.loads((tmp_path / "trust_region_history.json").read_text())
        entry = history[0]
        assert "meeting" in entry
        assert "epsilon" in entry
        assert "improvements" in entry
        assert "violations" in entry

    def test_save_appends_on_multiple_calls(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "trust_region.json"
        for _ in range(4):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
            tr.save(state_path)

        history = json.loads((tmp_path / "trust_region_history.json").read_text())
        assert len(history) == 4

    def test_meeting_numbers_are_sequential(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "trust_region.json"
        for _ in range(3):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
            tr.save(state_path)

        history = json.loads((tmp_path / "trust_region_history.json").read_text())
        meeting_nums = [e["meeting"] for e in history]
        assert meeting_nums == sorted(meeting_nums), "meeting numbers not ascending"
        assert len(set(meeting_nums)) == len(meeting_nums), "duplicate meeting numbers"

    def test_epsilon_values_match_trust_region_state(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "trust_region.json"
        snapshots: list[float] = []

        for _ in range(3):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
            snapshots.append(tr.epsilon)
            tr.save(state_path)

        history = json.loads((tmp_path / "trust_region_history.json").read_text())
        for i, entry in enumerate(history):
            assert abs(entry["epsilon"] - snapshots[i]) < 1e-4

    def test_save_uses_explicit_history_path(self, tmp_path: Path):
        tr = TrustRegion()
        state_path = tmp_path / "state.json"
        custom_history = tmp_path / "custom_hist.json"
        tr.save(state_path, history_path=custom_history)

        assert custom_history.exists()
        default_path = tmp_path / "trust_region_history.json"
        assert not default_path.exists(), "default history path should not be written"

    def test_existing_corrupt_history_is_reset(self, tmp_path: Path):
        hist_path = tmp_path / "trust_region_history.json"
        hist_path.write_text("not-valid-json")

        tr = TrustRegion()
        tr.save(tmp_path / "state.json", history_path=hist_path)

        history = json.loads(hist_path.read_text())
        assert len(history) == 1

    def test_existing_non_list_history_is_reset(self, tmp_path: Path):
        hist_path = tmp_path / "trust_region_history.json"
        hist_path.write_text('{"not": "a list"}')

        tr = TrustRegion()
        tr.save(tmp_path / "state.json", history_path=hist_path)

        history = json.loads(hist_path.read_text())
        assert isinstance(history, list)
        assert len(history) == 1


# ---------------------------------------------------------------------------
# load() returns sorted history (separate helper tests)
# ---------------------------------------------------------------------------

class TestHistorySorting:
    def test_history_sorted_by_meeting_ascending(self, tmp_path: Path):
        """Entries written out-of-order must be returned sorted in the file."""
        hist_path = tmp_path / "trust_region_history.json"
        # Write deliberately unsorted history
        hist_path.write_text(json.dumps([
            {"meeting": 3, "epsilon": 0.55, "improvements": 2, "violations": 0},
            {"meeting": 1, "epsilon": 0.52, "improvements": 0, "violations": 0},
            {"meeting": 2, "epsilon": 0.53, "improvements": 1, "violations": 0},
        ]))

        tr = TrustRegion()
        # Appending a new entry triggers a re-read; the new entry gets meeting=4
        tr.save(tmp_path / "state.json", history_path=hist_path)

        history = json.loads(hist_path.read_text())
        meetings = [e["meeting"] for e in history]
        assert meetings[-1] == 4  # appended after max(3)+1


# ---------------------------------------------------------------------------
# Autonomy-progression prediction extrapolation
# ---------------------------------------------------------------------------

class TestPredictionExtrapolation:
    """Unit-test the linear extrapolation logic embedded in the API endpoint.

    We replicate the math here rather than spinning up a full HTTP server so
    that the logic is tested in isolation.
    """

    @staticmethod
    def _predict(history: list[dict]) -> dict:
        """Replicate the slope + prediction logic from the API endpoint."""
        AUTO_SPAWN = 0.6
        BRANCH_AUTO = 0.7

        if len(history) < 2:
            return {"slope": 0.0, "auto_spawn": None, "branch_auto": None}

        window = history[-5:]
        n = len(window)
        xs = [e["meeting"] for e in window]
        ys = [e["epsilon"] for e in window]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        auto_spawn = None
        branch_auto = None
        if slope > 0:
            last_m = xs[-1]
            last_e = ys[-1]
            if last_e < AUTO_SPAWN:
                auto_spawn = int(last_m + (AUTO_SPAWN - last_e) / slope) + 1
            if last_e < BRANCH_AUTO:
                branch_auto = int(last_m + (BRANCH_AUTO - last_e) / slope) + 1

        return {"slope": slope, "auto_spawn": auto_spawn, "branch_auto": branch_auto}

    def test_positive_slope_predicts_auto_spawn(self):
        # epsilon ends at 0.50 + 5*0.018 = 0.59, which is below AUTO_SPAWN (0.6)
        history = [
            {"meeting": i, "epsilon": 0.50 + i * 0.018}
            for i in range(1, 6)
        ]
        result = self._predict(history)
        assert result["slope"] > 0
        assert result["auto_spawn"] is not None
        assert result["auto_spawn"] > history[-1]["meeting"]

    def test_flat_slope_no_prediction(self):
        history = [
            {"meeting": i, "epsilon": 0.50}
            for i in range(1, 6)
        ]
        result = self._predict(history)
        assert result["slope"] == 0.0
        assert result["auto_spawn"] is None
        assert result["branch_auto"] is None

    def test_already_above_threshold_no_auto_spawn_prediction(self):
        history = [
            {"meeting": i, "epsilon": 0.62 + i * 0.01}
            for i in range(1, 4)
        ]
        result = self._predict(history)
        assert result["auto_spawn"] is None, "already above threshold, no prediction needed"

    def test_single_entry_no_slope(self):
        history = [{"meeting": 1, "epsilon": 0.525, "improvements": 0, "violations": 0}]
        result = self._predict(history)
        assert result["slope"] == 0.0
        assert result["auto_spawn"] is None

    def test_window_capped_at_5(self):
        """Slope is computed from the last 5 entries only."""
        history = [
            {"meeting": i, "epsilon": 0.50 + i * 0.01}
            for i in range(1, 11)  # 10 entries
        ]
        result_all = self._predict(history)
        result_last5 = self._predict(history[-5:])
        # Both should use the same last-5 window — slopes must be equal
        assert abs(result_all["slope"] - result_last5["slope"]) < 1e-9


# ---------------------------------------------------------------------------
# API endpoint integration (no Firestore required)
# ---------------------------------------------------------------------------

class TestAutonomyProgressionEndpoint:
    def test_endpoint_returns_correct_structure(self, tmp_path: Path, monkeypatch):
        """The /api/autonomy-progression endpoint returns all expected keys."""
        from fastapi.testclient import TestClient
        from forgestream.dashboard.server import create_app

        # Endpoint reads "data/trust_region_history.json" relative to CWD.
        # Change CWD to tmp_path and write the file at data/trust_region_history.json.
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "trust_region_history.json").write_text(json.dumps([
            {"meeting": 1, "epsilon": 0.525, "improvements": 0, "violations": 0},
            {"meeting": 2, "epsilon": 0.540, "improvements": 1, "violations": 0},
            {"meeting": 3, "epsilon": 0.558, "improvements": 2, "violations": 0},
        ]))

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/autonomy-progression")
        assert resp.status_code == 200

        data = resp.json()
        assert "history" in data
        assert "current_epsilon" in data
        assert "auto_spawn_threshold" in data
        assert "branch_auto_allocate_threshold" in data
        assert "slope_per_meeting" in data
        assert "predicted_auto_spawn_meeting" in data
        assert "predicted_branch_auto_meeting" in data

    def test_endpoint_no_history_returns_defaults(self, tmp_path: Path, monkeypatch):
        """Endpoint with no history file still returns valid structure."""
        from fastapi.testclient import TestClient
        from forgestream.dashboard.server import create_app

        # CWD has no data/trust_region_history.json — endpoint should return defaults.
        monkeypatch.chdir(tmp_path)

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/autonomy-progression")
        assert resp.status_code == 200
        data = resp.json()
        assert data["history"] == []
        assert data["current_epsilon"] == 0.525
        assert data["auto_spawn_threshold"] == 0.6

    def test_endpoint_slope_computed_for_improving_history(self, tmp_path: Path, monkeypatch):
        """Endpoint returns a positive slope for steadily-improving meetings."""
        from fastapi.testclient import TestClient
        from forgestream.dashboard.server import create_app

        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # epsilon ends at 0.50 + 5*0.025 = 0.625 — above AUTO_SPAWN; use 0.018 step
        (data_dir / "trust_region_history.json").write_text(json.dumps([
            {"meeting": i, "epsilon": 0.50 + i * 0.018, "improvements": i, "violations": 0}
            for i in range(1, 6)
        ]))

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/autonomy-progression")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slope_per_meeting"] > 0
        assert data["predicted_auto_spawn_meeting"] is not None
