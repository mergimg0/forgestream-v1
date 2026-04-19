"""Tests for TrustRegion save/load persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from forgestream.governor.trust_region import TrustRegion


class TestTrustRegionPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trust_region.json"

            tr = TrustRegion()
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
            original_epsilon = tr.epsilon

            tr.save(path)
            assert path.exists()

            loaded = TrustRegion.load(path)
            assert loaded._consecutive_improvements == tr._consecutive_improvements
            assert loaded._total_violations == tr._total_violations
            assert loaded._meeting_count == tr._meeting_count
            assert loaded.epsilon == pytest.approx(original_epsilon)

    def test_load_missing_file_returns_default(self):
        loaded = TrustRegion.load("/nonexistent/path.json")
        assert loaded._consecutive_improvements == 0
        assert loaded._meeting_count == 0

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "trust.json"
            tr = TrustRegion()
            tr.save(path)
            assert path.exists()
