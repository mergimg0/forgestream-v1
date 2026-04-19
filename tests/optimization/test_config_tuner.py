"""Tests for ConfigTuner -- GRPO-based system config parameter tuning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.optimization.analyzer import AnalysisResult, PerformanceAnalyzer
from forgestream.optimization.config_tuner import ConfigTuner


def _make_event(event_type: EventType, payload: dict | None = None) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.5,
        payload=payload or {},
    )


def _make_analysis(bottleneck_count: int = 0, timeout_count: int = 0) -> AnalysisResult:
    return AnalysisResult(
        bottleneck_count=bottleneck_count,
        bottleneck_timestamps=[],
        emotion_claim_correlation=0.5,
        agent_timeout_count=timeout_count,
        claims_per_minute=3.0,
    )


class TestConfigTuner:
    def test_tune_returns_valid_config(self):
        """tune() returns a dict with all expected parameter keys."""
        tuner = ConfigTuner()
        current = {k: (lo + hi) / 2 for k, (lo, hi) in ConfigTuner.TUNABLE_PARAMS.items()}
        analysis = _make_analysis()

        result = tuner.tune(current, analysis, human_score=0.7)

        assert isinstance(result, dict)
        for key in ConfigTuner.TUNABLE_PARAMS:
            assert key in result

    def test_params_stay_in_range(self):
        """All tuned parameters stay within their (min, max) bounds."""
        tuner = ConfigTuner()
        # Start at midpoint
        current = {k: (lo + hi) / 2 for k, (lo, hi) in ConfigTuner.TUNABLE_PARAMS.items()}
        analysis = _make_analysis(bottleneck_count=3, timeout_count=2)

        for _ in range(20):  # Run multiple times to check bounds robustly
            result = tuner.tune(current, analysis, human_score=0.5)
            for key, (lo, hi) in ConfigTuner.TUNABLE_PARAMS.items():
                assert lo <= result[key] <= hi, (
                    f"{key}={result[key]} out of range [{lo}, {hi}]"
                )
            current = result

    def test_backward_compatible(self):
        """tune() with extra unknown keys in current_params doesn't crash."""
        tuner = ConfigTuner()
        current = {k: (lo + hi) / 2 for k, (lo, hi) in ConfigTuner.TUNABLE_PARAMS.items()}
        current["unknown_future_param"] = 42.0  # Extra key
        analysis = _make_analysis()

        # Should not raise
        result = tuner.tune(current, analysis, human_score=0.6)
        assert isinstance(result, dict)
        # Known params should be present
        for key in ConfigTuner.TUNABLE_PARAMS:
            assert key in result

    def test_tunable_params_covers_expected_keys(self):
        """TUNABLE_PARAMS contains the expected system configuration keys."""
        expected = {
            "spawn_cooldown_seconds",
            "scaffold_timeout_minutes",
            "max_concurrent_research",
            "max_concurrent_scaffold",
            "emotion_stride_seconds",
            "emotion_window_seconds",
        }
        assert expected == set(ConfigTuner.TUNABLE_PARAMS.keys())
