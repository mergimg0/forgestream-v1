"""Tests for PerformanceAnalyzer -- bottleneck detection, emotion-quality correlation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.optimization.analyzer import AnalysisResult, PerformanceAnalyzer


def _make_event(
    event_type: EventType,
    payload: dict | None = None,
    ts: datetime | None = None,
) -> Event:
    return Event(
        event_type=event_type,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="test",
        evaluator=0.5,
        payload=payload or {},
        timestamp=ts or datetime.now(timezone.utc),
    )


class TestBottleneckDetection:
    def test_bottleneck_detection_finds_gaps(self):
        """Consecutive CLAIM events with > 10s gap are flagged as bottlenecks."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(EventType.CLAIM, ts=t0),
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=5)),   # 5s gap — ok
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=20)),  # 15s gap — bottleneck
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=45)),  # 25s gap — bottleneck
        ]
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(events)

        assert isinstance(result, AnalysisResult)
        assert result.bottleneck_count == 2
        assert len(result.bottleneck_timestamps) == 2

    def test_no_bottlenecks_in_normal_flow(self):
        """No bottlenecks when all CLAIM gaps are <= 10s."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(EventType.CLAIM, ts=t0),
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=3)),
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=7)),
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=10)),
        ]
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(events)

        assert result.bottleneck_count == 0
        assert result.bottleneck_timestamps == []

    def test_empty_events(self):
        """Analyzer handles empty event list gracefully."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([])

        assert isinstance(result, AnalysisResult)
        assert result.bottleneck_count == 0
        assert result.emotion_claim_correlation is None
        assert result.bottleneck_timestamps == []


class TestEmotionQualityCorrelation:
    def test_emotion_quality_correlation(self):
        """Pearson r is computed between engagement windows and claim density."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # High engagement at t=0..30 → dense claims
        # Low engagement at t=30..60 → sparse claims
        events: list[Event] = []

        # High-engagement window: arousal ~0.8, many claims
        for i in range(6):
            events.append(_make_event(
                EventType.EMOTION_STATE,
                payload={"arousal": 0.8, "valence": 0.7},
                ts=t0 + timedelta(seconds=i * 5),
            ))
        for i in range(5):
            events.append(_make_event(
                EventType.CLAIM,
                payload={"confidence": 0.9},
                ts=t0 + timedelta(seconds=i * 6),
            ))

        # Low-engagement window: arousal ~0.2, few claims
        for i in range(6):
            events.append(_make_event(
                EventType.EMOTION_STATE,
                payload={"arousal": 0.2, "valence": 0.3},
                ts=t0 + timedelta(seconds=30 + i * 5),
            ))
        # Only 1 claim in low-engagement window
        events.append(_make_event(
            EventType.CLAIM,
            payload={"confidence": 0.4},
            ts=t0 + timedelta(seconds=50),
        ))

        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(events)

        # Should find a positive correlation
        assert result.emotion_claim_correlation is not None
        assert isinstance(result.emotion_claim_correlation, float)
        # Correlation should be positive (high engagement → more claims)
        assert result.emotion_claim_correlation > 0.0

    def test_emotion_quality_correlation_no_emotion_events(self):
        """Returns None when no EMOTION_STATE events present."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(EventType.CLAIM, ts=t0),
            _make_event(EventType.CLAIM, ts=t0 + timedelta(seconds=5)),
        ]
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(events)

        assert result.emotion_claim_correlation is None
