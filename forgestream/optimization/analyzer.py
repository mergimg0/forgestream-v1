"""PerformanceAnalyzer -- bottleneck detection, emotion-quality correlation, agent analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from forgestream.events.schema import Event, EventType


@dataclass
class AnalysisResult:
    """Output of PerformanceAnalyzer.analyze()."""

    # Bottleneck detection
    bottleneck_count: int = 0
    bottleneck_timestamps: list[datetime] = field(default_factory=list)

    # Emotion-quality correlation
    emotion_claim_correlation: float | None = None

    # Agent performance
    agent_timeout_count: int = 0

    # Throughput
    claims_per_minute: float = 0.0

    # Extra data for report generation
    high_engagement_confidence_avg: float | None = None
    low_engagement_confidence_avg: float | None = None
    rapport_verification_ratio: float | None = None


class PerformanceAnalyzer:
    """Processes the full ECEF event log and computes performance metrics.

    Bottleneck Detection:
        Consecutive CLAIM events with a gap > 10s are flagged as bottlenecks.

    Emotion-Extraction Correlation:
        Pearson r between mean EMOTION_STATE arousal in time windows
        and CLAIM density in those same windows (window size = 30s).

    Agent Performance:
        Counts ARTIFACT events with payload["timeout"] == True.
    """

    BOTTLENECK_THRESHOLD_SECONDS: float = 10.0
    WINDOW_SECONDS: float = 30.0

    def analyze(self, events: list[Event]) -> AnalysisResult:
        """Compute all performance metrics from the event log."""
        if not events:
            return AnalysisResult()

        bottleneck_count, bottleneck_timestamps = self._detect_bottlenecks(events)
        emotion_corr = self._emotion_quality_correlation(events)
        timeout_count = self._agent_timeout_analysis(events)
        claims_per_minute = self._claims_per_minute(events)
        high_conf, low_conf = self._engagement_confidence_split(events)
        rapport_ratio = self._rapport_verification_ratio(events)

        return AnalysisResult(
            bottleneck_count=bottleneck_count,
            bottleneck_timestamps=bottleneck_timestamps,
            emotion_claim_correlation=emotion_corr,
            agent_timeout_count=timeout_count,
            claims_per_minute=claims_per_minute,
            high_engagement_confidence_avg=high_conf,
            low_engagement_confidence_avg=low_conf,
            rapport_verification_ratio=rapport_ratio,
        )

    def _detect_bottlenecks(
        self, events: list[Event]
    ) -> tuple[int, list[datetime]]:
        """Find consecutive CLAIM pairs with gap > BOTTLENECK_THRESHOLD_SECONDS."""
        claims = sorted(
            [e for e in events if e.event_type == EventType.CLAIM],
            key=lambda e: e.timestamp,
        )
        bottleneck_timestamps: list[datetime] = []
        for prev, curr in zip(claims, claims[1:]):
            gap = (curr.timestamp - prev.timestamp).total_seconds()
            if gap > self.BOTTLENECK_THRESHOLD_SECONDS:
                bottleneck_timestamps.append(curr.timestamp)
        return len(bottleneck_timestamps), bottleneck_timestamps

    def _emotion_quality_correlation(
        self, events: list[Event]
    ) -> float | None:
        """Pearson r between EMOTION_STATE arousal window averages and CLAIM density.

        Divides the meeting into WINDOW_SECONDS bins.
        For each bin: compute mean arousal + count claims.
        Returns Pearson r, or None if insufficient data.
        """
        emotion_events = [e for e in events if e.event_type == EventType.EMOTION_STATE]
        claim_events = [e for e in events if e.event_type == EventType.CLAIM]

        if not emotion_events or not claim_events:
            return None

        # Find time bounds
        all_timestamps = [e.timestamp for e in events]
        t_min = min(all_timestamps)
        t_max = max(all_timestamps)
        duration = (t_max - t_min).total_seconds()

        if duration <= 0:
            return None

        # Build windows
        n_windows = max(2, int(duration / self.WINDOW_SECONDS) + 1)
        window_arousal: list[list[float]] = [[] for _ in range(n_windows)]
        window_claims: list[int] = [0] * n_windows

        for e in emotion_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            arousal = e.payload.get("arousal", 0.5)
            window_arousal[w].append(arousal)

        for e in claim_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            window_claims[w] += 1

        # Only include windows with at least one emotion reading
        xs: list[float] = []
        ys: list[float] = []
        for w in range(n_windows):
            if window_arousal[w]:
                xs.append(sum(window_arousal[w]) / len(window_arousal[w]))
                ys.append(float(window_claims[w]))

        if len(xs) < 2:
            return None

        return _pearson_r(xs, ys)

    def _agent_timeout_analysis(self, events: list[Event]) -> int:
        """Count ARTIFACT events with payload["timeout"] == True."""
        return sum(
            1
            for e in events
            if e.event_type == EventType.ARTIFACT
            and e.payload.get("timeout") is True
        )

    def _claims_per_minute(self, events: list[Event]) -> float:
        """Compute average claims per minute over the meeting duration."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if not claims or len(events) < 2:
            return 0.0
        t_min = min(e.timestamp for e in events)
        t_max = max(e.timestamp for e in events)
        duration_minutes = (t_max - t_min).total_seconds() / 60.0
        if duration_minutes <= 0:
            return 0.0
        return len(claims) / duration_minutes

    def _engagement_confidence_split(
        self, events: list[Event]
    ) -> tuple[float | None, float | None]:
        """Mean claim confidence in high-engagement (arousal > 0.7) vs low-engagement windows.

        Returns (high_avg, low_avg) or (None, None) if insufficient data.
        """
        emotion_events = sorted(
            [e for e in events if e.event_type == EventType.EMOTION_STATE],
            key=lambda e: e.timestamp,
        )
        claim_events = sorted(
            [e for e in events if e.event_type == EventType.CLAIM],
            key=lambda e: e.timestamp,
        )

        if not emotion_events or not claim_events:
            return None, None

        # Build window-level engagement scores
        all_timestamps = [e.timestamp for e in events]
        t_min = min(all_timestamps)
        duration = (max(all_timestamps) - t_min).total_seconds()
        if duration <= 0:
            return None, None

        n_windows = max(2, int(duration / self.WINDOW_SECONDS) + 1)
        window_arousal: list[list[float]] = [[] for _ in range(n_windows)]

        for e in emotion_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            window_arousal[w].append(e.payload.get("arousal", 0.5))

        high_confidences: list[float] = []
        low_confidences: list[float] = []

        for e in claim_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            if not window_arousal[w]:
                continue
            mean_arousal = sum(window_arousal[w]) / len(window_arousal[w])
            conf = e.payload.get("confidence", 0.5)
            if mean_arousal > 0.7:
                high_confidences.append(conf)
            else:
                low_confidences.append(conf)

        high_avg = sum(high_confidences) / len(high_confidences) if high_confidences else None
        low_avg = sum(low_confidences) / len(low_confidences) if low_confidences else None
        return high_avg, low_avg

    def _rapport_verification_ratio(self, events: list[Event]) -> float | None:
        """Ratio of VERIFIED_FINDING events in high-rapport windows vs overall.

        Returns ratio or None if no RAPPORT_SCORE events.
        """
        rapport_events = sorted(
            [e for e in events if e.event_type == EventType.RAPPORT_SCORE],
            key=lambda e: e.timestamp,
        )
        finding_events = sorted(
            [e for e in events if e.event_type == EventType.VERIFIED_FINDING],
            key=lambda e: e.timestamp,
        )

        if not rapport_events or not finding_events:
            return None

        all_timestamps = [e.timestamp for e in events]
        t_min = min(all_timestamps)
        duration = (max(all_timestamps) - t_min).total_seconds()
        if duration <= 0:
            return None

        n_windows = max(2, int(duration / self.WINDOW_SECONDS) + 1)
        window_rapport: list[list[float]] = [[] for _ in range(n_windows)]

        for e in rapport_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            window_rapport[w].append(e.payload.get("composite", 0.5))

        high_findings = 0
        low_findings = 0

        for e in finding_events:
            t = (e.timestamp - t_min).total_seconds()
            w = min(int(t / self.WINDOW_SECONDS), n_windows - 1)
            if not window_rapport[w]:
                continue
            mean_rapport = sum(window_rapport[w]) / len(window_rapport[w])
            if mean_rapport > 0.6:
                high_findings += 1
            else:
                low_findings += 1

        total = high_findings + low_findings
        if total == 0:
            return None
        return high_findings / total


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)
