"""Per-speaker prosodic time series accumulator.

Subscribes to PROSODIC_FEATURE events and maintains rolling time series
per speaker_id. Used by GroupDynamicsEngine for TLCC/CRQA computation.
"""

from __future__ import annotations

from collections import deque


class SpeakerTimeSeries:
    """Accumulates prosodic features per speaker over time.

    Parameters:
        max_duration_seconds: Maximum history to retain per speaker.
        stride_seconds: Duration each feature entry represents.
    """

    def __init__(
        self,
        max_duration_seconds: float = 120.0,
        stride_seconds: float = 1.0,
    ) -> None:
        self._stride_seconds = stride_seconds
        max_entries = max(1, int(max_duration_seconds / stride_seconds))
        self._max_entries = max_entries
        self._series: dict[str, deque[tuple[int, dict]]] = {}

    def add_feature(
        self, speaker_id: str, timestamp_ms: int, features: dict
    ) -> None:
        """Add a prosodic feature snapshot for a speaker."""
        if speaker_id not in self._series:
            self._series[speaker_id] = deque(maxlen=self._max_entries)
        self._series[speaker_id].append((timestamp_ms, features))

    def get_f0_series(
        self, speaker_a: str, speaker_b: str
    ) -> tuple[list[float], list[float]]:
        """Get F0 mean time series for two speakers."""
        a_entries = self._series.get(speaker_a, deque())
        b_entries = self._series.get(speaker_b, deque())
        f0_a = [feat.get("f0_mean", 0.0) for _, feat in a_entries]
        f0_b = [feat.get("f0_mean", 0.0) for _, feat in b_entries]
        return f0_a, f0_b

    def get_energy_series(self, speaker_id: str) -> list[float]:
        """Get energy RMS time series for a speaker."""
        entries = self._series.get(speaker_id, deque())
        return [feat.get("energy_rms", 0.0) for _, feat in entries]

    def speaker_ids(self) -> list[str]:
        """List all known speaker IDs."""
        return list(self._series.keys())

    def speaking_durations(self) -> dict[str, float]:
        """Compute total speaking duration per speaker in seconds.

        Each entry represents one stride of audio.
        """
        return {
            sid: len(entries) * self._stride_seconds
            for sid, entries in self._series.items()
        }
