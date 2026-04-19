"""Group dynamics computation: TLCC, CRQA, participation parity, entropy.

Subscribes to EventBus. Every 60 seconds, computes group dynamics from
accumulated per-speaker prosodic time series and emits an ENTRAINMENT_SNAPSHOT.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from uuid import UUID

import numpy as np
from scipy import signal as scipy_signal

from forgestream.events.schema import Event, EventType

from .speaker import SpeakerTimeSeries

logger = logging.getLogger(__name__)

AUTHOR = "dynamics_engine"
SNAPSHOT_INTERVAL_MS = 60_000


class GroupDynamicsEngine:
    """Computes meta-vectors from per-speaker prosodic time series.

    Subscribes to the EventBus. Accumulates PROSODIC_FEATURE events into
    per-speaker time series, and every 60 seconds computes TLCC, CRQA,
    participation parity, and turn-taking entropy, emitting an
    ENTRAINMENT_SNAPSHOT event.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator
        self._speaker_series = SpeakerTimeSeries()
        self._last_snapshot_ms = 0
        self._speaker_sequence: list[str] = []
        self._last_speaker: str | None = None
        self._pair_correlation_history: dict[str, list[float]] = {}

    async def on_event(self, event: Event) -> None:
        """EventBus handler. Accumulates PROSODIC_FEATURE, periodically emits snapshots."""
        if event.author == AUTHOR:
            return
        if event.event_type != EventType.PROSODIC_FEATURE:
            return

        payload = event.payload
        speaker_id = payload.get("speaker_id", "unknown")
        timestamp_ms = payload.get("timestamp_ms", 0)

        self._speaker_series.add_feature(speaker_id, timestamp_ms, payload)

        # Track turn-taking sequence
        if speaker_id != self._last_speaker:
            self._speaker_sequence.append(speaker_id)
            self._last_speaker = speaker_id

        # Emit snapshot every SNAPSHOT_INTERVAL_MS
        if timestamp_ms - self._last_snapshot_ms >= SNAPSHOT_INTERVAL_MS:
            self._last_snapshot_ms = timestamp_ms
            await self._emit_snapshot(event.session_id, event.branch_id, timestamp_ms)

    def compute_tlcc(
        self,
        series_a: list[float],
        series_b: list[float],
        max_lag: int = 30,
    ) -> tuple[float, int]:
        """Time-Lagged Cross-Correlation.

        Returns (peak_correlation, lag_index).
        Positive lag = series_a leads series_b.
        """
        if len(series_a) < 3 or len(series_b) < 3:
            return 0.0, 0

        a = np.array(series_a, dtype=np.float64)
        b = np.array(series_b, dtype=np.float64)
        a = a - np.mean(a)
        b = b - np.mean(b)

        correlation = scipy_signal.correlate(a, b, mode="full")
        norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if norm > 0:
            correlation = correlation / norm

        mid = len(correlation) // 2
        lo = max(0, mid - max_lag)
        hi = min(len(correlation), mid + max_lag + 1)
        window = correlation[lo:hi]

        if len(window) == 0:
            return 0.0, 0

        peak_idx = int(np.argmax(np.abs(window)))
        lag = peak_idx - (mid - lo)
        return float(window[peak_idx]), lag

    def compute_crqa_recurrence_rate(
        self,
        series_a: list[float],
        series_b: list[float],
        radius: float = 0.1,
    ) -> float:
        """Cross-Recurrence Quantification: recurrence rate.

        Returns fraction of points in cross-recurrence plot that are recurrent.
        """
        if len(series_a) == 0 or len(series_b) == 0:
            return 0.0

        a = np.array(series_a, dtype=np.float64)
        b = np.array(series_b, dtype=np.float64)
        combined_std = np.std(np.concatenate([a, b]))
        if combined_std == 0:
            return 1.0  # constant identical signals

        threshold = radius * combined_std
        dist = np.abs(a[:, None] - b[None, :])
        recurrent = dist < threshold
        return float(np.mean(recurrent))

    def compute_participation_parity(
        self, durations: dict[str, float]
    ) -> float:
        """How evenly distributed is speaking time?

        Returns 1.0 for perfectly even, 0.0 for single-speaker dominance.
        Uses 1.0 - Gini coefficient.
        """
        if len(durations) <= 1:
            return 1.0

        values = sorted(durations.values())
        n = len(values)
        total = sum(values)
        if total == 0:
            return 1.0

        # Gini coefficient
        cumulative = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(values):
            cumulative += v
            weighted_sum += (i + 1) * v

        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, 1.0 - gini))

    def compute_turn_taking_entropy(
        self, speaker_sequence: list[str]
    ) -> float:
        """Shannon entropy of the turn-taking transition matrix.

        High entropy = unpredictable/collaborative turn-taking.
        Low entropy = rigid/formal structure.
        Normalized to [0, 1] by dividing by max possible entropy.
        """
        if len(speaker_sequence) < 2:
            return 0.0

        # Build transition counts
        transitions: dict[tuple[str, str], int] = {}
        for i in range(len(speaker_sequence) - 1):
            pair = (speaker_sequence[i], speaker_sequence[i + 1])
            transitions[pair] = transitions.get(pair, 0) + 1

        total = sum(transitions.values())
        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in transitions.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max entropy (log2 of unique transitions possible)
        speakers = set(speaker_sequence)
        max_transitions = len(speakers) ** 2
        max_entropy = math.log2(max_transitions) if max_transitions > 1 else 1.0

        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def compute_collective_engagement(
        self, prosodic_payloads: list[dict]
    ) -> float:
        """Mean arousal x mean F0 variability across all speakers."""
        if not prosodic_payloads:
            return 0.0
        arousals = [p.get("arousal", 0.5) for p in prosodic_payloads]
        f0_stds = [p.get("f0_std", 0.0) for p in prosodic_payloads]
        mean_arousal = sum(arousals) / len(arousals)
        mean_f0_var = min(1.0, (sum(f0_stds) / len(f0_stds)) / 80.0)
        return mean_arousal * mean_f0_var

    async def _emit_snapshot(
        self, session_id: UUID, branch_id: UUID, timestamp_ms: int
    ) -> None:
        """Build and emit ENTRAINMENT_SNAPSHOT event."""
        speakers = self._speaker_series.speaker_ids()
        durations = self._speaker_series.speaking_durations()

        # Compute pairwise metrics
        speaker_pairs = []
        for sp_a, sp_b in combinations(speakers, 2):
            f0_a, f0_b = self._speaker_series.get_f0_series(sp_a, sp_b)
            if len(f0_a) >= 3 and len(f0_b) >= 3:
                f0_corr, f0_lag = self.compute_tlcc(f0_a, f0_b)
                en_a = self._speaker_series.get_energy_series(sp_a)
                en_b = self._speaker_series.get_energy_series(sp_b)
                en_corr, _ = self.compute_tlcc(en_a, en_b)
                sync = self.compute_crqa_recurrence_rate(f0_a, f0_b)
            else:
                f0_corr, f0_lag, en_corr, sync = 0.0, 0, 0.0, 0.0

            # Track correlation history for convergence trend
            pair_key = f"{sp_a}:{sp_b}"
            if pair_key not in self._pair_correlation_history:
                self._pair_correlation_history[pair_key] = []
            self._pair_correlation_history[pair_key].append(abs(f0_corr))
            history = self._pair_correlation_history[pair_key]

            # Convergence trend: Pearson r of |correlation| vs time
            if len(history) >= 3:
                x = np.arange(len(history), dtype=np.float64)
                y = np.array(history, dtype=np.float64)
                if y.std() > 1e-10:
                    trend = float(np.corrcoef(x, y)[0, 1])
                    if np.isnan(trend):
                        trend = 0.0
                else:
                    trend = 0.0
            else:
                trend = 0.0

            speaker_pairs.append({
                "speaker_a": sp_a,
                "speaker_b": sp_b,
                "f0_correlation": round(f0_corr, 4),
                "f0_lag_ms": float(f0_lag) * 1000.0,
                "energy_correlation": round(en_corr, 4),
                "synchrony_score": round(sync, 4),
                "convergence_trend": round(trend, 4),
            })

        # Determine dominant speaker
        dominant = max(durations, key=lambda k: durations[k]) if durations else None

        # Collect recent prosodic payloads for engagement calc
        all_payloads: list[dict] = []
        for entries in self._speaker_series._series.values():
            for _, feat in entries:
                all_payloads.append(feat)

        group_metrics = {
            "participation_parity": round(
                self.compute_participation_parity(durations), 4
            ),
            "collective_engagement": round(
                self.compute_collective_engagement(all_payloads), 4
            ),
            "turn_taking_entropy": round(
                self.compute_turn_taking_entropy(self._speaker_sequence), 4
            ),
            "dominant_speaker": dominant,
        }

        event = Event(
            event_type=EventType.ENTRAINMENT_SNAPSHOT,
            session_id=session_id,
            branch_id=branch_id,
            author=AUTHOR,
            evaluator=0.0,
            payload={
                "timestamp_ms": timestamp_ms,
                "window_duration_ms": SNAPSHOT_INTERVAL_MS,
                "speaker_pairs": speaker_pairs,
                "group_metrics": group_metrics,
            },
        )
        await self._orchestrator.process_event(event)
