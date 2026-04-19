"""RapportEngine — multi-component rapport scoring.

Subscribes to EventBus. Consumes PROSODIC_FEATURE events (for disengagement
detection and arousal/valence tracking) and ENTRAINMENT_SNAPSHOT events
(as trigger to compute rapport). Emits RAPPORT_SCORE events with composite
+ per-pair component scores.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from itertools import combinations

import numpy as np

from forgestream.events.schema import Event, EventType

from .crqa_router import CRQAComputeRouter
from .disengagement import DisengagementDetector
from .transfer_entropy import compute_symmetry, compute_transfer_entropy

logger = logging.getLogger(__name__)

AUTHOR = "rapport_engine"

EARLY_WEIGHTS = {
    "attentiveness": 0.35,
    "positivity": 0.30,
    "coordination": 0.15,
    "symmetry": 0.20,
}
ESTABLISHED_WEIGHTS = {
    "attentiveness": 0.20,
    "positivity": 0.15,
    "coordination": 0.40,
    "symmetry": 0.25,
}


def interpolate_weights_from_t(t: float) -> dict[str, float]:
    """Interpolate between early and established weight profiles using t directly."""
    t = max(0.0, min(1.0, t))
    return {
        k: EARLY_WEIGHTS[k] * (1 - t) + ESTABLISHED_WEIGHTS[k] * t
        for k in EARLY_WEIGHTS
    }


def interpolate_weights(meeting_count: int) -> dict[str, float]:
    """Sigmoid interpolation between early and established weight profiles."""
    t = 1.0 / (1.0 + math.exp(-(meeting_count - 3)))
    return interpolate_weights_from_t(t)


EQUAL_WEIGHTS = {
    "attentiveness": 0.25,
    "positivity": 0.25,
    "coordination": 0.25,
    "symmetry": 0.25,
}

MATURITY_SNAPSHOTS_REQUIRED = 6  # ~3 minutes at 30s intervals


def infer_maturity(
    coordination_values: list[float],
    symmetry_values: list[float],
) -> float:
    """Infer relationship maturity from observed prosodic signals.

    Uses coordination level (CRQA %DET) and symmetry stability to estimate
    how established the speaker relationship is. Returns t in [0, 1].

    Low coordination + high symmetry variance -> early relationship (t~0)
    High coordination + low symmetry variance -> established relationship (t~1)
    """
    if not coordination_values or not symmetry_values:
        return 0.0

    mean_coord = sum(coordination_values) / len(coordination_values)

    # Symmetry stability = inverse of coefficient of variation
    sym_mean = sum(symmetry_values) / len(symmetry_values)
    if sym_mean > 0.01:
        sym_std = (
            sum((v - sym_mean) ** 2 for v in symmetry_values) / len(symmetry_values)
        ) ** 0.5
        sym_stability = max(0.0, 1.0 - (sym_std / sym_mean))
    else:
        sym_stability = 0.0

    # Weighted combination: coordination is the stronger signal
    raw = 0.7 * mean_coord + 0.3 * sym_stability

    return max(0.0, min(1.0, raw))


class RapportEngine:
    """Multi-component rapport scoring engine.

    Parameters:
        orchestrator: The ForgeStream Orchestrator.
        meeting_count: Number of prior meetings (for weight interpolation).
        damping_factor: Disengagement damping multiplier.
        runpod_endpoint: RunPod CRQA endpoint URL.
        runpod_timeout: RunPod request timeout.
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        meeting_count: int = 1,
        damping_factor: float = 0.3,
        runpod_endpoint: str = "",
        runpod_timeout: float = 4.0,
        rapport_weights: dict[str, float] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._meeting_count = meeting_count

        # Start with equal weights; refine after maturity inference
        if rapport_weights is not None:
            self._weights = rapport_weights
            self._maturity_inferred = True
        else:
            self._weights = EQUAL_WEIGHTS.copy()
            self._maturity_inferred = False

        # Track coordination and symmetry for maturity inference
        self._coordination_history: list[float] = []
        self._symmetry_history: list[float] = []
        self._snapshot_count = 0

        self._disengagement = DisengagementDetector(damping_factor=damping_factor)
        self._crqa_router = CRQAComputeRouter(
            runpod_endpoint=runpod_endpoint, timeout=runpod_timeout,
        )

        self._arousal_series: dict[str, deque[float]] = {}
        self._valence_series: dict[str, deque[float]] = {}
        self._f0_series: dict[str, deque[float]] = {}
        self._series_maxlen = 60

        # CRQA parameters — estimated from first 60s of data
        self._crqa_params_estimated = False
        self._crqa_embedding_dim = 3
        self._crqa_time_delay = 2
        self._crqa_radius = 0.25

        self._recent_composites: deque[float] = deque(maxlen=10)

    async def on_event(self, event: Event) -> None:
        """EventBus handler."""
        if event.author == AUTHOR:
            return

        if event.event_type == EventType.PROSODIC_FEATURE:
            self._handle_prosodic(event)
        elif event.event_type == EventType.ENTRAINMENT_SNAPSHOT:
            await self._handle_snapshot(event)

    def _handle_prosodic(self, event: Event) -> None:
        """Update per-speaker series and disengagement detector."""
        p = event.payload
        speaker = p.get("speaker_id", "unknown")

        self._disengagement.update(speaker, p)

        for series_dict, key, default in [
            (self._arousal_series, "arousal", 0.5),
            (self._valence_series, "valence", 0.5),
            (self._f0_series, "f0_mean", 0.0),
        ]:
            if speaker not in series_dict:
                series_dict[speaker] = deque(maxlen=self._series_maxlen)
            series_dict[speaker].append(p.get(key, default))

    async def _handle_snapshot(self, event: Event) -> None:
        """Compute and emit RAPPORT_SCORE when triggered by ENTRAINMENT_SNAPSHOT."""
        speakers = list(self._arousal_series.keys())
        if len(speakers) < 2:
            return

        pair_scores = []
        for sp_a, sp_b in combinations(speakers, 2):
            score = await self._compute_pair_rapport(sp_a, sp_b)
            pair_scores.append(score)

        # Always accumulate coordination/symmetry for observability
        for ps in pair_scores:
            self._coordination_history.append(ps["coordination"])
            self._symmetry_history.append(ps["symmetry"])
        self._snapshot_count += 1

        # Only infer and override weights if not already loaded from disk
        if not self._maturity_inferred and self._snapshot_count >= MATURITY_SNAPSHOTS_REQUIRED:
            t = infer_maturity(self._coordination_history, self._symmetry_history)
            self._weights = interpolate_weights_from_t(t)
            self._maturity_inferred = True
            logger.info(
                "Maturity inferred: t=%.3f, weights=%s",
                t, {k: round(v, 3) for k, v in self._weights.items()},
            )

        composites = [p["composite"] for p in pair_scores]
        group_composite = sum(composites) / len(composites) if composites else 0.5

        self._recent_composites.append(group_composite)
        group_trend = self._compute_trend()

        payload = {
            "timestamp_ms": event.payload.get("timestamp_ms", 0),
            "window_duration_ms": 30000,
            "pair_scores": pair_scores,
            "group_composite": round(group_composite, 4),
            "group_trend": round(group_trend, 4),
            "disengaged_speakers": self._disengagement.disengaged_speakers(),
            "weights_applied": {k: round(v, 4) for k, v in self._weights.items()},
            "meeting_count": self._meeting_count,
            "inferred_maturity": round(
                infer_maturity(self._coordination_history, self._symmetry_history), 4
            ) if (self._maturity_inferred and self._coordination_history) else None,
        }

        rapport_event = Event(
            event_type=EventType.RAPPORT_SCORE,
            session_id=event.session_id,
            branch_id=event.branch_id,
            author=AUTHOR,
            evaluator=0.0,
            payload=payload,
        )
        await self._orchestrator.process_event(rapport_event)

    def _estimate_crqa_params(self) -> None:
        """Estimate CRQA parameters from accumulated F0 data.

        Uses the longest available F0 series. Targets 3-5% recurrence rate
        by adjusting radius. Embedding dim and delay use simple heuristics
        since full AMI/FNN requires scipy.spatial which is heavy.
        """
        if self._crqa_params_estimated:
            return

        longest = []
        for series in self._f0_series.values():
            if len(series) > len(longest):
                longest = list(series)

        if len(longest) < 30:
            return  # not enough data yet

        self._crqa_params_estimated = True
        arr = np.array(longest, dtype=np.float64)
        std = arr.std()
        if std > 0:
            # Radius targeting ~3-5% recurrence: start at 0.3*std, adjust
            self._crqa_radius = 0.3
        # Embedding dim: 3 is standard for prosodic signals
        self._crqa_embedding_dim = 3
        # Time delay: 2 is standard at 1s stride (every other sample)
        self._crqa_time_delay = 2
        logger.info(
            "CRQA params estimated: dim=%d, delay=%d, radius=%.2f",
            self._crqa_embedding_dim, self._crqa_time_delay, self._crqa_radius,
        )

    async def _compute_pair_rapport(self, sp_a: str, sp_b: str) -> dict:
        """Compute rapport components for a speaker pair."""
        self._estimate_crqa_params()
        arousal_a = list(self._arousal_series.get(sp_a, deque()))
        arousal_b = list(self._arousal_series.get(sp_b, deque()))
        attentiveness = self._pearson_clamped(arousal_a, arousal_b)

        valence_a = list(self._valence_series.get(sp_a, deque()))
        valence_b = list(self._valence_series.get(sp_b, deque()))
        positivity = self._valence_proximity(valence_a, valence_b)

        f0_a = list(self._f0_series.get(sp_a, deque()))
        f0_b = list(self._f0_series.get(sp_b, deque()))
        crqa = await self._crqa_router.compute(
            f0_a, f0_b,
            embedding_dim=self._crqa_embedding_dim,
            time_delay=self._crqa_time_delay,
            radius=self._crqa_radius,
        )
        coordination = min(1.0, crqa.det)

        te_a_to_b = compute_transfer_entropy(f0_a, f0_b, lag=1)
        te_b_to_a = compute_transfer_entropy(f0_b, f0_a, lag=1)
        symmetry = compute_symmetry(te_a_to_b, te_b_to_a)

        raw_composite = (
            self._weights["attentiveness"] * attentiveness
            + self._weights["positivity"] * positivity
            + self._weights["coordination"] * coordination
            + self._weights["symmetry"] * symmetry
        )

        damping = self._disengagement.get_pair_damping(sp_a, sp_b)
        composite = raw_composite * damping

        return {
            "speaker_a": sp_a,
            "speaker_b": sp_b,
            "attentiveness": round(attentiveness, 4),
            "positivity": round(positivity, 4),
            "coordination": round(coordination, 4),
            "symmetry": round(symmetry, 4),
            "composite": round(max(0.0, min(1.0, composite)), 4),
            "disengagement_damped": damping < 1.0,
            "surrogate_validated": crqa.surrogate_validated,
        }

    def _compute_trend(self) -> float:
        """Pearson correlation of recent composites vs time index."""
        values = list(self._recent_composites)
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values), dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        if y.std() < 1e-10:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr) if not math.isnan(corr) else 0.0

    @staticmethod
    def _pearson_clamped(a: list[float], b: list[float]) -> float:
        """Pearson correlation clamped to [0, 1]."""
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        arr_a = np.array(a[-n:], dtype=np.float64)
        arr_b = np.array(b[-n:], dtype=np.float64)
        if arr_a.std() < 1e-10 or arr_b.std() < 1e-10:
            return 0.0
        r = np.corrcoef(arr_a, arr_b)[0, 1]
        return max(0.0, float(r)) if not math.isnan(r) else 0.0

    @staticmethod
    def _valence_proximity(a: list[float], b: list[float]) -> float:
        """Inverted mean valence distance."""
        if not a or not b:
            return 0.5
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        return 1.0 - abs(mean_a - mean_b)
