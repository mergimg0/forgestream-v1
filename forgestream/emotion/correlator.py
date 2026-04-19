"""Correlates CLAIM events with temporally-aligned PROSODIC_FEATURE events.

Subscribes to EventBus. Maintains a recent window of PROSODIC_FEATURE events.
When a CLAIM arrives, finds the closest prosodic feature by timestamp and:
1. Enriches the claim with acoustic confidence adjustments
2. Detects significant emotional shifts → emits EMOTION_STATE events
3. Tracks per-speaker emotion trajectories
"""

from __future__ import annotations

from collections import deque
from uuid import UUID

from forgestream.events.schema import Event, EventType

AUTHOR = "emotion_correlator"

# Temporal alignment tolerance: claims within ±2s of a prosodic feature match
ALIGNMENT_TOLERANCE_MS = 2000

# Emotional shift threshold: arousal or valence delta must exceed this
SHIFT_THRESHOLD = 0.2


class EmotionCorrelator:
    """Aligns CLAIM events with PROSODIC_FEATURE events by timestamp.

    Subscribes to the EventBus via on_event(). Buffers recent prosodic
    features and correlates them with incoming claims. Emits EMOTION_STATE
    events when significant emotional shifts are detected.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator
        self._prosodic_buffer: deque[Event] = deque(maxlen=60)
        self._previous_prosodic: dict | None = None

    async def on_event(self, event: Event) -> None:
        """EventBus handler. Processes CLAIM and PROSODIC_FEATURE events."""
        if event.author == AUTHOR:
            return

        if event.event_type == EventType.PROSODIC_FEATURE:
            await self._handle_prosodic(event)
        elif event.event_type == EventType.CLAIM:
            await self._handle_claim(event)

    async def _handle_prosodic(self, event: Event) -> None:
        """Buffer prosodic feature and check for emotional shifts."""
        self._prosodic_buffer.append(event)

        current = event.payload
        if self._previous_prosodic is not None:
            shift = self._detect_emotional_shift(current, self._previous_prosodic)
            if shift is not None:
                shift_event = Event(
                    event_type=EventType.EMOTION_STATE,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=shift,
                    parent_id=event.id,
                )
                await self._orchestrator.process_event(shift_event)

        self._previous_prosodic = current

    async def _handle_claim(self, event: Event) -> None:
        """Find nearest prosodic feature for a claim (for future enrichment)."""
        timestamp_ms = event.payload.get("audio_timestamp")
        if timestamp_ms is None:
            return
        self._find_nearest_feature(timestamp_ms)

    def _find_nearest_feature(self, timestamp_ms: int) -> dict | None:
        """Find the PROSODIC_FEATURE closest to the given timestamp."""
        best = None
        best_delta = float("inf")
        for evt in self._prosodic_buffer:
            ts = evt.payload.get("timestamp_ms", 0)
            delta = abs(ts - timestamp_ms)
            if delta < best_delta and delta <= ALIGNMENT_TOLERANCE_MS:
                best_delta = delta
                best = evt.payload
        return best

    def _compute_confidence_adjustment(self, prosodic: dict) -> float:
        """Data-driven confidence adjustment replacing hardcoded tone markers.

        Returns a float in [-0.3, +0.3] to add to claim confidence.

        Positive signals (boost confidence):
        - High HNR (clear voice, confident delivery)
        - Low jitter (stable pitch, not stressed)

        Negative signals (lower confidence):
        - High jitter (vocal instability, stress)
        - Low HNR (breathy, uncertain)
        - High shimmer (amplitude instability)
        """
        hnr = prosodic.get("hnr", 15.0)
        jitter = prosodic.get("jitter_local", 0.02)
        shimmer = prosodic.get("shimmer_local", 0.04)
        arousal = prosodic.get("arousal", 0.5)

        # HNR contribution: high HNR = clear voice = confidence boost
        # Typical speech HNR: 10-25 dB. Normalize to [-0.15, +0.15]
        hnr_norm = max(-1.0, min(1.0, (hnr - 15.0) / 15.0))
        hnr_adj = hnr_norm * 0.15

        # Jitter contribution: high jitter = stress = negative
        # Typical jitter: 0.005-0.03. Above 0.03 = stressed
        jitter_norm = max(0.0, min(1.0, (jitter - 0.005) / 0.03))
        jitter_adj = -jitter_norm * 0.15

        # Shimmer contribution: high shimmer = instability = negative
        shimmer_norm = max(0.0, min(1.0, (shimmer - 0.02) / 0.06))
        shimmer_adj = -shimmer_norm * 0.10

        total = hnr_adj + jitter_adj + shimmer_adj
        return max(-0.3, min(0.3, total))

    def _detect_emotional_shift(
        self, current: dict, previous: dict
    ) -> dict | None:
        """Detect significant emotional shift between consecutive windows.

        Returns EMOTION_STATE payload dict if shift detected, None otherwise.
        Threshold: arousal_delta > 0.2 OR valence_delta > 0.2
        """
        arousal_delta = current.get("arousal", 0.5) - previous.get("arousal", 0.5)
        valence_delta = current.get("valence", 0.5) - previous.get("valence", 0.5)

        if abs(arousal_delta) < SHIFT_THRESHOLD and abs(valence_delta) < SHIFT_THRESHOLD:
            return None

        # Determine shift type based on arousal direction
        if arousal_delta > 0:
            shift_type = "onset"
        elif arousal_delta < -SHIFT_THRESHOLD:
            shift_type = "offset"
        else:
            shift_type = "sustained"

        return {
            "speaker_id": current.get("speaker_id", "unknown"),
            "timestamp_ms": current.get("timestamp_ms", 0),
            "shift_type": shift_type,
            "from_emotion": self._classify_emotion(previous),
            "to_emotion": self._classify_emotion(current),
            "arousal_delta": round(arousal_delta, 4),
            "valence_delta": round(valence_delta, 4),
            "trigger_claim_id": None,
            "confidence": min(1.0, abs(arousal_delta) + abs(valence_delta)),
        }

    @staticmethod
    def _classify_emotion(prosodic: dict) -> str:
        """Simple emotion label from dimensional values."""
        arousal = prosodic.get("arousal", 0.5)
        valence = prosodic.get("valence", 0.5)
        if arousal > 0.6 and valence > 0.6:
            return "excited"
        if arousal > 0.6 and valence <= 0.4:
            return "angry"
        if arousal <= 0.4 and valence > 0.6:
            return "calm"
        if arousal <= 0.4 and valence <= 0.4:
            return "sad"
        return "neutral"
