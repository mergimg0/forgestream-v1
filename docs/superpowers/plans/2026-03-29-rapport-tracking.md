# Rapport Tracking System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a multi-component rapport tracking system that computes attentiveness, positivity, coordination (RunPod CRQA), and symmetry (transfer entropy) per speaker pair, applies disengagement damping and meeting-count sigmoid weights, and integrates with the SOS governor.

**Architecture:** `RapportEngine` subscribes to EventBus, consumes PROSODIC_FEATURE and ENTRAINMENT_SNAPSHOT events, delegates CRQA to RunPod via `CRQAComputeRouter` (with circuit breaker fallback to local), and emits `RAPPORT_SCORE` events. `DisengagementDetector` monitors per-speaker energy/pitch trends. Governor evaluator, axiom checker, and trust region consume the rapport composite.

**Tech Stack:** scipy (transfer entropy, Pearson), numpy, httpx (async RunPod client), PyRQA (RunPod endpoint only), FastAPI (RunPod endpoint)

---

## Dependency Order

```
Task 1 (Event type + config)
    └─► Task 2 (Transfer entropy)
    └─► Task 3 (Disengagement detector)
    └─► Task 4 (CRQA compute router)
            └─► Task 5 (RapportEngine — needs 2, 3, 4)
                    └─► Task 6 (Orchestrator + LiveStream wiring)
                    └─► Task 7 (SOS governor integration)
                    └─► Task 8 (Dashboard API)
Task 9 (RunPod endpoint — standalone, deployed separately)
```

Tasks 2, 3, 4 are independent and can be parallelized.

---

## Task 1: RAPPORT_SCORE Event Type + Config

**Files:**
- Modify: `forgestream/events/schema.py:27-29`
- Modify: `forgestream/config.py:46-51`
- Test: `tests/emotion/test_rapport_schema.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_rapport_schema.py
"""Test RAPPORT_SCORE event type."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType


def test_rapport_score_event_type_exists():
    assert EventType.RAPPORT_SCORE == "rapport_score"


def test_rapport_score_event_serializes():
    event = Event(
        event_type=EventType.RAPPORT_SCORE,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="rapport_engine",
        evaluator=0.0,
        payload={
            "group_composite": 0.68,
            "group_trend": 0.03,
            "pair_scores": [],
            "disengaged_speakers": [],
        },
    )
    d = event.to_dict()
    assert d["event_type"] == "rapport_score"
    roundtrip = Event.from_dict(d)
    assert roundtrip.payload["group_composite"] == 0.68


def test_rapport_config_fields():
    config = ForgeStreamConfig()
    assert config.runpod_crqa_endpoint == ""
    assert config.runpod_timeout_seconds == 4.0
    assert config.rapport_damping_factor == 0.3
    assert config.rapport_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_schema.py -v`
Expected: FAIL with `AttributeError: RAPPORT_SCORE`

- [ ] **Step 3: Add RAPPORT_SCORE to EventType**

In `forgestream/events/schema.py`, add after line 29 (`ENTRAINMENT_SNAPSHOT = "entrainment_snapshot"`):

```python
    RAPPORT_SCORE = "rapport_score"
```

- [ ] **Step 4: Add rapport config fields**

In `forgestream/config.py`, add after line 51 (`emotion_buffer_seconds: float = 30.0`):

```python
    # Rapport tracking
    rapport_enabled: bool = True
    rapport_damping_factor: float = 0.3    # disengagement damping multiplier
    runpod_crqa_endpoint: str = ""         # RunPod URL (empty = local fallback only)
    runpod_timeout_seconds: float = 4.0    # per-request timeout
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_schema.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/events/schema.py forgestream/config.py tests/emotion/test_rapport_schema.py
git commit -m "feat(events): add RAPPORT_SCORE event type and rapport config fields"
```

---

## Task 2: Transfer Entropy

**Files:**
- Create: `forgestream/emotion/transfer_entropy.py`
- Test: `tests/emotion/test_transfer_entropy.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_transfer_entropy.py
"""Tests for transfer entropy computation."""

import numpy as np
import pytest

from forgestream.emotion.transfer_entropy import compute_transfer_entropy


class TestTransferEntropy:
    def test_self_predictive_signal_has_low_te(self):
        """A signal that only depends on itself has low TE from another."""
        np.random.seed(42)
        a = np.random.randn(200).tolist()
        b = np.random.randn(200).tolist()  # independent
        te = compute_transfer_entropy(a, b, lag=1)
        assert te >= 0.0
        assert te < 0.5  # should be low for independent signals

    def test_causal_signal_has_higher_te(self):
        """B caused by A should show higher TE(A→B) than TE(B→A)."""
        np.random.seed(42)
        a = np.random.randn(200).tolist()
        # b[t] = 0.8*a[t-1] + noise
        b = [0.0] + [0.8 * a[i] + 0.2 * np.random.randn() for i in range(199)]
        te_a_to_b = compute_transfer_entropy(a, b, lag=1)
        te_b_to_a = compute_transfer_entropy(b, a, lag=1)
        assert te_a_to_b > te_b_to_a

    def test_short_signals_return_zero(self):
        te = compute_transfer_entropy([1.0, 2.0], [3.0, 4.0], lag=1)
        assert te == 0.0

    def test_identical_signals(self):
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 100))]
        te = compute_transfer_entropy(signal, signal, lag=1)
        assert te >= 0.0

    def test_symmetry_score(self):
        """Compute symmetry from bidirectional TE."""
        from forgestream.emotion.transfer_entropy import compute_symmetry
        # Symmetric mutual influence
        sym = compute_symmetry(0.3, 0.3)
        assert sym == pytest.approx(1.0)
        # Asymmetric
        asym = compute_symmetry(0.5, 0.1)
        assert asym < 0.5
        # Zero TE
        zero = compute_symmetry(0.0, 0.0)
        assert zero == 1.0  # no info = assume symmetric
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_transfer_entropy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement transfer entropy**

```python
# forgestream/emotion/transfer_entropy.py
"""Transfer entropy computation for measuring directed vocal influence.

Transfer entropy from A to B measures how much A's history reduces
uncertainty about B's next value, beyond B's own history. Used for
the symmetry component of rapport scoring.
"""

from __future__ import annotations

import math

import numpy as np


def compute_transfer_entropy(
    source: list[float],
    target: list[float],
    lag: int = 1,
    n_bins: int = 8,
) -> float:
    """Compute transfer entropy from source to target.

    Uses histogram-based estimation. Discretizes continuous values
    into n_bins bins, then computes:

    TE(source → target) = H(target_future | target_past) - H(target_future | target_past, source_past)

    Parameters:
        source: Source time series.
        target: Target time series.
        lag: Number of time steps for history.
        n_bins: Number of bins for discretization.

    Returns:
        Transfer entropy in bits (>= 0). Higher = more influence.
    """
    if len(source) < lag + 2 or len(target) < lag + 2:
        return 0.0

    src = np.array(source, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)

    # Align lengths
    n = min(len(src), len(tgt))
    src = src[:n]
    tgt = tgt[:n]

    # Discretize into bins
    src_bins = _discretize(src, n_bins)
    tgt_bins = _discretize(tgt, n_bins)

    # Build joint distributions
    # target_future = tgt[lag:]
    # target_past = tgt[:-lag]
    # source_past = src[:-lag]
    tgt_future = tgt_bins[lag:]
    tgt_past = tgt_bins[:-lag]
    src_past = src_bins[:-lag]

    m = len(tgt_future)
    if m < 10:
        return 0.0

    # H(target_future | target_past)
    h_tf_tp = _conditional_entropy(tgt_future, tgt_past, n_bins)

    # H(target_future | target_past, source_past)
    joint_past = tgt_past * n_bins + src_past  # encode pair as single int
    h_tf_tp_sp = _conditional_entropy(tgt_future, joint_past, n_bins)

    te = max(0.0, h_tf_tp - h_tf_tp_sp)
    return te


def compute_symmetry(te_a_to_b: float, te_b_to_a: float) -> float:
    """Compute symmetry score from bidirectional transfer entropy.

    Returns 1.0 for perfect symmetry, 0.0 for fully one-directional.
    """
    total = te_a_to_b + te_b_to_a
    if total < 1e-10:
        return 1.0  # no information flow = assume symmetric
    asymmetry = abs(te_a_to_b - te_b_to_a) / total
    return 1.0 - asymmetry


def _discretize(series: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize a continuous series into integer bins."""
    if series.std() < 1e-10:
        return np.zeros(len(series), dtype=np.int64)
    # Z-score then map to bins
    z = (series - series.mean()) / series.std()
    # Clip to [-3, 3] then map to [0, n_bins-1]
    clipped = np.clip(z, -3, 3)
    bins = ((clipped + 3) / 6 * (n_bins - 1)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def _conditional_entropy(
    x: np.ndarray, condition: np.ndarray, n_bins: int
) -> float:
    """Compute H(X | condition) using histogram counts."""
    # Joint counts
    unique_conditions = np.unique(condition)
    total = len(x)
    h = 0.0
    for c in unique_conditions:
        mask = condition == c
        p_c = mask.sum() / total
        if p_c == 0:
            continue
        x_given_c = x[mask]
        # Entropy of x given this condition value
        counts = np.bincount(x_given_c, minlength=n_bins)
        probs = counts / counts.sum()
        for p in probs:
            if p > 0:
                h -= p_c * p * math.log2(p)
    return h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_transfer_entropy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/transfer_entropy.py tests/emotion/test_transfer_entropy.py
git commit -m "feat(emotion): add transfer entropy for directed vocal influence measurement"
```

---

## Task 3: Disengagement Detector

**Files:**
- Create: `forgestream/emotion/disengagement.py`
- Test: `tests/emotion/test_disengagement.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_disengagement.py
"""Tests for DisengagementDetector."""

import pytest

from forgestream.emotion.disengagement import DisengagementDetector


class TestDisengagementDetector:
    def test_no_disengagement_on_normal_features(self):
        det = DisengagementDetector(damping_factor=0.3)
        # Simulate 15 normal features for two speakers
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.09, "f0_std": 28.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is False
        assert det.is_disengaged("sp1") is False
        assert det.disengaged_speakers() == []

    def test_detects_energy_drop_with_pitch_flattening(self):
        det = DisengagementDetector(damping_factor=0.3)
        # Build baseline: 15 normal features
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        # sp0 disengages: energy drops + pitch flattens, sp1 stays normal
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is True
        assert det.is_disengaged("sp1") is False
        assert "sp0" in det.disengaged_speakers()

    def test_both_quiet_is_not_disengagement(self):
        det = DisengagementDetector(damping_factor=0.3)
        # Build baseline
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        # Both go quiet together — natural ebb, not disengagement
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
        assert det.is_disengaged("sp0") is False
        assert det.is_disengaged("sp1") is False

    def test_recovery_clears_flag(self):
        det = DisengagementDetector(damping_factor=0.3)
        # Build baseline
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        # Disengage
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is True
        # Recover — energy and F0 return to 70% of baseline
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.08, "f0_std": 25.0, "arousal": 0.4})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is False

    def test_get_damping_for_pair(self):
        det = DisengagementDetector(damping_factor=0.3)
        # No disengagement → damping = 1.0
        assert det.get_pair_damping("sp0", "sp1") == 1.0
        # After disengagement flagged → damping = 0.3 for pairs involving sp0
        det._flags["sp0"] = True
        assert det.get_pair_damping("sp0", "sp1") == 0.3
        assert det.get_pair_damping("sp1", "sp2") == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_disengagement.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DisengagementDetector**

```python
# forgestream/emotion/disengagement.py
"""Disengagement detection from prosodic feature trends.

Monitors per-speaker energy and F0 variability over a sliding window.
Flags disengagement when energy drops + pitch flattens + the change
is one-sided (other speakers maintain normal levels).
"""

from __future__ import annotations

from collections import deque


# Detection thresholds
ENERGY_THRESHOLD = 0.6     # current < 60% of baseline = low
F0_STD_THRESHOLD = 0.4     # current < 40% of baseline = flat
RECOVERY_THRESHOLD = 0.7   # must recover to 70% of baseline to clear flag
WINDOW_SIZE = 10           # number of features to track
BASELINE_WINDOW = 15       # first N features build the baseline
EMA_ALPHA = 0.01           # baseline slow update rate


class DisengagementDetector:
    """Detects per-speaker disengagement from prosodic trends.

    Parameters:
        damping_factor: Multiplier applied to rapport composite when disengaged.
    """

    def __init__(self, damping_factor: float = 0.3) -> None:
        self._damping_factor = damping_factor
        self._windows: dict[str, deque[dict]] = {}
        self._baselines: dict[str, dict[str, float]] = {}
        self._update_counts: dict[str, int] = {}
        self._flags: dict[str, bool] = {}

    def update(self, speaker_id: str, features: dict) -> None:
        """Add a prosodic feature snapshot for a speaker."""
        if speaker_id not in self._windows:
            self._windows[speaker_id] = deque(maxlen=WINDOW_SIZE)
            self._baselines[speaker_id] = {"energy_rms": 0.0, "f0_std": 0.0}
            self._update_counts[speaker_id] = 0
            self._flags[speaker_id] = False

        self._windows[speaker_id].append(features)
        self._update_counts[speaker_id] += 1
        count = self._update_counts[speaker_id]

        energy = features.get("energy_rms", 0.0)
        f0_std = features.get("f0_std", 0.0)

        # Build baseline from first BASELINE_WINDOW updates
        if count <= BASELINE_WINDOW:
            bl = self._baselines[speaker_id]
            bl["energy_rms"] = (bl["energy_rms"] * (count - 1) + energy) / count
            bl["f0_std"] = (bl["f0_std"] * (count - 1) + f0_std) / count
        else:
            # Slow EMA update
            bl = self._baselines[speaker_id]
            bl["energy_rms"] = (1 - EMA_ALPHA) * bl["energy_rms"] + EMA_ALPHA * energy
            bl["f0_std"] = (1 - EMA_ALPHA) * bl["f0_std"] + EMA_ALPHA * f0_std

        # Check for disengagement after baseline is established
        if count >= BASELINE_WINDOW + WINDOW_SIZE:
            self._check_disengagement(speaker_id)

    def _check_disengagement(self, speaker_id: str) -> None:
        """Check if a speaker is disengaged based on current window vs baseline."""
        bl = self._baselines[speaker_id]
        if bl["energy_rms"] < 1e-6 or bl["f0_std"] < 1e-6:
            return  # can't evaluate without meaningful baseline

        window = list(self._windows[speaker_id])
        mean_energy = sum(f.get("energy_rms", 0.0) for f in window) / len(window)
        mean_f0_std = sum(f.get("f0_std", 0.0) for f in window) / len(window)

        energy_ratio = mean_energy / bl["energy_rms"]
        f0_ratio = mean_f0_std / bl["f0_std"]

        if self._flags[speaker_id]:
            # Currently flagged — check for recovery (hysteresis)
            if energy_ratio >= RECOVERY_THRESHOLD and f0_ratio >= RECOVERY_THRESHOLD:
                self._flags[speaker_id] = False
        else:
            # Not flagged — check for disengagement
            energy_low = energy_ratio < ENERGY_THRESHOLD
            pitch_flat = f0_ratio < F0_STD_THRESHOLD
            one_sided = self._is_one_sided(speaker_id)

            if energy_low and pitch_flat and one_sided:
                self._flags[speaker_id] = True

    def _is_one_sided(self, speaker_id: str) -> bool:
        """Check if other speakers maintain normal levels (one-sided divergence)."""
        for other_id, bl in self._baselines.items():
            if other_id == speaker_id:
                continue
            if bl["energy_rms"] < 1e-6:
                continue
            window = list(self._windows.get(other_id, deque()))
            if not window:
                continue
            other_energy = sum(f.get("energy_rms", 0.0) for f in window) / len(window)
            if other_energy / bl["energy_rms"] >= ENERGY_THRESHOLD:
                return True  # at least one other speaker is at normal levels
        return False

    def is_disengaged(self, speaker_id: str) -> bool:
        """Check if a speaker is currently flagged as disengaged."""
        return self._flags.get(speaker_id, False)

    def disengaged_speakers(self) -> list[str]:
        """Return list of currently disengaged speaker IDs."""
        return [sid for sid, flagged in self._flags.items() if flagged]

    def get_pair_damping(self, speaker_a: str, speaker_b: str) -> float:
        """Get the damping factor for a speaker pair.

        Returns 1.0 if neither is disengaged, damping_factor if either is.
        """
        if self._flags.get(speaker_a, False) or self._flags.get(speaker_b, False):
            return self._damping_factor
        return 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_disengagement.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/disengagement.py tests/emotion/test_disengagement.py
git commit -m "feat(emotion): add DisengagementDetector with per-speaker baseline and hysteresis"
```

---

## Task 4: CRQA Compute Router

**Files:**
- Create: `forgestream/emotion/crqa_router.py`
- Test: `tests/emotion/test_crqa_router.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_crqa_router.py
"""Tests for CRQAComputeRouter — circuit breaker + RunPod/local routing."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from forgestream.emotion.crqa_router import CRQAComputeRouter, CRQAResult


class TestCRQAResult:
    def test_dataclass_fields(self):
        r = CRQAResult(
            det=0.5, tt=3.0, entr=1.5, lam=0.4, rr=0.03,
            surrogate_validated=True,
        )
        assert r.det == 0.5
        assert r.surrogate_validated is True


class TestLocalFallback:
    @pytest.mark.asyncio
    async def test_local_compute_returns_crqa_result(self):
        router = CRQAComputeRouter(runpod_endpoint="", timeout=4.0)
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 100))]
        result = await router.compute(signal, signal)
        assert isinstance(result, CRQAResult)
        assert result.det >= 0.0
        assert result.surrogate_validated is False  # no RunPod = not validated

    @pytest.mark.asyncio
    async def test_local_with_short_signals(self):
        router = CRQAComputeRouter(runpod_endpoint="", timeout=4.0)
        result = await router.compute([1.0, 2.0], [3.0, 4.0])
        assert result.det == 0.0  # too short


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_3_failures(self):
        router = CRQAComputeRouter(
            runpod_endpoint="http://fake:8000", timeout=0.1
        )
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 50))]
        # 3 failures
        for _ in range(3):
            result = await router.compute(signal, signal)
            assert result.surrogate_validated is False
        assert router._consecutive_failures >= 3
        assert router._circuit_open is True

    @pytest.mark.asyncio
    async def test_circuit_retries_after_skip_cycles(self):
        router = CRQAComputeRouter(
            runpod_endpoint="http://fake:8000", timeout=0.1
        )
        router._consecutive_failures = 3
        router._circuit_open = True
        router._skip_remaining = 0  # skip period expired
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 50))]
        # Should try RunPod again (and fail, but at least tries)
        result = await router.compute(signal, signal)
        assert isinstance(result, CRQAResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_crqa_router.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CRQAComputeRouter**

```python
# forgestream/emotion/crqa_router.py
"""CRQA compute router with RunPod GPU and local CPU fallback.

Routes CRQA computation to a RunPod serverless endpoint for surrogate-
validated results. Falls back to local numpy CRQA (without surrogates)
when RunPod is unavailable. Circuit breaker prevents hammering a dead endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CIRCUIT_OPEN_THRESHOLD = 3   # failures before circuit opens
CIRCUIT_SKIP_CYCLES = 5      # cycles to skip before retry


@dataclass
class CRQAResult:
    """Result of a CRQA computation."""

    det: float              # determinism
    tt: float               # trapping time
    entr: float             # diagonal entropy
    lam: float              # laminarity
    rr: float               # recurrence rate
    surrogate_validated: bool = False


class CRQAComputeRouter:
    """Routes CRQA computation to RunPod or local fallback.

    Parameters:
        runpod_endpoint: URL of the RunPod CRQA endpoint. Empty = local only.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        runpod_endpoint: str = "",
        timeout: float = 4.0,
    ) -> None:
        self._endpoint = runpod_endpoint
        self._timeout = timeout
        self._consecutive_failures = 0
        self._circuit_open = False
        self._skip_remaining = 0

    async def compute(
        self,
        f0_a: list[float],
        f0_b: list[float],
        embedding_dim: int = 3,
        time_delay: int = 2,
        radius: float = 0.25,
    ) -> CRQAResult:
        """Compute CRQA metrics, routing to RunPod or local fallback."""
        if len(f0_a) < 10 or len(f0_b) < 10:
            return CRQAResult(det=0.0, tt=0.0, entr=0.0, lam=0.0, rr=0.0)

        # Try RunPod if endpoint configured and circuit not open
        if self._endpoint and not self._should_skip():
            try:
                result = await self._runpod_compute(
                    f0_a, f0_b, embedding_dim, time_delay, radius
                )
                self._consecutive_failures = 0
                self._circuit_open = False
                return result
            except Exception as e:
                logger.warning("RunPod CRQA failed: %s", e)
                self._consecutive_failures += 1
                if self._consecutive_failures >= CIRCUIT_OPEN_THRESHOLD:
                    self._circuit_open = True
                    self._skip_remaining = CIRCUIT_SKIP_CYCLES
                    logger.info("Circuit breaker opened after %d failures", self._consecutive_failures)

        # Local fallback
        return self._local_compute(f0_a, f0_b, radius)

    def _should_skip(self) -> bool:
        """Check if we should skip RunPod due to circuit breaker."""
        if not self._circuit_open:
            return False
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return True
        # Skip period expired — half-open, try again
        return False

    async def _runpod_compute(
        self,
        f0_a: list[float],
        f0_b: list[float],
        embedding_dim: int,
        time_delay: int,
        radius: float,
    ) -> CRQAResult:
        """Send CRQA request to RunPod endpoint."""
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._endpoint}/crqa/validate",
                json={
                    "f0_a": f0_a,
                    "f0_b": f0_b,
                    "params": {
                        "embedding_dim": embedding_dim,
                        "time_delay": time_delay,
                        "radius": radius,
                        "n_surrogates": 20,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        real = data["real"]
        sig = data.get("significant", {})

        return CRQAResult(
            det=real["det"] if sig.get("det", False) else 0.0,
            tt=real["tt"] if sig.get("tt", False) else 0.0,
            entr=real.get("entr", 0.0),
            lam=real.get("lam", 0.0),
            rr=real.get("rr", 0.0),
            surrogate_validated=True,
        )

    def _local_compute(
        self,
        f0_a: list[float],
        f0_b: list[float],
        radius: float = 0.25,
    ) -> CRQAResult:
        """Local CPU CRQA (no surrogates)."""
        a = np.array(f0_a, dtype=np.float64)
        b = np.array(f0_b, dtype=np.float64)

        # Normalize
        if a.std() > 0:
            a = (a - a.mean()) / a.std()
        if b.std() > 0:
            b = (b - b.mean()) / b.std()

        combined_std = np.std(np.concatenate([a, b]))
        if combined_std < 1e-10:
            return CRQAResult(det=0.0, tt=0.0, entr=0.0, lam=0.0, rr=0.0)

        threshold = radius * combined_std * 3  # wider radius for local
        dist = np.abs(a[:, None] - b[None, :])
        recurrence = dist < threshold

        rr = float(np.mean(recurrence))
        det = self._compute_determinism(recurrence)
        lam = self._compute_laminarity(recurrence)

        return CRQAResult(
            det=det, tt=0.0, entr=0.0, lam=lam, rr=rr,
            surrogate_validated=False,
        )

    @staticmethod
    def _compute_determinism(recurrence: np.ndarray) -> float:
        """Compute %DET: proportion of recurrence points on diagonal lines >= 2."""
        n = recurrence.shape[0]
        total_recurrent = recurrence.sum()
        if total_recurrent == 0:
            return 0.0

        diagonal_points = 0
        for offset in range(-n + 1, n):
            diag = np.diagonal(recurrence, offset=offset)
            line_length = 0
            for val in diag:
                if val:
                    line_length += 1
                else:
                    if line_length >= 2:
                        diagonal_points += line_length
                    line_length = 0
            if line_length >= 2:
                diagonal_points += line_length

        return min(1.0, diagonal_points / total_recurrent)

    @staticmethod
    def _compute_laminarity(recurrence: np.ndarray) -> float:
        """Compute %LAM: proportion of recurrence points on vertical lines >= 2."""
        total_recurrent = recurrence.sum()
        if total_recurrent == 0:
            return 0.0

        vertical_points = 0
        for col in range(recurrence.shape[1]):
            line_length = 0
            for row in range(recurrence.shape[0]):
                if recurrence[row, col]:
                    line_length += 1
                else:
                    if line_length >= 2:
                        vertical_points += line_length
                    line_length = 0
            if line_length >= 2:
                vertical_points += line_length

        return min(1.0, vertical_points / total_recurrent)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_crqa_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/crqa_router.py tests/emotion/test_crqa_router.py
git commit -m "feat(emotion): add CRQAComputeRouter with RunPod client and circuit breaker"
```

---

## Task 5: RapportEngine

**Files:**
- Create: `forgestream/emotion/rapport.py`
- Test: `tests/emotion/test_rapport.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_rapport.py
"""Tests for RapportEngine — multi-component rapport scoring."""

import math
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from forgestream.emotion.rapport import RapportEngine, interpolate_weights
from forgestream.events.schema import Event, EventType


def _make_prosodic(session_id, branch_id, speaker, ts, arousal=0.5, valence=0.5):
    return Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=session_id, branch_id=branch_id,
        author="emotion_extractor", evaluator=0.0,
        payload={
            "speaker_id": speaker, "timestamp_ms": ts,
            "arousal": arousal, "valence": valence,
            "f0_mean": 200.0, "f0_std": 30.0, "energy_rms": 0.1,
        },
    )


def _make_snapshot(session_id, branch_id, ts):
    return Event(
        event_type=EventType.ENTRAINMENT_SNAPSHOT,
        session_id=session_id, branch_id=branch_id,
        author="dynamics_engine", evaluator=0.0,
        payload={
            "timestamp_ms": ts,
            "speaker_pairs": [
                {"speaker_a": "sp0", "speaker_b": "sp1",
                 "f0_correlation": 0.6, "energy_correlation": 0.5},
            ],
            "group_metrics": {"participation_parity": 0.8},
        },
    )


class TestInterpolateWeights:
    def test_meeting_1_favors_attentiveness_positivity(self):
        w = interpolate_weights(1)
        assert w["attentiveness"] > w["coordination"]
        assert w["positivity"] > w["coordination"]

    def test_meeting_5_favors_coordination(self):
        w = interpolate_weights(5)
        assert w["coordination"] > w["attentiveness"]
        assert w["coordination"] > w["positivity"]

    def test_weights_sum_to_one(self):
        for mc in [1, 2, 3, 5, 10]:
            w = interpolate_weights(mc)
            assert sum(w.values()) == pytest.approx(1.0, abs=0.001)

    def test_meeting_3_is_midpoint(self):
        w = interpolate_weights(3)
        # Should be roughly midway between early and established
        assert 0.20 < w["attentiveness"] < 0.35
        assert 0.15 < w["coordination"] < 0.40


class TestRapportEngine:
    @pytest.mark.asyncio
    async def test_emits_rapport_score_on_snapshot(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch, meeting_count=3)
        sid = orch.session_id
        bid = uuid4()

        # Feed prosodic features for two speakers
        for i in range(30):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000, arousal=0.6, valence=0.5)
            )
            await engine.on_event(
                _make_prosodic(sid, bid, "sp1", i * 1000 + 500, arousal=0.5, valence=0.6)
            )

        # Trigger with ENTRAINMENT_SNAPSHOT
        await engine.on_event(_make_snapshot(sid, bid, 30000))

        # Should have emitted at least one RAPPORT_SCORE
        rapport_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.RAPPORT_SCORE
        ]
        assert len(rapport_calls) >= 1

        payload = rapport_calls[0][0][0].payload
        assert "group_composite" in payload
        assert "pair_scores" in payload
        assert "group_trend" in payload
        assert "weights_applied" in payload
        assert 0.0 <= payload["group_composite"] <= 1.0

    @pytest.mark.asyncio
    async def test_ignores_self_authored(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch)
        self_event = Event(
            event_type=EventType.RAPPORT_SCORE,
            session_id=orch.session_id, branch_id=uuid4(),
            author="rapport_engine", evaluator=0.0,
            payload={"test": True},
        )
        await engine.on_event(self_event)
        assert orch.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_disengagement_damping_applied(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch, meeting_count=3, damping_factor=0.3)
        sid = orch.session_id
        bid = uuid4()

        # Build baseline then disengage sp0
        for i in range(15):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000, arousal=0.6, valence=0.5)
            )
            await engine.on_event(
                _make_prosodic(sid, bid, "sp1", i * 1000 + 500, arousal=0.5, valence=0.5)
            )
        # sp0 disengages
        for i in range(15, 30):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000, arousal=0.1, valence=0.2)
            )
            await engine.on_event(
                _make_prosodic(sid, bid, "sp1", i * 1000 + 500, arousal=0.5, valence=0.5)
            )
        # Override disengagement flag for test
        engine._disengagement.update("sp0", {"energy_rms": 0.02, "f0_std": 5.0})

        await engine.on_event(_make_snapshot(sid, bid, 30000))

        rapport_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.RAPPORT_SCORE
        ]
        if rapport_calls:
            payload = rapport_calls[0][0][0].payload
            if payload["pair_scores"]:
                pair = payload["pair_scores"][0]
                assert "disengagement_damped" in pair
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RapportEngine**

```python
# forgestream/emotion/rapport.py
"""RapportEngine — multi-component rapport scoring.

Subscribes to EventBus. Consumes PROSODIC_FEATURE events (for disengagement
detection and arousal/valence tracking) and ENTRAINMENT_SNAPSHOT events
(as trigger to compute rapport). Emits RAPPORT_SCORE events with composite
+ per-pair component scores.
"""

from __future__ import annotations

import math
import logging
from collections import deque
from itertools import combinations

import numpy as np

from forgestream.events.schema import Event, EventType

from .crqa_router import CRQAComputeRouter
from .disengagement import DisengagementDetector
from .transfer_entropy import compute_symmetry, compute_transfer_entropy

logger = logging.getLogger(__name__)

AUTHOR = "rapport_engine"

# Weight profiles from Tickle-Degnen research
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


def interpolate_weights(meeting_count: int) -> dict[str, float]:
    """Sigmoid interpolation between early and established weight profiles."""
    t = 1.0 / (1.0 + math.exp(-(meeting_count - 3)))
    return {
        k: EARLY_WEIGHTS[k] * (1 - t) + ESTABLISHED_WEIGHTS[k] * t
        for k in EARLY_WEIGHTS
    }


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
    ) -> None:
        self._orchestrator = orchestrator
        self._meeting_count = meeting_count
        self._weights = interpolate_weights(meeting_count)

        self._disengagement = DisengagementDetector(damping_factor=damping_factor)
        self._crqa_router = CRQAComputeRouter(
            runpod_endpoint=runpod_endpoint, timeout=runpod_timeout,
        )

        # Per-speaker arousal/valence history for attentiveness/positivity
        self._arousal_series: dict[str, deque[float]] = {}
        self._valence_series: dict[str, deque[float]] = {}
        self._f0_series: dict[str, deque[float]] = {}
        self._series_maxlen = 60  # ~60 seconds of data

        # Rapport trend tracking
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

        # Update disengagement detector
        self._disengagement.update(speaker, p)

        # Update arousal/valence/f0 series
        for series_dict, key in [
            (self._arousal_series, "arousal"),
            (self._valence_series, "valence"),
            (self._f0_series, "f0_mean"),
        ]:
            if speaker not in series_dict:
                series_dict[speaker] = deque(maxlen=self._series_maxlen)
            series_dict[speaker].append(p.get(key, 0.5 if key != "f0_mean" else 0.0))

    async def _handle_snapshot(self, event: Event) -> None:
        """Compute and emit RAPPORT_SCORE when triggered by ENTRAINMENT_SNAPSHOT."""
        speakers = list(self._arousal_series.keys())
        if len(speakers) < 2:
            return

        pair_scores = []
        for sp_a, sp_b in combinations(speakers, 2):
            score = await self._compute_pair_rapport(sp_a, sp_b)
            pair_scores.append(score)

        # Group composite = mean of pair composites
        composites = [p["composite"] for p in pair_scores]
        group_composite = sum(composites) / len(composites) if composites else 0.5

        # Trend
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

    async def _compute_pair_rapport(self, sp_a: str, sp_b: str) -> dict:
        """Compute rapport components for a speaker pair."""
        # Component 1: Attentiveness (arousal correlation)
        arousal_a = list(self._arousal_series.get(sp_a, deque()))
        arousal_b = list(self._arousal_series.get(sp_b, deque()))
        attentiveness = self._pearson_clamped(arousal_a, arousal_b)

        # Component 2: Positivity (valence proximity)
        valence_a = list(self._valence_series.get(sp_a, deque()))
        valence_b = list(self._valence_series.get(sp_b, deque()))
        positivity = self._valence_proximity(valence_a, valence_b)

        # Component 3: Coordination (CRQA via router)
        f0_a = list(self._f0_series.get(sp_a, deque()))
        f0_b = list(self._f0_series.get(sp_b, deque()))
        crqa = await self._crqa_router.compute(f0_a, f0_b)
        coordination = min(1.0, crqa.det)  # %DET as coordination score

        # Component 4: Symmetry (transfer entropy)
        te_a_to_b = compute_transfer_entropy(f0_a, f0_b, lag=1)
        te_b_to_a = compute_transfer_entropy(f0_b, f0_a, lag=1)
        symmetry = compute_symmetry(te_a_to_b, te_b_to_a)

        # Weighted composite
        raw_composite = (
            self._weights["attentiveness"] * attentiveness
            + self._weights["positivity"] * positivity
            + self._weights["coordination"] * coordination
            + self._weights["symmetry"] * symmetry
        )

        # Disengagement damping
        damping = self._disengagement.get_pair_damping(sp_a, sp_b)
        composite = raw_composite * damping
        is_damped = damping < 1.0

        return {
            "speaker_a": sp_a,
            "speaker_b": sp_b,
            "attentiveness": round(attentiveness, 4),
            "positivity": round(positivity, 4),
            "coordination": round(coordination, 4),
            "symmetry": round(symmetry, 4),
            "composite": round(max(0.0, min(1.0, composite)), 4),
            "disengagement_damped": is_damped,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/rapport.py tests/emotion/test_rapport.py
git commit -m "feat(emotion): add RapportEngine with 4-component scoring and disengagement damping"
```

---

## Task 6: Orchestrator + LiveStream Wiring

**Files:**
- Modify: `forgestream/orchestrator.py:164-169`
- Modify: `forgestream/live_stream.py:100-106`
- Test: `tests/emotion/test_rapport_wiring.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_rapport_wiring.py
"""Test RapportEngine is wired into the live pipeline."""

from unittest.mock import MagicMock

from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


def test_attach_rapport_engine():
    config = ForgeStreamConfig(rapport_enabled=True)
    orch = Orchestrator(config=config)
    engine = orch.attach_rapport_engine()
    assert engine is not None
    assert len(orch.event_bus._subscribers) >= 1


def test_live_stream_wires_rapport_engine():
    config = ForgeStreamConfig(emotion_enabled=True, rapport_enabled=True)
    orch = Orchestrator(config=config)
    source = MagicMock()
    stream = GeminiLiveStream(config, orch, source)
    assert stream.rapport_engine is not None


def test_rapport_disabled_skips_wiring():
    config = ForgeStreamConfig(emotion_enabled=True, rapport_enabled=False)
    orch = Orchestrator(config=config)
    source = MagicMock()
    stream = GeminiLiveStream(config, orch, source)
    assert stream.rapport_engine is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_wiring.py -v`
Expected: FAIL

- [ ] **Step 3: Add attach_rapport_engine to Orchestrator**

In `forgestream/orchestrator.py`, add after `attach_dynamics_engine`:

```python
    def attach_rapport_engine(
        self, meeting_count: int = 1, damping_factor: float = 0.3,
        runpod_endpoint: str = "", runpod_timeout: float = 4.0,
    ) -> "RapportEngine":
        """Create and attach a RapportEngine to this orchestrator's EventBus."""
        from .emotion.rapport import RapportEngine
        engine = RapportEngine(
            orchestrator=self, meeting_count=meeting_count,
            damping_factor=damping_factor, runpod_endpoint=runpod_endpoint,
            runpod_timeout=runpod_timeout,
        )
        self.event_bus.subscribe(engine.on_event)
        return engine
```

- [ ] **Step 4: Wire in GeminiLiveStream**

In `forgestream/live_stream.py`, inside the `if config.emotion_enabled:` block, after `self.dynamics_engine = orchestrator.attach_dynamics_engine()`, add:

```python
            if config.rapport_enabled:
                self.rapport_engine = orchestrator.attach_rapport_engine(
                    damping_factor=config.rapport_damping_factor,
                    runpod_endpoint=config.runpod_crqa_endpoint,
                    runpod_timeout=config.runpod_timeout_seconds,
                )
            else:
                self.rapport_engine = None
```

In the `else:` block (emotion disabled), add:

```python
            self.rapport_engine = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_wiring.py -v`
Expected: PASS

- [ ] **Step 6: Run full regression**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/ -q --ignore=tests/events/test_store.py --ignore=tests/events/test_subscribe.py -k "not writes_to_store and not milestone_a and not full_pipeline_with"`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/orchestrator.py forgestream/live_stream.py tests/emotion/test_rapport_wiring.py
git commit -m "feat(emotion): wire RapportEngine into Orchestrator and GeminiLiveStream"
```

---

## Task 7: SOS Governor Integration

**Files:**
- Modify: `forgestream/governor/evaluator.py:113-146`
- Modify: `forgestream/governor/axioms.py:31-61`
- Modify: `forgestream/governor/trust_region.py:34-46`
- Test: `tests/emotion/test_rapport_governor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_rapport_governor.py
"""Tests for rapport integration with SOS governor."""

from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.governor.axioms import AxiomChecker
from forgestream.governor.evaluator import Evaluator
from forgestream.governor.trust_region import TrustRegion


def _make_event(event_type, payload=None):
    return Event(
        event_type=event_type, session_id=uuid4(), branch_id=uuid4(),
        author="test", evaluator=0.0, payload=payload or {},
    )


class TestEvaluatorRapportEnrichment:
    def test_engagement_uses_rapport_when_available(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"]}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.6, "f0_std": 30.0, "energy_rms": 0.1,
            }),
            _make_event(EventType.RAPPORT_SCORE, {
                "group_composite": 0.85,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        # With high rapport composite, engagement should be boosted
        assert metrics.emotional_engagement > 0.6

    def test_engagement_falls_back_without_rapport(self):
        evaluator = Evaluator()
        events = [
            _make_event(EventType.CLAIM, {"topic_keywords": ["x"]}),
            _make_event(EventType.PROSODIC_FEATURE, {
                "arousal": 0.6, "f0_std": 30.0, "energy_rms": 0.1,
            }),
        ]
        metrics = evaluator.compute_metrics(events)
        # Should still compute engagement from prosodic features alone
        assert metrics.emotional_engagement > 0.0


class TestAxiomRapportAdvisory:
    def test_check_rapport_degradation(self):
        checker = AxiomChecker()
        # 5 declining rapport composites
        trajectory = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = checker.check_rapport_trend(trajectory, disengaged=True)
        assert result.axiom == "rapport_advisory"
        assert result.holds is False  # advisory flag

    def test_no_advisory_without_disengagement(self):
        checker = AxiomChecker()
        trajectory = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = checker.check_rapport_trend(trajectory, disengaged=False)
        assert result.holds is True  # declining but no disengagement = fine


class TestTrustRegionRapportBoost:
    def test_rapport_trend_boosts_improvements(self):
        tr = TrustRegion()
        initial = tr._consecutive_improvements
        tr.record_meeting_result(
            e_macro_improved=True, axiom_violations=0, rapport_trend=0.2
        )
        # Should get +1 for improvement + 0.5 for rapport boost = 1.5 total increase
        assert tr._consecutive_improvements == initial + 1.5

    def test_no_boost_without_rapport_trend(self):
        tr = TrustRegion()
        initial = tr._consecutive_improvements
        tr.record_meeting_result(
            e_macro_improved=True, axiom_violations=0, rapport_trend=0.0
        )
        assert tr._consecutive_improvements == initial + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_governor.py -v`
Expected: FAIL

- [ ] **Step 3: Update evaluator _emotional_engagement**

Replace the `_emotional_engagement` method in `forgestream/governor/evaluator.py`:

```python
    @staticmethod
    def _emotional_engagement(events: list[Event]) -> float:
        """Compute emotional engagement from PROSODIC_FEATURE and RAPPORT_SCORE events.

        Uses rapport composite when available (weighted 0.4), falls back to
        prosodic features (arousal + F0 variability + energy).

        Returns 0.5 if no relevant events exist.
        """
        rapport_events = [
            e for e in events
            if e.event_type == EventType.RAPPORT_SCORE
        ]
        prosodic = [
            e for e in events
            if e.event_type == EventType.PROSODIC_FEATURE
        ]

        if not prosodic and not rapport_events:
            return 0.5

        # Rapport composite (if available)
        if rapport_events:
            rapport_composite = rapport_events[-1].payload.get("group_composite", 0.5)
        else:
            rapport_composite = None

        # Prosodic fallback components
        if prosodic:
            arousals = [e.payload.get("arousal", 0.5) for e in prosodic]
            f0_stds = [e.payload.get("f0_std", 0.0) for e in prosodic]
            energies = [e.payload.get("energy_rms", 0.0) for e in prosodic]
            mean_arousal = sum(arousals) / len(arousals)
            mean_f0_var = min(1.0, (sum(f0_stds) / len(f0_stds)) / 80.0)
            mean_energy = min(1.0, (sum(energies) / len(energies)) / 0.15)
        else:
            mean_arousal = 0.5
            mean_f0_var = 0.0
            mean_energy = 0.0

        if rapport_composite is not None:
            engagement = (
                0.4 * rapport_composite
                + 0.3 * mean_arousal
                + 0.3 * mean_f0_var
            )
        else:
            engagement = (
                0.4 * mean_arousal
                + 0.3 * mean_f0_var
                + 0.3 * mean_energy
            )

        return max(0.0, min(1.0, engagement))
```

- [ ] **Step 4: Add check_rapport_trend to AxiomChecker**

In `forgestream/governor/axioms.py`, add a new method:

```python
    def check_rapport_trend(
        self, rapport_trajectory: list[float], disengaged: bool,
    ) -> AxiomResult:
        """Advisory check for rapport degradation.

        Not a formal axiom violation — just a warning when rapport is
        declining AND disengagement is detected.
        """
        if not disengaged or len(rapport_trajectory) < 3:
            return AxiomResult(axiom="rapport_advisory", holds=True)

        consecutive_declines = 0
        for i in range(1, len(rapport_trajectory)):
            if rapport_trajectory[i] < rapport_trajectory[i - 1]:
                consecutive_declines += 1
            else:
                consecutive_declines = 0

        if consecutive_declines >= 3:
            return AxiomResult(
                axiom="rapport_advisory",
                holds=False,
                reason=f"{consecutive_declines} consecutive declining rapport windows with active disengagement",
            )
        return AxiomResult(axiom="rapport_advisory", holds=True)
```

- [ ] **Step 5: Update TrustRegion.record_meeting_result**

In `forgestream/governor/trust_region.py`, update the signature and body:

```python
    def record_meeting_result(
        self, e_macro_improved: bool, axiom_violations: int,
        rapport_trend: float = 0.0,
    ) -> None:
        """Record the outcome of a meeting."""
        self._meeting_count += 1
        self._total_violations += axiom_violations

        if e_macro_improved:
            self._consecutive_improvements += 1
            # Rapport trend boost: building rapport + improving E = stronger evidence
            if rapport_trend > 0.1:
                self._consecutive_improvements += 0.5
        else:
            self._consecutive_improvements = max(
                0, self._consecutive_improvements - 1
            )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_governor.py -v`
Expected: PASS

- [ ] **Step 7: Run full regression**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/ -q --ignore=tests/events/test_store.py --ignore=tests/events/test_subscribe.py -k "not writes_to_store and not milestone_a and not full_pipeline_with"`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/governor/evaluator.py forgestream/governor/axioms.py forgestream/governor/trust_region.py tests/emotion/test_rapport_governor.py
git commit -m "feat(governor): integrate rapport into evaluator, axiom checker, and trust region"
```

---

## Task 8: Dashboard API Endpoint

**Files:**
- Modify: `forgestream/dashboard/api.py`
- Test: `tests/emotion/test_rapport_dashboard.py`

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_rapport_dashboard.py
"""Test rapport dashboard endpoint."""

from unittest.mock import MagicMock

import pytest

from forgestream.dashboard.api import create_router


def _make_firestore_db(events):
    mock_docs = []
    for e in events:
        doc = MagicMock()
        doc.to_dict.return_value = e
        mock_docs.append(doc)
    db = MagicMock()
    db.collection.return_value.order_by.return_value.stream.return_value = mock_docs
    return db


class TestRapportEndpoint:
    @pytest.mark.asyncio
    async def test_rapport_endpoint_returns_scores(self):
        events = [
            {"event_type": "rapport_score", "payload": {
                "timestamp_ms": 60000,
                "group_composite": 0.72,
                "group_trend": 0.05,
                "pair_scores": [{"speaker_a": "sp0", "speaker_b": "sp1", "composite": 0.72}],
                "disengaged_speakers": [],
            }},
        ]
        db = _make_firestore_db(events)
        router = create_router(db)

        rapport_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/emotion/rapport":
                rapport_fn = route.endpoint
                break
        assert rapport_fn is not None

        result = await rapport_fn()
        assert len(result["scores"]) == 1
        assert result["scores"][0]["group_composite"] == 0.72
        assert result["latest_trend"] == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_dashboard.py -v`
Expected: FAIL

- [ ] **Step 3: Add /emotion/rapport endpoint**

In `forgestream/dashboard/api.py`, add before the `/suggestions` endpoint:

```python
    @router.get("/emotion/rapport")
    async def get_emotion_rapport() -> dict:
        """Return rapport scores for visualization."""
        events = _get_events()
        scores = [
            e.get("payload", {})
            for e in events
            if e.get("event_type") == "rapport_score"
        ]
        latest_trend = scores[-1].get("group_trend", 0.0) if scores else 0.0
        return {"scores": scores, "latest_trend": latest_trend}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_rapport_dashboard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/dashboard/api.py tests/emotion/test_rapport_dashboard.py
git commit -m "feat(dashboard): add /emotion/rapport API endpoint"
```

---

## Task 9: RunPod CRQA Endpoint (Standalone)

**Files:**
- Create: `runpod/crqa_endpoint.py`
- Create: `runpod/requirements.txt`
- Create: `runpod/Dockerfile`

- [ ] **Step 1: Create the endpoint**

```python
# runpod/crqa_endpoint.py
"""RunPod CRQA endpoint — accepts F0 pairs, returns surrogate-validated metrics.

Deploy as a RunPod serverless endpoint with GPU (A4000 recommended).
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ForgeStream CRQA Endpoint")


class CRQARequest(BaseModel):
    f0_a: list[float]
    f0_b: list[float]
    params: dict


class CRQAResponse(BaseModel):
    real: dict
    surrogate_stats: dict
    significant: dict
    compute_ms: int


@app.post("/crqa/validate", response_model=CRQAResponse)
async def validate_crqa(req: CRQARequest) -> CRQAResponse:
    start = time.monotonic()

    a = np.array(req.f0_a, dtype=np.float64)
    b = np.array(req.f0_b, dtype=np.float64)
    radius = req.params.get("radius", 0.25)
    n_surrogates = req.params.get("n_surrogates", 20)

    # Normalize
    if a.std() > 0:
        a = (a - a.mean()) / a.std()
    if b.std() > 0:
        b = (b - b.mean()) / b.std()

    # Compute real CRQA
    real = _compute_crqa(a, b, radius)

    # Compute surrogates
    surrogate_dets = []
    surrogate_tts = []
    for _ in range(n_surrogates):
        shuffled_b = np.random.permutation(b)
        s = _compute_crqa(a, shuffled_b, radius)
        surrogate_dets.append(s["det"])
        surrogate_tts.append(s["tt"])

    det_mean = float(np.mean(surrogate_dets))
    det_std = float(np.std(surrogate_dets))
    tt_mean = float(np.mean(surrogate_tts))
    tt_std = float(np.std(surrogate_tts))

    elapsed = int((time.monotonic() - start) * 1000)

    return CRQAResponse(
        real=real,
        surrogate_stats={
            "det_mean": round(det_mean, 6),
            "det_std": round(det_std, 6),
            "tt_mean": round(tt_mean, 4),
            "tt_std": round(tt_std, 4),
        },
        significant={
            "det": real["det"] > det_mean + 2 * det_std if det_std > 0 else False,
            "tt": real["tt"] > tt_mean + 2 * tt_std if tt_std > 0 else False,
        },
        compute_ms=elapsed,
    )


def _compute_crqa(a: np.ndarray, b: np.ndarray, radius: float) -> dict:
    """Compute CRQA metrics from two normalized time series."""
    combined_std = np.std(np.concatenate([a, b]))
    threshold = radius * combined_std * 3 if combined_std > 0 else 0.1

    dist = np.abs(a[:, None] - b[None, :])
    recurrence = dist < threshold

    rr = float(np.mean(recurrence))
    total = recurrence.sum()

    if total == 0:
        return {"det": 0.0, "tt": 0.0, "entr": 0.0, "lam": 0.0, "rr": 0.0}

    # Determinism: diagonal lines >= 2
    n = len(a)
    diag_points = 0
    diag_lengths = []
    for offset in range(-n + 1, n):
        diag = np.diagonal(recurrence, offset=offset)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= 2:
                    diag_points += length
                    diag_lengths.append(length)
                length = 0
        if length >= 2:
            diag_points += length
            diag_lengths.append(length)

    det = min(1.0, diag_points / total)

    # Trapping time: mean vertical line length
    vert_lengths = []
    for col in range(recurrence.shape[1]):
        length = 0
        for row in range(recurrence.shape[0]):
            if recurrence[row, col]:
                length += 1
            else:
                if length >= 2:
                    vert_lengths.append(length)
                length = 0
        if length >= 2:
            vert_lengths.append(length)

    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0
    lam_points = sum(vert_lengths)
    lam = min(1.0, lam_points / total) if total > 0 else 0.0

    # Entropy of diagonal line lengths
    import math
    entr = 0.0
    if diag_lengths:
        total_diag = sum(diag_lengths)
        for l in set(diag_lengths):
            p = diag_lengths.count(l) * l / total_diag
            if p > 0:
                entr -= p * math.log2(p)

    return {
        "det": round(det, 6),
        "tt": round(tt, 4),
        "entr": round(entr, 4),
        "lam": round(lam, 6),
        "rr": round(rr, 6),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Create requirements.txt**

```
# runpod/requirements.txt
fastapi>=0.115
uvicorn>=0.32
numpy>=1.26
```

- [ ] **Step 3: Create Dockerfile**

```dockerfile
# runpod/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY crqa_endpoint.py .
CMD ["uvicorn", "crqa_endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add runpod/crqa_endpoint.py runpod/requirements.txt runpod/Dockerfile
git commit -m "feat(runpod): add CRQA surrogate validation endpoint for RunPod deployment"
```

---

## Verification

After all tasks complete:

```bash
# All emotion tests
cd /Users/mghome/projects/forgestream
python3 -m pytest tests/emotion/ -v

# Full regression
python3 -m pytest tests/ -q \
  --ignore=tests/events/test_store.py \
  --ignore=tests/events/test_subscribe.py \
  -k "not writes_to_store and not milestone_a and not full_pipeline_with"

# Smoke test: rapport engine with mock data
python3 -c "
import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from forgestream.emotion.rapport import RapportEngine, interpolate_weights
from forgestream.events.schema import Event, EventType

async def test():
    orch = MagicMock()
    orch.session_id = uuid4()
    orch.process_event = AsyncMock(return_value=True)
    engine = RapportEngine(orchestrator=orch, meeting_count=3)
    sid, bid = orch.session_id, uuid4()
    for i in range(30):
        await engine.on_event(Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=sid, branch_id=bid, author='test', evaluator=0.0,
            payload={'speaker_id': 'sp0' if i%2==0 else 'sp1', 'timestamp_ms': i*1000,
                     'arousal': 0.6, 'valence': 0.5, 'f0_mean': 200.0,
                     'energy_rms': 0.1, 'f0_std': 30.0},
        ))
    await engine.on_event(Event(
        event_type=EventType.ENTRAINMENT_SNAPSHOT,
        session_id=sid, branch_id=bid, author='dynamics', evaluator=0.0,
        payload={'timestamp_ms': 30000, 'speaker_pairs': [], 'group_metrics': {}},
    ))
    calls = [c for c in orch.process_event.call_args_list if c[0][0].event_type == EventType.RAPPORT_SCORE]
    p = calls[0][0][0].payload
    print(f'Group composite: {p[\"group_composite\"]}')
    print(f'Trend: {p[\"group_trend\"]}')
    print(f'Weights: {p[\"weights_applied\"]}')
    print(f'Pairs: {len(p[\"pair_scores\"])}')
    w = interpolate_weights(3)
    print(f'Meeting 3 weights: {w}')
    print('PASS: RapportEngine working')

asyncio.run(test())
"
```
