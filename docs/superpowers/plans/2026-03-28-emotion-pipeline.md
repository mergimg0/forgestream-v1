# ForgeStream Emotion Pipeline — Full Multi-Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-layer audio emotion detection pipeline to ForgeStream that extracts prosodic features, detects emotional states, measures group dynamics (entrainment, dominance, synchrony), feeds into the SOS evaluator, and enables GRPO self-improvement on emotional engagement.

**Architecture:** Audio chunks are tee'd from the existing `AudioSource` into both the Gemini claim path (unchanged) and a new parallel `EmotionExtractor`. The extractor runs openSMILE eGeMAPS + Parselmouth on sliding windows and emits `PROSODIC_FEATURE` events through the Orchestrator. Downstream consumers (EmotionCorrelator, GroupDynamicsEngine, Evaluator) subscribe to the EventBus and produce derived events. All emotion data is append-only ECEF events — replay, GRPO retroactive scoring, and dashboard time-travel all work automatically.

**Tech Stack:** openSMILE (eGeMAPS 88-feature set), Parselmouth (Praat-in-Python for F0/jitter/shimmer/HNR), SenseVoice-Small (optional, 70ms emotion classification), numpy, scipy (TLCC/DTW), asyncio (parallel extraction)

---

## Dependency Graph Between Phases

```
Phase 0 (AudioRingBuffer)
    └─► Phase 1 (EmotionExtractor + features)
            ├─► Phase 2 (EmotionCorrelator — claim alignment)
            │       └─► Phase 3 (Evaluator extension)
            │               └─► Phase 5 (GRPO emotion tuning)
            └─► Phase 4 (GroupDynamicsEngine — TLCC/CRQA/entropy)
                    └─► Phase 6 (Dashboard emotion visualization)
                            └─► Phase 7 (Cross-meeting emotion corpus)
```

Phases 2 and 4 are independent of each other and can be parallelized.

---

## Event Schema Design (Holistic — All Consumers Considered)

Three new `EventType` values. Payload shapes are designed once here and consumed by all phases.

### `PROSODIC_FEATURE` — emitted every ~1 second per speaker

```python
# Emitted by: EmotionExtractor (Phase 1)
# Consumed by: EmotionCorrelator (Phase 2), GroupDynamicsEngine (Phase 4),
#              Evaluator (Phase 3), Dashboard (Phase 6), GRPO (Phase 5)
payload = {
    "speaker_id": "unknown",          # str — "unknown" until diarization wired
    "timestamp_ms": 14200,            # int — position in audio timeline
    "chunk_index": 28,                # int — ring buffer chunk index for alignment
    "window_duration_ms": 3000,       # int — analysis window size

    # Categorical emotion (SenseVoice, optional — None if model not loaded)
    "emotion_tag": "excited",         # str | None
    "emotion_confidence": 0.82,       # float | None

    # Dimensional emotion (derived from eGeMAPS)
    "arousal": 0.73,                  # float [0,1] — from energy + F0 range
    "valence": 0.61,                  # float [0,1] — from spectral centroid + HNR
    "dominance": 0.55,                # float [0,1] — from loudness + duration

    # Voice quality (Parselmouth)
    "f0_mean": 185.2,                # float Hz
    "f0_std": 32.1,                  # float Hz
    "f0_contour": [180.1, 182.3, ...], # list[float] — raw F0 per 10ms frame (for TLCC)
    "energy_rms": 0.042,             # float — RMS energy
    "jitter_local": 0.012,           # float — cycle-to-cycle pitch variation
    "shimmer_local": 0.034,          # float — cycle-to-cycle amplitude variation
    "hnr": 18.5,                     # float dB — harmonics-to-noise ratio
    "spectral_centroid": 2340.0,     # float Hz — brightness

    # Full feature vector (openSMILE eGeMAPS)
    "egemaps_vector": [0.1, 0.2, ...],  # list[float] — 88 dims for ML downstream
}
```

### `EMOTION_STATE` — emitted on significant emotional shifts

```python
# Emitted by: EmotionCorrelator (Phase 2)
# Consumed by: SynthesisEngine (Phase 2), Dashboard (Phase 6), GRPO (Phase 5)
payload = {
    "speaker_id": "unknown",
    "timestamp_ms": 14200,
    "shift_type": "onset",            # str — "onset" | "peak" | "offset" | "sustained"
    "from_emotion": "neutral",        # str
    "to_emotion": "excited",          # str
    "arousal_delta": 0.35,            # float — change magnitude
    "valence_delta": 0.20,            # float
    "trigger_claim_id": "uuid-str",   # str | None — temporally aligned claim
    "confidence": 0.78,               # float
}
```

### `ENTRAINMENT_SNAPSHOT` — emitted every 60 seconds

```python
# Emitted by: GroupDynamicsEngine (Phase 4)
# Consumed by: Evaluator (Phase 3), Dashboard (Phase 6), GRPO (Phase 5)
payload = {
    "timestamp_ms": 60000,
    "window_duration_ms": 30000,
    "speaker_pairs": [
        {
            "speaker_a": "speaker_0",
            "speaker_b": "speaker_1",
            "f0_correlation": 0.65,       # TLCC peak correlation
            "f0_lag_ms": 120.0,           # TLCC lag (positive = a leads)
            "energy_correlation": 0.58,
            "synchrony_score": 0.42,      # CRQA recurrence rate
            "convergence_trend": 0.03,    # Pearson r of proximity over time
        }
    ],
    "group_metrics": {
        "participation_parity": 0.78,     # evenness of speaking time [0,1]
        "collective_engagement": 0.65,    # mean arousal × F0 variability
        "turn_taking_entropy": 2.3,       # bits — predictability of turn order
        "dominant_speaker": "speaker_0",  # str | None
    },
}
```

---

## File Structure (All Phases)

### New files

| File | Phase | Responsibility |
|------|-------|---------------|
| `forgestream/emotion/__init__.py` | 0 | Package exports |
| `forgestream/emotion/buffer.py` | 0 | `AudioRingBuffer` — lock-free ring buffer for audio chunks |
| `forgestream/emotion/features.py` | 1 | `ProsodicFeatures` dataclass, `EGeMAPSExtractor`, `PraatExtractor` |
| `forgestream/emotion/extractor.py` | 1 | `EmotionExtractor` — orchestrates feature extraction on sliding windows |
| `forgestream/emotion/correlator.py` | 2 | `EmotionCorrelator` — aligns claims with prosodic features, detects shifts |
| `forgestream/emotion/dynamics.py` | 4 | `GroupDynamicsEngine` — TLCC, CRQA, transfer entropy, dominance |
| `forgestream/emotion/speaker.py` | 4 | `SpeakerTimeSeries` — per-speaker prosodic time series accumulator |
| `tests/emotion/__init__.py` | 0 | Test package |
| `tests/emotion/test_buffer.py` | 0 | Ring buffer tests |
| `tests/emotion/test_features.py` | 1 | Feature extraction tests |
| `tests/emotion/test_extractor.py` | 1 | EmotionExtractor integration tests |
| `tests/emotion/test_correlator.py` | 2 | Correlator tests |
| `tests/emotion/test_dynamics.py` | 4 | Group dynamics algorithm tests |
| `tests/emotion/test_evaluator_emotion.py` | 3 | Extended evaluator tests |

### Modified files

| File | Phase | Changes |
|------|-------|---------|
| `forgestream/events/schema.py` | 0 | Add `PROSODIC_FEATURE`, `EMOTION_STATE`, `ENTRAINMENT_SNAPSHOT` to `EventType` |
| `forgestream/live_stream.py` | 1 | Wire `AudioRingBuffer` + `EmotionExtractor` into `_send_loop` and `start()` |
| `forgestream/config.py` | 1 | Add emotion pipeline configuration fields |
| `forgestream/governor/evaluator.py` | 3 | Add `emotional_engagement` weight + metric |
| `forgestream/governor/__init__.py` | 3 | Update exports |
| `forgestream/post_meeting.py` | 5 | Support 5-weight GRPO + emotion correlation analysis |
| `forgestream/governor/improvement.py` | 5 | Extend `WeightTuner` for 5 weights + tone adjustment tuning |
| `forgestream/graph/materializer.py` | 6 | Handle `PROSODIC_FEATURE` → speaker emotion nodes |
| `pyproject.toml` | 1 | Add `emotion` and `emotion-ml` optional dependency groups |

---

## Phase 0: AudioRingBuffer

### Task 0.1: Add new event types to schema

**Files:**
- Modify: `forgestream/events/schema.py:12-26`
- Test: `tests/events/test_schema.py` (existing, extend)

- [ ] **Step 1: Write test for new event types**

```python
# tests/emotion/__init__.py
# (empty file)
```

```python
# tests/emotion/test_schema_extension.py
"""Test that new emotion event types exist and serialize correctly."""

from uuid import uuid4

from forgestream.events.schema import Event, EventType


def test_prosodic_feature_event_type_exists():
    assert EventType.PROSODIC_FEATURE == "prosodic_feature"


def test_emotion_state_event_type_exists():
    assert EventType.EMOTION_STATE == "emotion_state"


def test_entrainment_snapshot_event_type_exists():
    assert EventType.ENTRAINMENT_SNAPSHOT == "entrainment_snapshot"


def test_prosodic_feature_event_serializes():
    event = Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=uuid4(),
        branch_id=uuid4(),
        author="emotion_extractor",
        evaluator=0.0,
        payload={
            "speaker_id": "unknown",
            "timestamp_ms": 1000,
            "chunk_index": 2,
            "arousal": 0.5,
            "valence": 0.5,
            "dominance": 0.5,
        },
    )
    d = event.to_dict()
    assert d["event_type"] == "prosodic_feature"
    assert d["payload"]["arousal"] == 0.5

    roundtrip = Event.from_dict(d)
    assert roundtrip.event_type == EventType.PROSODIC_FEATURE
    assert roundtrip.payload["speaker_id"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_schema_extension.py -v`
Expected: FAIL with `AttributeError: PROSODIC_FEATURE`

- [ ] **Step 3: Add event types to schema**

In `forgestream/events/schema.py`, add after line 26 (`MEETING_SUMMARY = "meeting_summary"`):

```python
    PROSODIC_FEATURE = "prosodic_feature"
    EMOTION_STATE = "emotion_state"
    ENTRAINMENT_SNAPSHOT = "entrainment_snapshot"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_schema_extension.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/events/schema.py tests/emotion/__init__.py tests/emotion/test_schema_extension.py
git commit -m "feat(events): add PROSODIC_FEATURE, EMOTION_STATE, ENTRAINMENT_SNAPSHOT event types"
```

---

### Task 0.2: AudioRingBuffer

**Files:**
- Create: `forgestream/emotion/__init__.py`
- Create: `forgestream/emotion/buffer.py`
- Test: `tests/emotion/test_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/emotion/test_buffer.py
"""Tests for AudioRingBuffer — the tee between AudioSource and EmotionExtractor."""

import pytest

from forgestream.emotion.buffer import AudioRingBuffer


class TestAudioRingBuffer:
    def test_write_and_read_single_chunk(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk = b"\x00\x01" * 8000  # 0.5s at 16kHz 16-bit mono = 16000 bytes
        idx = buf.write_chunk(chunk)
        assert idx == 0
        assert buf.read_chunk(0) == chunk

    def test_sequential_writes_increment_index(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk = b"\x00" * 16000
        idx0 = buf.write_chunk(chunk)
        idx1 = buf.write_chunk(chunk)
        idx2 = buf.write_chunk(chunk)
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_read_nonexistent_chunk_returns_none(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        assert buf.read_chunk(0) is None
        assert buf.read_chunk(99) is None

    def test_old_chunks_evicted_after_capacity(self):
        # 2 seconds capacity, 0.5s chunks = 4 chunks max
        buf = AudioRingBuffer(capacity_seconds=2.0, sample_rate=16000)
        chunk_size = 16000  # 0.5s
        chunks = [bytes([i % 256]) * chunk_size for i in range(6)]
        for c in chunks:
            buf.write_chunk(c)
        # Chunks 0, 1 should be evicted; chunks 2-5 should remain
        assert buf.read_chunk(0) is None
        assert buf.read_chunk(1) is None
        assert buf.read_chunk(2) is not None
        assert buf.read_chunk(5) is not None

    def test_read_window_returns_recent_audio(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        chunk_size = 16000
        for i in range(6):
            buf.write_chunk(bytes([i]) * chunk_size)
        # Read last 1.5 seconds = 3 chunks
        window = buf.read_window(duration_seconds=1.5)
        assert len(window) == chunk_size * 3
        # Should contain the 3 most recent chunks (indices 3, 4, 5)
        assert window[:chunk_size] == bytes([3]) * chunk_size
        assert window[-chunk_size:] == bytes([5]) * chunk_size

    def test_read_window_clamps_to_available(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        buf.write_chunk(b"\x42" * 16000)
        # Request more than available
        window = buf.read_window(duration_seconds=10.0)
        assert len(window) == 16000  # only 1 chunk available

    def test_chunk_timestamp_ms(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        buf.write_chunk(b"\x00" * 16000)  # chunk 0: 0-500ms
        buf.write_chunk(b"\x00" * 16000)  # chunk 1: 500-1000ms
        buf.write_chunk(b"\x00" * 16000)  # chunk 2: 1000-1500ms
        assert buf.chunk_timestamp_ms(0) == 0
        assert buf.chunk_timestamp_ms(1) == 500
        assert buf.chunk_timestamp_ms(2) == 1000

    def test_chunk_count(self):
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        assert buf.chunk_count == 0
        buf.write_chunk(b"\x00" * 16000)
        assert buf.chunk_count == 1
        buf.write_chunk(b"\x00" * 16000)
        assert buf.chunk_count == 2

    def test_read_window_as_numpy(self):
        numpy = pytest.importorskip("numpy")
        buf = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)
        # Write a known signal: 0.5s of silence
        buf.write_chunk(b"\x00\x00" * 8000)
        arr = buf.read_window_numpy(duration_seconds=0.5)
        assert arr.dtype == numpy.int16
        assert len(arr) == 8000  # 8000 samples at 16kHz for 0.5s
        assert arr.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forgestream.emotion'`

- [ ] **Step 3: Implement AudioRingBuffer**

```python
# forgestream/emotion/__init__.py
"""Audio emotion detection pipeline for ForgeStream."""
```

```python
# forgestream/emotion/buffer.py
"""AudioRingBuffer — stores audio chunks for parallel emotion extraction.

Lock-free ring buffer. The Gemini send loop writes chunks; the EmotionExtractor
reads them. Old chunks are evicted when capacity is exceeded. Chunk indices
are monotonically increasing and never reused.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMeta:
    """Metadata for a stored audio chunk."""

    index: int
    offset: int  # byte offset within the backing buffer
    length: int


class AudioRingBuffer:
    """Ring buffer for PCM 16kHz mono int16 audio chunks.

    Parameters:
        capacity_seconds: Maximum audio duration to retain.
        sample_rate: Audio sample rate in Hz.
        chunk_duration: Duration of each chunk in seconds.
    """

    def __init__(
        self,
        capacity_seconds: float = 30.0,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._bytes_per_chunk = int(sample_rate * 2 * chunk_duration)
        max_chunks = int(capacity_seconds / chunk_duration)
        self._max_chunks = max(1, max_chunks)

        self._chunks: deque[tuple[int, bytes]] = deque(
            maxlen=self._max_chunks
        )
        self._next_index: int = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently stored."""
        return len(self._chunks)

    def write_chunk(self, chunk: bytes) -> int:
        """Write a chunk and return its index.

        If the buffer is full, the oldest chunk is evicted.
        """
        idx = self._next_index
        self._next_index += 1
        self._chunks.append((idx, chunk))
        return idx

    def read_chunk(self, chunk_index: int) -> bytes | None:
        """Read a chunk by its index. Returns None if evicted or not found."""
        for idx, data in self._chunks:
            if idx == chunk_index:
                return data
        return None

    def read_window(self, duration_seconds: float) -> bytes:
        """Read the most recent N seconds of audio as contiguous bytes."""
        n_chunks = int(duration_seconds / self._chunk_duration)
        n_chunks = min(n_chunks, len(self._chunks))
        if n_chunks == 0:
            return b""
        recent = list(self._chunks)[-n_chunks:]
        return b"".join(data for _, data in recent)

    def read_window_numpy(self, duration_seconds: float) -> "numpy.ndarray":
        """Read the most recent N seconds as a numpy int16 array."""
        import numpy

        raw = self.read_window(duration_seconds)
        if not raw:
            return numpy.array([], dtype=numpy.int16)
        return numpy.frombuffer(raw, dtype=numpy.int16)

    def chunk_timestamp_ms(self, chunk_index: int) -> int:
        """Convert a chunk index to its start timestamp in milliseconds."""
        return int(chunk_index * self._chunk_duration * 1000)

    def latest_chunk_index(self) -> int | None:
        """Return the index of the most recently written chunk, or None."""
        if not self._chunks:
            return None
        return self._chunks[-1][0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_buffer.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/__init__.py forgestream/emotion/buffer.py tests/emotion/test_buffer.py
git commit -m "feat(emotion): add AudioRingBuffer for parallel audio tee"
```

---

## Phase 1: EmotionExtractor + Feature Extraction

### Task 1.1: Add emotion dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml:11-29`

- [ ] **Step 1: Add emotion dependency groups**

In `pyproject.toml`, add after line 16 (`]` closing `dev`):

```toml
emotion = [
    "opensmile>=2.5",
    "praat-parselmouth>=0.4",
    "numpy>=1.26",
]
emotion-ml = [
    "forgestream[emotion]",
    "funasr>=1.2",
    "torch>=2.0",
]
```

Update the `all` group (line 27-29) to include emotion:

```toml
all = [
    "forgestream[dev,gemini,tui,dashboard,emotion]",
]
```

- [ ] **Step 2: Install emotion dependencies**

Run: `cd /Users/mghome/projects/forgestream && pip install -e ".[emotion]"`
Expected: opensmile, praat-parselmouth, numpy install successfully

- [ ] **Step 3: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add pyproject.toml
git commit -m "build: add emotion pipeline dependencies (opensmile, parselmouth, numpy)"
```

---

### Task 1.2: Add emotion config fields

**Files:**
- Modify: `forgestream/config.py:9-45`
- Test: `tests/test_config.py` (extend)

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_config.py
"""Test emotion pipeline configuration fields."""

from forgestream.config import ForgeStreamConfig


def test_emotion_config_defaults():
    config = ForgeStreamConfig()
    assert config.emotion_enabled is True
    assert config.emotion_window_seconds == 3.0
    assert config.emotion_stride_seconds == 1.0
    assert config.emotion_ml_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_config.py -v`
Expected: FAIL with `AttributeError: emotion_enabled`

- [ ] **Step 3: Add config fields**

In `forgestream/config.py`, add after line 44 (`data_dir: str = "data"`):

```python
    # Emotion pipeline
    emotion_enabled: bool = True
    emotion_window_seconds: float = 3.0   # analysis window duration
    emotion_stride_seconds: float = 1.0   # emit interval
    emotion_ml_enabled: bool = False       # SenseVoice/emotion2vec (requires torch)
    emotion_buffer_seconds: float = 30.0   # ring buffer capacity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/config.py tests/emotion/test_config.py
git commit -m "feat(config): add emotion pipeline configuration fields"
```

---

### Task 1.3: ProsodicFeatures dataclass + extractors

**Files:**
- Create: `forgestream/emotion/features.py`
- Test: `tests/emotion/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/emotion/test_features.py
"""Tests for prosodic feature extraction using openSMILE and Parselmouth."""

import numpy as np
import pytest

from forgestream.emotion.features import (
    EGeMAPSExtractor,
    PraatExtractor,
    ProsodicFeatures,
)


def _make_sine_wave(
    freq: float = 200.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 10000.0,
) -> np.ndarray:
    """Generate a sine wave as int16 samples (simulates voiced speech)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal


class TestProsodicFeatures:
    def test_dataclass_fields(self):
        f = ProsodicFeatures(
            f0_mean=185.0,
            f0_std=30.0,
            f0_contour=[180.0, 185.0, 190.0],
            energy_rms=0.04,
            jitter_local=0.01,
            shimmer_local=0.03,
            hnr=18.0,
            spectral_centroid=2300.0,
            egemaps_vector=[0.1] * 88,
            arousal=0.5,
            valence=0.5,
            dominance=0.5,
        )
        assert f.f0_mean == 185.0
        assert len(f.egemaps_vector) == 88

    def test_to_payload(self):
        f = ProsodicFeatures(
            f0_mean=185.0,
            f0_std=30.0,
            f0_contour=[180.0],
            energy_rms=0.04,
            jitter_local=0.01,
            shimmer_local=0.03,
            hnr=18.0,
            spectral_centroid=2300.0,
            egemaps_vector=[0.1] * 88,
            arousal=0.6,
            valence=0.7,
            dominance=0.4,
        )
        p = f.to_payload()
        assert p["f0_mean"] == 185.0
        assert p["arousal"] == 0.6
        assert len(p["egemaps_vector"]) == 88


class TestPraatExtractor:
    def test_extract_from_sine_wave(self):
        signal = _make_sine_wave(freq=200.0, duration=1.0)
        extractor = PraatExtractor(sample_rate=16000)
        result = extractor.extract(signal)
        # A 200Hz sine wave should give F0 near 200
        assert 150.0 < result["f0_mean"] < 250.0
        assert result["f0_std"] >= 0.0
        assert len(result["f0_contour"]) > 0
        assert result["energy_rms"] > 0.0
        assert isinstance(result["jitter_local"], float)
        assert isinstance(result["shimmer_local"], float)
        assert isinstance(result["hnr"], float)

    def test_extract_from_silence(self):
        signal = np.zeros(16000, dtype=np.int16)
        extractor = PraatExtractor(sample_rate=16000)
        result = extractor.extract(signal)
        # Silence: F0 should be 0 or NaN-clamped to 0
        assert result["f0_mean"] == 0.0


class TestEGeMAPSExtractor:
    def test_extract_returns_88_features(self):
        opensmile = pytest.importorskip("opensmile")
        signal = _make_sine_wave(freq=200.0, duration=1.0)
        extractor = EGeMAPSExtractor(sample_rate=16000)
        vector = extractor.extract(signal)
        assert len(vector) == 88

    def test_extract_from_silence(self):
        opensmile = pytest.importorskip("opensmile")
        signal = np.zeros(16000, dtype=np.int16)
        extractor = EGeMAPSExtractor(sample_rate=16000)
        vector = extractor.extract(signal)
        assert len(vector) == 88
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forgestream.emotion.features'`

- [ ] **Step 3: Implement features module**

```python
# forgestream/emotion/features.py
"""Prosodic feature extraction using openSMILE eGeMAPS and Parselmouth.

Two extractors:
- PraatExtractor: F0, jitter, shimmer, HNR, energy, spectral centroid
- EGeMAPSExtractor: Full 88-dimensional eGeMAPS feature vector

Both accept numpy int16 arrays at 16kHz and return plain Python types
(no pandas/numpy in the output — safe for JSON serialization in event payloads).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProsodicFeatures:
    """Complete prosodic feature set for one analysis window."""

    # Pitch
    f0_mean: float
    f0_std: float
    f0_contour: list[float]

    # Energy
    energy_rms: float

    # Voice quality
    jitter_local: float
    shimmer_local: float
    hnr: float
    spectral_centroid: float

    # Full eGeMAPS vector
    egemaps_vector: list[float] = field(default_factory=list)

    # Dimensional emotion (derived)
    arousal: float = 0.5
    valence: float = 0.5
    dominance: float = 0.5

    def to_payload(self) -> dict:
        """Convert to ECEF event payload dict."""
        return {
            "f0_mean": self.f0_mean,
            "f0_std": self.f0_std,
            "f0_contour": self.f0_contour,
            "energy_rms": self.energy_rms,
            "jitter_local": self.jitter_local,
            "shimmer_local": self.shimmer_local,
            "hnr": self.hnr,
            "spectral_centroid": self.spectral_centroid,
            "egemaps_vector": self.egemaps_vector,
            "arousal": self.arousal,
            "valence": self.valence,
            "dominance": self.dominance,
        }


class PraatExtractor:
    """Extract F0, jitter, shimmer, HNR, energy, spectral centroid via Parselmouth."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate

    def extract(self, signal: "numpy.ndarray") -> dict:
        """Extract voice quality features from an int16 numpy array.

        Returns a dict with: f0_mean, f0_std, f0_contour, energy_rms,
        jitter_local, shimmer_local, hnr, spectral_centroid.
        """
        import numpy as np
        import parselmouth
        from parselmouth import praat

        # Convert int16 to float64 in [-1, 1] range for Parselmouth
        signal_float = signal.astype(np.float64) / 32768.0
        snd = parselmouth.Sound(signal_float, sampling_frequency=self._sample_rate)

        # Pitch extraction
        pitch = snd.to_pitch(time_step=0.01)  # 10ms frames
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        if len(voiced) == 0:
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_contour": [],
                "energy_rms": float(np.sqrt(np.mean(signal_float**2))),
                "jitter_local": 0.0,
                "shimmer_local": 0.0,
                "hnr": 0.0,
                "spectral_centroid": 0.0,
            }

        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_contour = [float(v) for v in f0_values]

        # Energy (RMS)
        energy_rms = float(np.sqrt(np.mean(signal_float**2)))

        # Jitter and Shimmer via PointProcess
        point_process = praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 600
        )
        jitter_local = praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = praat.call(
            [snd, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        # Clamp NaN to 0
        if math.isnan(jitter_local):
            jitter_local = 0.0
        if math.isnan(shimmer_local):
            shimmer_local = 0.0

        # Harmonics-to-Noise Ratio
        harmonicity = snd.to_harmonicity()
        hnr = praat.call(harmonicity, "Get mean", 0, 0)
        if math.isnan(hnr):
            hnr = 0.0

        # Spectral centroid
        spectrum = snd.to_spectrum()
        spectral_centroid = praat.call(
            spectrum, "Get centre of gravity...", 1
        )
        if math.isnan(spectral_centroid):
            spectral_centroid = 0.0

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_contour": f0_contour,
            "energy_rms": energy_rms,
            "jitter_local": float(jitter_local),
            "shimmer_local": float(shimmer_local),
            "hnr": float(hnr),
            "spectral_centroid": float(spectral_centroid),
        }


class EGeMAPSExtractor:
    """Extract the 88-dimensional eGeMAPS feature vector via openSMILE."""

    def __init__(self, sample_rate: int = 16000) -> None:
        import opensmile

        self._sample_rate = sample_rate
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, signal: "numpy.ndarray") -> list[float]:
        """Extract eGeMAPS functionals from an int16 numpy array.

        Returns a list of 88 floats.
        """
        import numpy as np

        # openSMILE expects float32 in [-1, 1]
        signal_float = signal.astype(np.float32) / 32768.0
        result = self._smile.process_signal(signal_float, self._sample_rate)
        return [float(v) for v in result.values[0]]


def derive_dimensional_emotion(
    f0_mean: float,
    f0_std: float,
    energy_rms: float,
    hnr: float,
    spectral_centroid: float,
) -> tuple[float, float, float]:
    """Derive arousal, valence, dominance from prosodic features.

    Uses established psychoacoustic mappings:
    - Arousal: energy + F0 variability (high energy + high F0 range = high arousal)
    - Valence: spectral centroid + HNR (bright + clear = positive)
    - Dominance: energy + F0 mean (loud + low pitch = dominant)

    Returns (arousal, valence, dominance) each in [0, 1].
    """
    # Normalize features to approximate [0, 1] using typical speech ranges
    # F0 range: 75-600 Hz, typical std: 0-80 Hz
    # Energy RMS: 0-0.3 for speech
    # HNR: 0-40 dB
    # Spectral centroid: 500-5000 Hz

    norm_f0_std = min(1.0, f0_std / 80.0)
    norm_energy = min(1.0, energy_rms / 0.15)
    norm_hnr = min(1.0, max(0.0, hnr / 30.0))
    norm_centroid = min(1.0, max(0.0, (spectral_centroid - 500) / 4500))
    norm_f0_mean = min(1.0, max(0.0, (f0_mean - 75) / 525))

    arousal = 0.6 * norm_energy + 0.4 * norm_f0_std
    valence = 0.5 * norm_centroid + 0.5 * norm_hnr
    dominance = 0.6 * norm_energy + 0.4 * (1.0 - norm_f0_mean)  # lower pitch = more dominant

    return (
        max(0.0, min(1.0, arousal)),
        max(0.0, min(1.0, valence)),
        max(0.0, min(1.0, dominance)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_features.py -v`
Expected: All tests PASS (eGeMAPS tests skip if opensmile not installed)

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/features.py tests/emotion/test_features.py
git commit -m "feat(emotion): add PraatExtractor, EGeMAPSExtractor, ProsodicFeatures"
```

---

### Task 1.4: EmotionExtractor — sliding window orchestrator

**Files:**
- Create: `forgestream/emotion/extractor.py`
- Test: `tests/emotion/test_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/emotion/test_extractor.py
"""Tests for EmotionExtractor — the sliding-window feature extraction orchestrator."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from forgestream.emotion.buffer import AudioRingBuffer
from forgestream.emotion.extractor import EmotionExtractor
from forgestream.events.schema import Event, EventType


def _make_sine_chunk(freq: float = 200.0) -> bytes:
    """Generate a 0.5s PCM chunk of a sine wave."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    signal = (10000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal.tobytes()


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.session_id = uuid4()
    orch.process_event = AsyncMock(return_value=True)
    return orch


@pytest.fixture
def audio_buffer():
    return AudioRingBuffer(capacity_seconds=10.0)


class TestEmotionExtractor:
    @pytest.mark.asyncio
    async def test_emits_prosodic_feature_event_after_stride(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=1.5,   # 3 chunks
            stride_seconds=1.0,   # every 2 chunks
        )
        chunk = _make_sine_chunk(200.0)

        # Feed 2 chunks (= 1 stride)
        await extractor.process_chunk(chunk, chunk_index=0)
        assert mock_orchestrator.process_event.call_count == 0

        await extractor.process_chunk(chunk, chunk_index=1)
        assert mock_orchestrator.process_event.call_count == 1

        event = mock_orchestrator.process_event.call_args[0][0]
        assert event.event_type == EventType.PROSODIC_FEATURE
        assert event.author == "emotion_extractor"
        assert "f0_mean" in event.payload
        assert "arousal" in event.payload
        assert "egemaps_vector" in event.payload or event.payload.get("egemaps_vector") is not None

    @pytest.mark.asyncio
    async def test_does_not_emit_before_stride(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=1.5,
            stride_seconds=1.0,
        )
        chunk = _make_sine_chunk(200.0)
        await extractor.process_chunk(chunk, chunk_index=0)
        assert mock_orchestrator.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_payload_contains_all_required_fields(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=0.5,  # 1 chunk window
            stride_seconds=0.5,  # emit every chunk
        )
        chunk = _make_sine_chunk(200.0)
        await extractor.process_chunk(chunk, chunk_index=0)

        event = mock_orchestrator.process_event.call_args[0][0]
        payload = event.payload
        required_fields = [
            "speaker_id", "timestamp_ms", "chunk_index", "window_duration_ms",
            "f0_mean", "f0_std", "f0_contour", "energy_rms",
            "jitter_local", "shimmer_local", "hnr", "spectral_centroid",
            "arousal", "valence", "dominance",
        ]
        for f in required_fields:
            assert f in payload, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_graceful_with_silence(
        self, mock_orchestrator, audio_buffer
    ):
        extractor = EmotionExtractor(
            orchestrator=mock_orchestrator,
            audio_buffer=audio_buffer,
            branch_id=uuid4(),
            window_seconds=0.5,
            stride_seconds=0.5,
        )
        silence = b"\x00\x00" * 8000  # 0.5s silence
        await extractor.process_chunk(silence, chunk_index=0)

        event = mock_orchestrator.process_event.call_args[0][0]
        assert event.payload["f0_mean"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forgestream.emotion.extractor'`

- [ ] **Step 3: Implement EmotionExtractor**

```python
# forgestream/emotion/extractor.py
"""EmotionExtractor — sliding-window prosodic feature extraction.

Accumulates audio chunks in a sliding window. At each stride boundary,
concatenates the window, runs Parselmouth + openSMILE, derives dimensional
emotion, and emits a PROSODIC_FEATURE event through the Orchestrator.

Runs feature extraction in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from forgestream.events.schema import Event, EventType

from .buffer import AudioRingBuffer
from .features import (
    EGeMAPSExtractor,
    PraatExtractor,
    ProsodicFeatures,
    derive_dimensional_emotion,
)

logger = logging.getLogger(__name__)

AUTHOR = "emotion_extractor"


class EmotionExtractor:
    """Sliding-window prosodic feature extractor.

    Parameters:
        orchestrator: The ForgeStream Orchestrator to emit events through.
        audio_buffer: The shared AudioRingBuffer.
        branch_id: The current branch ID for emitted events.
        window_seconds: Duration of the analysis window (default 3.0s).
        stride_seconds: How often to emit features (default 1.0s).
        chunk_duration: Duration of each incoming chunk (default 0.5s).
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        audio_buffer: AudioRingBuffer,
        branch_id: UUID,
        window_seconds: float = 3.0,
        stride_seconds: float = 1.0,
        chunk_duration: float = 0.5,
    ) -> None:
        self._orchestrator = orchestrator
        self._audio_buffer = audio_buffer
        self._branch_id = branch_id

        self._window_seconds = window_seconds
        self._stride_seconds = stride_seconds
        self._chunk_duration = chunk_duration

        # Chunk accumulation
        max_window_chunks = int(window_seconds / chunk_duration)
        self._chunk_window: deque[bytes] = deque(maxlen=max_window_chunks)
        self._chunks_per_stride = max(1, int(stride_seconds / chunk_duration))
        self._stride_counter = 0

        # Extractors (initialized lazily)
        self._praat = PraatExtractor(sample_rate=16000)
        self._egemaps: EGeMAPSExtractor | None = None
        self._egemaps_available: bool | None = None

        # Thread pool for CPU-bound feature extraction
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_egemaps(self) -> EGeMAPSExtractor | None:
        """Lazily initialize eGeMAPS extractor. Returns None if unavailable."""
        if self._egemaps_available is None:
            try:
                self._egemaps = EGeMAPSExtractor(sample_rate=16000)
                self._egemaps_available = True
            except ImportError:
                logger.info("openSMILE not available — eGeMAPS features disabled")
                self._egemaps_available = False
        return self._egemaps if self._egemaps_available else None

    async def process_chunk(self, chunk: bytes, chunk_index: int) -> None:
        """Process an incoming audio chunk.

        Accumulates chunks and runs extraction at stride boundaries.
        """
        self._chunk_window.append(chunk)
        self._stride_counter += 1

        if self._stride_counter >= self._chunks_per_stride:
            self._stride_counter = 0
            window_bytes = b"".join(self._chunk_window)
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                self._executor,
                self._extract_features,
                window_bytes,
            )
            await self._emit_event(features, chunk_index)

    def _extract_features(self, window_bytes: bytes) -> ProsodicFeatures:
        """Run feature extraction on a concatenated audio window (CPU-bound)."""
        import numpy as np

        signal = np.frombuffer(window_bytes, dtype=np.int16)

        # Parselmouth: F0, jitter, shimmer, HNR, spectral centroid
        praat_result = self._praat.extract(signal)

        # eGeMAPS: 88-dim vector (if available)
        egemaps = self._get_egemaps()
        egemaps_vector = egemaps.extract(signal) if egemaps else [0.0] * 88

        # Derive dimensional emotion
        arousal, valence, dominance = derive_dimensional_emotion(
            f0_mean=praat_result["f0_mean"],
            f0_std=praat_result["f0_std"],
            energy_rms=praat_result["energy_rms"],
            hnr=praat_result["hnr"],
            spectral_centroid=praat_result["spectral_centroid"],
        )

        return ProsodicFeatures(
            f0_mean=praat_result["f0_mean"],
            f0_std=praat_result["f0_std"],
            f0_contour=praat_result["f0_contour"],
            energy_rms=praat_result["energy_rms"],
            jitter_local=praat_result["jitter_local"],
            shimmer_local=praat_result["shimmer_local"],
            hnr=praat_result["hnr"],
            spectral_centroid=praat_result["spectral_centroid"],
            egemaps_vector=egemaps_vector,
            arousal=arousal,
            valence=valence,
            dominance=dominance,
        )

    async def _emit_event(
        self, features: ProsodicFeatures, chunk_index: int
    ) -> None:
        """Emit a PROSODIC_FEATURE event through the Orchestrator."""
        payload = features.to_payload()
        payload.update({
            "speaker_id": "unknown",  # Phase 4 adds real speaker IDs
            "timestamp_ms": self._audio_buffer.chunk_timestamp_ms(chunk_index),
            "chunk_index": chunk_index,
            "window_duration_ms": int(self._window_seconds * 1000),
            "emotion_tag": None,       # Phase 1.5 adds SenseVoice
            "emotion_confidence": None,
        })

        event = Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=self._orchestrator.session_id,
            branch_id=self._branch_id,
            author=AUTHOR,
            evaluator=0.0,
            payload=payload,
        )
        await self._orchestrator.process_event(event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_extractor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/extractor.py tests/emotion/test_extractor.py
git commit -m "feat(emotion): add EmotionExtractor with sliding-window Parselmouth + eGeMAPS"
```

---

### Task 1.5: Wire EmotionExtractor into GeminiLiveStream

**Files:**
- Modify: `forgestream/live_stream.py:60-67,127-134,146-158`
- Test: `tests/test_live_stream.py` (extend)

- [ ] **Step 1: Write failing test**

```python
# tests/emotion/test_live_stream_wiring.py
"""Test that GeminiLiveStream wires the emotion pipeline correctly."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import EventType
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


def test_live_stream_creates_audio_buffer_when_emotion_enabled():
    config = ForgeStreamConfig(emotion_enabled=True)
    store = MagicMock()
    orchestrator = Orchestrator(config=config, store=store)
    source = MagicMock()

    stream = GeminiLiveStream(config, orchestrator, source)
    assert stream.audio_buffer is not None
    assert stream.emotion_extractor is not None


def test_live_stream_skips_emotion_when_disabled():
    config = ForgeStreamConfig(emotion_enabled=False)
    store = MagicMock()
    orchestrator = Orchestrator(config=config, store=store)
    source = MagicMock()

    stream = GeminiLiveStream(config, orchestrator, source)
    assert stream.audio_buffer is None
    assert stream.emotion_extractor is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_live_stream_wiring.py -v`
Expected: FAIL with `AttributeError: 'GeminiLiveStream' object has no attribute 'audio_buffer'`

- [ ] **Step 3: Wire emotion pipeline into GeminiLiveStream**

In `forgestream/live_stream.py`, add imports at the top (after line 17):

```python
from .emotion.buffer import AudioRingBuffer
from .emotion.extractor import EmotionExtractor
```

In `__init__` (after line 83, `self.materializer = GraphMaterializer()`), add:

```python
        # Emotion pipeline (parallel to claim extraction)
        if config.emotion_enabled:
            self.audio_buffer = AudioRingBuffer(
                capacity_seconds=config.emotion_buffer_seconds,
            )
            self.emotion_extractor = EmotionExtractor(
                orchestrator=orchestrator,
                audio_buffer=self.audio_buffer,
                branch_id=self.branch_id,
                window_seconds=config.emotion_window_seconds,
                stride_seconds=config.emotion_stride_seconds,
            )
            self._emotion_queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        else:
            self.audio_buffer = None
            self.emotion_extractor = None
            self._emotion_queue = None
```

In `_send_loop` (replace lines 146-158):

```python
    async def _send_loop(self) -> None:
        """Send audio chunks to Gemini and tee into emotion pipeline."""
        try:
            async for chunk in self.audio_source.chunks():
                if not self._active:
                    break
                # Tee into emotion pipeline
                if self.audio_buffer is not None:
                    chunk_idx = self.audio_buffer.write_chunk(chunk)
                    self._emotion_queue.put_nowait((chunk, chunk_idx))
                # Send to Gemini (existing behavior)
                if self._session:
                    await self._session.send(
                        {"data": chunk, "mime_type": "audio/pcm"}
                    )
        except asyncio.CancelledError:
            pass
```

In `start` (add the emotion extraction loop task, after line 133):

```python
    async def start(self) -> None:
        """Start streaming audio and receiving claims."""
        await self.audio_source.start()
        self._tasks = [
            asyncio.create_task(self._send_loop()),
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._context_injection_loop()),
        ]
        if self.emotion_extractor is not None:
            self._tasks.append(
                asyncio.create_task(self._emotion_extraction_loop())
            )
```

Add the new method (after `_context_injection_loop`):

```python
    async def _emotion_extraction_loop(self) -> None:
        """Process audio chunks through the EmotionExtractor."""
        try:
            while self._active:
                chunk, chunk_idx = await self._emotion_queue.get()
                await self.emotion_extractor.process_chunk(chunk, chunk_idx)
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/emotion/test_live_stream_wiring.py -v`
Expected: PASS

Run full test suite to check for regressions:
Run: `cd /Users/mghome/projects/forgestream && python3 -m pytest tests/ -q --ignore=tests/events/test_store.py --ignore=tests/events/test_subscribe.py -k "not writes_to_store and not milestone_a and not full_pipeline_with"`
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/live_stream.py tests/emotion/test_live_stream_wiring.py
git commit -m "feat(emotion): wire EmotionExtractor into GeminiLiveStream send loop"
```

---

### Task 1.6: Update emotion package exports

**Files:**
- Modify: `forgestream/emotion/__init__.py`

- [ ] **Step 1: Update exports**

```python
# forgestream/emotion/__init__.py
"""Audio emotion detection pipeline for ForgeStream."""
from .buffer import AudioRingBuffer
from .extractor import EmotionExtractor
from .features import EGeMAPSExtractor, PraatExtractor, ProsodicFeatures

__all__ = [
    "AudioRingBuffer",
    "EGeMAPSExtractor",
    "EmotionExtractor",
    "PraatExtractor",
    "ProsodicFeatures",
]
```

- [ ] **Step 2: Commit**

```bash
cd /Users/mghome/projects/forgestream
git add forgestream/emotion/__init__.py
git commit -m "feat(emotion): export public API from emotion package"
```

---

## Phase 2: EmotionCorrelator — Claim-Emotion Alignment (Architecture)

> Detailed implementation steps deferred until Phase 0+1 are validated. Interface contracts defined here.

### Task 2.1: EmotionCorrelator

**Files:**
- Create: `forgestream/emotion/correlator.py`
- Test: `tests/emotion/test_correlator.py`

**Interface contract:**

```python
# forgestream/emotion/correlator.py
"""Correlates CLAIM events with temporally-aligned PROSODIC_FEATURE events.

Subscribes to EventBus. Maintains a recent window of PROSODIC_FEATURE events.
When a CLAIM arrives, finds the closest prosodic feature by timestamp and:
1. Enriches the claim with acoustic confidence adjustments
2. Detects significant emotional shifts → emits EMOTION_STATE events
3. Tracks per-speaker emotion trajectories
"""

AUTHOR = "emotion_correlator"

# Temporal alignment tolerance: claims within ±2s of a prosodic feature match
ALIGNMENT_TOLERANCE_MS = 2000


class EmotionCorrelator:
    def __init__(self, orchestrator: "Orchestrator") -> None: ...

    async def on_event(self, event: Event) -> None:
        """EventBus handler. Processes CLAIM and PROSODIC_FEATURE events."""
        # If PROSODIC_FEATURE: buffer it (deque, max 30 entries)
        # If CLAIM: find nearest prosodic feature by timestamp
        #   - Compute acoustic confidence adjustment:
        #       high arousal + emphasis → boost confidence
        #       high jitter + low HNR → stress indicator → lower confidence
        #   - Check for emotional shift (arousal_delta > 0.2 from previous window)
        #       → emit EMOTION_STATE event
        # Ignore self-authored events (AUTHOR check)

    def _find_nearest_feature(self, timestamp_ms: int) -> dict | None:
        """Find the PROSODIC_FEATURE closest to the given timestamp."""

    def _compute_confidence_adjustment(self, prosodic: dict) -> float:
        """Data-driven confidence adjustment replacing hardcoded tone markers.

        Returns a float in [-0.3, +0.3] to add to claim confidence.
        Uses: arousal, jitter, shimmer, HNR, f0_std
        """

    def _detect_emotional_shift(self, current: dict, previous: dict) -> dict | None:
        """Detect significant emotional shift between consecutive windows.

        Returns EMOTION_STATE payload dict if shift detected, None otherwise.
        Threshold: arousal_delta > 0.2 OR valence_delta > 0.2
        """
```

**Wiring:** `orchestrator.event_bus.subscribe(correlator.on_event)` — called in `GeminiLiveStream.__init__` or a new `attach_emotion_correlator()` method on Orchestrator.

**Test signatures:**

```python
# tests/emotion/test_correlator.py
class TestEmotionCorrelator:
    async def test_enriches_claim_with_prosodic_context(self): ...
    async def test_emits_emotion_state_on_significant_shift(self): ...
    async def test_ignores_self_authored_events(self): ...
    async def test_handles_claim_with_no_matching_prosodic(self): ...
    async def test_confidence_adjustment_range(self): ...
```

---

## Phase 3: Evaluator Extension (Architecture)

> Detailed implementation steps deferred. Interface contracts defined here.

### Task 3.1: Add emotional_engagement to Evaluator

**Files:**
- Modify: `forgestream/governor/evaluator.py:15-67`
- Test: `tests/emotion/test_evaluator_emotion.py`

**Interface contract:**

```python
# Changes to forgestream/governor/evaluator.py

@dataclass
class EvaluatorMetrics:
    knowledge_density: float
    verification_rate: float
    scaffold_success: float
    suggestion_uptake: float
    emotional_engagement: float  # NEW
    composite: float


class Evaluator:
    DEFAULT_WEIGHTS = {
        "knowledge": 0.25,       # was 0.30
        "verification": 0.25,    # was 0.30
        "scaffold": 0.20,        # was 0.25
        "uptake": 0.15,          # unchanged
        "engagement": 0.15,      # NEW
    }

    def _emotional_engagement(self, events: list[Event]) -> float:
        """Compute emotional engagement from PROSODIC_FEATURE events.

        Combines:
        - Mean arousal across speakers (weighted 0.3)
        - F0 variability across speakers (weighted 0.3)
        - Participation parity from ENTRAINMENT_SNAPSHOT (weighted 0.2)
        - Emotion transition frequency (weighted 0.2)

        Returns 0.5 if no prosodic events exist (graceful default).
        """
```

**Critical detail:** `PostMeetingSynthesis.load_weights()` at `post_meeting.py:108` filters for 4 keys. It MUST be updated in this phase to include `"engagement"`:

```python
def load_weights(self) -> dict[str, float]:
    weights_file = self.data_dir / "weights.json"
    if weights_file.exists():
        data = json.loads(weights_file.read_text())
        valid_keys = {"knowledge", "verification", "scaffold", "uptake", "engagement"}
        return {k: v for k, v in data.items() if k in valid_keys}
    return Evaluator.DEFAULT_WEIGHTS.copy()
```

`GRPO WeightTuner` already handles arbitrary key dicts — no change needed there. It must also handle backward compatibility: if `weights.json` has only 4 keys, merge with defaults to add the 5th.

**Test signatures:**

```python
# tests/emotion/test_evaluator_emotion.py
class TestEmotionalEngagement:
    def test_default_weights_sum_to_one(self): ...
    def test_engagement_from_prosodic_events(self): ...
    def test_engagement_defaults_to_half_without_prosodic(self): ...
    def test_composite_includes_engagement(self): ...
    def test_backward_compatible_with_4_weight_files(self): ...
```

---

## Phase 4: GroupDynamicsEngine — Meta-Vectors (Architecture)

> Detailed implementation steps deferred. Interface contracts and key algorithms defined here.

### Task 4.1: SpeakerTimeSeries

**Files:**
- Create: `forgestream/emotion/speaker.py`
- Test: `tests/emotion/test_speaker.py`

```python
# forgestream/emotion/speaker.py
"""Per-speaker prosodic time series accumulator.

Subscribes to PROSODIC_FEATURE events and maintains rolling time series
per speaker_id. Used by GroupDynamicsEngine for TLCC/CRQA computation.
"""

class SpeakerTimeSeries:
    def __init__(self, max_duration_seconds: float = 120.0) -> None:
        self._series: dict[str, deque[tuple[int, dict]]] = {}  # speaker_id → [(timestamp_ms, features)]

    def add_feature(self, speaker_id: str, timestamp_ms: int, features: dict) -> None:
        """Add a prosodic feature snapshot for a speaker."""

    def get_f0_contours(self, speaker_a: str, speaker_b: str) -> tuple[list[float], list[float]]:
        """Get aligned F0 mean time series for two speakers (for TLCC)."""

    def get_energy_series(self, speaker_id: str) -> list[float]:
        """Get energy RMS time series for a speaker."""

    def speaker_ids(self) -> list[str]:
        """List all known speaker IDs."""

    def speaking_durations(self) -> dict[str, float]:
        """Compute total speaking duration per speaker in seconds."""
```

### Task 4.2: GroupDynamicsEngine

**Files:**
- Create: `forgestream/emotion/dynamics.py`
- Test: `tests/emotion/test_dynamics.py`

```python
# forgestream/emotion/dynamics.py
"""Group dynamics computation: TLCC, CRQA, transfer entropy, dominance.

Subscribes to EventBus. Every 60 seconds, computes group dynamics from
accumulated per-speaker prosodic time series and emits an ENTRAINMENT_SNAPSHOT.
"""

import numpy as np
from scipy import signal as scipy_signal

AUTHOR = "dynamics_engine"
SNAPSHOT_INTERVAL_SECONDS = 60


class GroupDynamicsEngine:
    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator
        self._speaker_series = SpeakerTimeSeries()
        self._last_snapshot_ms = 0

    async def on_event(self, event: Event) -> None:
        """EventBus handler. Accumulates PROSODIC_FEATURE, periodically emits snapshots."""

    def compute_tlcc(
        self, series_a: list[float], series_b: list[float], max_lag: int = 30
    ) -> tuple[float, int]:
        """Time-Lagged Cross-Correlation.

        Returns (peak_correlation, lag_index).
        Positive lag = series_a leads series_b.

        Algorithm:
            correlation = scipy.signal.correlate(a, b, mode='full')
            normalize by sqrt(sum(a²) * sum(b²))
            peak = argmax within ±max_lag window
        """

    def compute_crqa_recurrence_rate(
        self, series_a: list[float], series_b: list[float], radius: float = 0.1
    ) -> float:
        """Cross-Recurrence Quantification: recurrence rate.

        Returns fraction of points in cross-recurrence plot that are recurrent.

        Algorithm:
            distance_matrix[i,j] = |a[i] - b[j]|
            recurrence_matrix = distance_matrix < radius * std(concatenate(a, b))
            recurrence_rate = sum(recurrence_matrix) / total_elements
        """

    def compute_participation_parity(self, durations: dict[str, float]) -> float:
        """How evenly distributed is speaking time?

        Returns 1.0 for perfectly even, 0.0 for single-speaker dominance.

        Algorithm: 1.0 - Gini coefficient of speaking durations.
        """

    def compute_turn_taking_entropy(self, speaker_sequence: list[str]) -> float:
        """Shannon entropy of the turn-taking transition matrix.

        High entropy = unpredictable/collaborative turn-taking.
        Low entropy = rigid/formal structure.
        """

    def compute_collective_engagement(self, prosodic_events: list[dict]) -> float:
        """Mean arousal × mean F0 variability across all speakers."""

    def _emit_snapshot(self, timestamp_ms: int) -> None:
        """Build and emit ENTRAINMENT_SNAPSHOT event."""
```

**Key algorithm implementations (reference):**

```python
def compute_tlcc(self, series_a, series_b, max_lag=30):
    a = np.array(series_a) - np.mean(series_a)
    b = np.array(series_b) - np.mean(series_b)
    correlation = scipy_signal.correlate(a, b, mode='full')
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm > 0:
        correlation = correlation / norm
    mid = len(correlation) // 2
    window = correlation[mid - max_lag : mid + max_lag + 1]
    peak_idx = np.argmax(np.abs(window))
    lag = peak_idx - max_lag
    return float(window[peak_idx]), int(lag)

def compute_crqa_recurrence_rate(self, series_a, series_b, radius=0.1):
    a = np.array(series_a)
    b = np.array(series_b)
    threshold = radius * np.std(np.concatenate([a, b]))
    dist = np.abs(a[:, None] - b[None, :])
    recurrent = dist < threshold
    return float(np.mean(recurrent))
```

**Test signatures:**

```python
# tests/emotion/test_dynamics.py
class TestTLCC:
    def test_identical_signals_have_correlation_one_lag_zero(self): ...
    def test_shifted_signal_has_correct_lag(self): ...
    def test_uncorrelated_signals_have_low_correlation(self): ...

class TestCRQA:
    def test_identical_signals_have_high_recurrence(self): ...
    def test_random_signals_have_low_recurrence(self): ...

class TestParticipationParity:
    def test_equal_durations_returns_one(self): ...
    def test_single_speaker_returns_near_zero(self): ...

class TestGroupDynamicsEngine:
    async def test_emits_snapshot_after_interval(self): ...
    async def test_snapshot_contains_all_required_fields(self): ...
```

---

## Phase 5: GRPO Emotion Tuning (Architecture)

> Detailed implementation steps deferred. Extends existing GRPO infrastructure.

### Task 5.1: Extend WeightTuner for 5 weights

**Files:**
- Modify: `forgestream/governor/improvement.py:12-72`
- Modify: `forgestream/post_meeting.py:103-139`

**Changes:**

1. `WeightTuner` already handles arbitrary weight dicts — no code change needed.

2. `PostMeetingSynthesis.load_weights()` at line 108 filters keys. Update the filter:

```python
def load_weights(self) -> dict[str, float]:
    weights_file = self.data_dir / "weights.json"
    if weights_file.exists():
        data = json.loads(weights_file.read_text())
        valid_keys = {"knowledge", "verification", "scaffold", "uptake", "engagement"}
        return {k: v for k, v in data.items() if k in valid_keys}
    return Evaluator.DEFAULT_WEIGHTS.copy()
```

3. Add emotion correlation to `compute_auto_score`:

```python
def compute_auto_score(self, events: list[Event]) -> float:
    # ... existing logic ...
    # Add engagement signal
    prosodic = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]
    if prosodic:
        mean_arousal = sum(e.payload.get("arousal", 0.5) for e in prosodic) / len(prosodic)
        engagement_bonus = 0.1 * mean_arousal
    else:
        engagement_bonus = 0.0
    return min(1.0, base_score + engagement_bonus)
```

### Task 5.2: Tone adjustment GRPO

**Files:**
- Create: `forgestream/governor/tone_tuner.py`

**Interface contract:**

```python
# forgestream/governor/tone_tuner.py
"""GRPO tuning of tone adjustment values.

Replaces the hardcoded HESITATION_PENALTY, BACKTRACK_PENALTY, EMPHASIS_BOOST,
EXCITEMENT_BOOST in extraction.py with data-driven values.
"""

class ToneAdjustmentTuner:
    DEFAULT_ADJUSTMENTS = {
        "hesitation_penalty": 0.15,
        "backtrack_penalty": 0.20,
        "emphasis_boost": 0.20,
        "excitement_boost": 0.15,
    }

    def tune(
        self,
        current: dict[str, float],
        events: list[Event],
        human_score: float,
    ) -> dict[str, float]:
        """GRPO-perturb tone adjustments and select best-performing variant.

        Correlates: for each claim, compute score with each adjustment set.
        Claims where acoustic features confirm the tone marker (e.g., high jitter
        confirms hesitation) get higher weight in the correlation.
        """
```

---

## Phase 6: Dashboard Emotion Visualization (Architecture)

> Detailed implementation steps deferred. Defines data endpoints and visualization components.

### Task 6.1: Emotion API endpoints

**Files:**
- Modify: `forgestream/dashboard/` (add endpoints)

**New endpoints:**

```python
# GET /api/emotion/timeline?session_id=...
# Returns: list of {timestamp_ms, speaker_id, arousal, valence, dominance, emotion_tag}

# GET /api/emotion/entrainment?session_id=...
# Returns: list of ENTRAINMENT_SNAPSHOT payloads

# GET /api/emotion/speakers?session_id=...
# Returns: per-speaker summary {speaker_id, total_duration, mean_arousal, mean_valence, dominance_rank}
```

### Task 6.2: Dashboard visualizations

**Components:**
1. **Emotion timeline** — per-speaker arousal/valence tracks overlaid on claim markers (D3 line chart)
2. **Entrainment heatmap** — speaker-pair correlation matrix animated over time (D3 heatmap)
3. **Engagement trajectory** — collective engagement line alongside E(π) (D3 dual-axis chart)
4. **Influence network** — transfer entropy as D3 force-directed graph (if Phase 4 includes transfer entropy)

### Task 6.3: Extend GraphMaterializer for PROSODIC_FEATURE events

**Files:**
- Modify: `forgestream/graph/materializer.py:77-82`

```python
# Add handler for PROSODIC_FEATURE
def _handle_prosodic_feature(self, graph: KnowledgeGraph, event: Event) -> None:
    speaker = event.payload.get("speaker_id", "unknown")
    # Create or update speaker node with latest emotion state
    # Add EMOTIONAL_STATE edge type between speaker and emotion

_handlers = {
    # ... existing ...
    EventType.PROSODIC_FEATURE: _handle_prosodic_feature,
}
```

---

## Phase 7: Cross-Meeting Emotion Corpus (Architecture)

> Detailed implementation steps deferred. Defines persistence and training data format.

### Task 7.1: Audio segment persistence

**Files:**
- Create: `forgestream/emotion/persistence.py`

```python
# forgestream/emotion/persistence.py
"""Persist raw audio segments aligned with prosodic features and claims.

After each meeting, save:
1. Full meeting audio as WAV/FLAC
2. Segment index: claim_id → (start_ms, end_ms) in the audio file
3. Feature vectors aligned with segments
4. Human feedback scores per meeting

This builds a proprietary training corpus over time.
"""

class EmotionCorpus:
    def __init__(self, corpus_dir: str = "data/emotion_corpus") -> None: ...

    def save_meeting_audio(
        self, session_id: str, audio_buffer: AudioRingBuffer
    ) -> str:
        """Save full meeting audio from ring buffer to WAV file."""

    def save_feature_index(
        self, session_id: str, prosodic_events: list[Event], claim_events: list[Event]
    ) -> str:
        """Save aligned feature-claim index as JSON."""

    def get_training_samples(self) -> list[dict]:
        """Load all (audio_segment, features, human_score) triples for fine-tuning."""
```

### Task 7.2: Wire into PostMeetingSynthesis

```python
# In PostMeetingSynthesis.run(), after generating report:
if self.config.emotion_enabled:
    corpus = EmotionCorpus(corpus_dir=str(self.data_dir / "emotion_corpus"))
    corpus.save_meeting_audio(session_id, audio_buffer)
    corpus.save_feature_index(session_id, prosodic_events, claim_events)
```

### Task 7.3: Emotion model fine-tuning pipeline (future)

After 10+ meetings with human feedback scores, fine-tune emotion2vec+seed on the corpus:

```python
# This is a standalone script, not a ForgeStream module
# forgestream/scripts/finetune_emotion.py

# 1. Load corpus: audio segments + aligned features + scores
# 2. Fine-tune emotion2vec+seed with LoRA adapter
# 3. Export fine-tuned model to data/models/emotion_finetuned/
# 4. Update config to use fine-tuned model
```

---

## Critical Execution Principles (Enforced Across All Phases)

### 1. Never block the claim path

The `_send_loop` → Gemini → `_receive_loop` → `process_event` path is latency-critical.
The emotion pipeline runs in a separate async task (`_emotion_extraction_loop`).
Feature extraction runs in a `ThreadPoolExecutor` (CPU-bound work off the event loop).
If the emotion queue backs up, chunks accumulate in the asyncio.Queue — they don't slow Gemini.

### 2. Event-sourced everything

Every prosodic vector, emotion state, and entrainment snapshot is an ECEF event.
Written to PostgreSQL + Firestore (dual-write, same as claims).
Replay works: `GraphMaterializer.materialize(all_events)` includes emotion data.
GRPO can retroactively re-score by replaying events with different weight vectors.

### 3. SOS compliance from day one

The emotion pipeline is a post-write observer (subscribes to EventBus).
The evaluator extension is additive (new weight, existing weights rebalanced).
AxiomChecker monitors emotional_engagement the same as other metrics:
- Axiom 1 (Monotone): 3 consecutive declining engagement windows → violation
- Axiom 2 (Bounded Step): emotional engagement delta within epsilon
- Axiom 3 (Constraint): previously verified emotional patterns preserved

### 4. GRPO discovers what matters

The hardcoded tone adjustments (hesitation -0.15, emphasis +0.20) become GRPO-tunable.
The engagement weight starts at 0.15 — GRPO will discover its optimal value for YOUR meetings.
Cross-meeting analysis reveals correlations: "meetings with engagement > 0.7 produce 2.3x more verified findings."

### 5. Phase incrementally

Each phase produces independently testable, deployable software:
- After Phase 0+1: Prosodic features flow as events. Dashboard shows raw feature data.
- After Phase 2+3: Claims are enriched. Evaluator is emotion-aware.
- After Phase 4: Group dynamics are measured. Entrainment snapshots appear.
- After Phase 5: GRPO tunes everything. System self-improves.
- After Phase 6+7: Full visualization. Training corpus grows.

---

## Verification Criteria

After Phase 0+1 is complete, these must all pass:

```bash
# Unit tests
cd /Users/mghome/projects/forgestream
python3 -m pytest tests/emotion/ -v

# Full regression suite
python3 -m pytest tests/ -q \
  --ignore=tests/events/test_store.py \
  --ignore=tests/events/test_subscribe.py \
  -k "not writes_to_store and not milestone_a and not full_pipeline_with"

# Manual smoke test: run mock meeting, verify PROSODIC_FEATURE events appear
python3 -c "
import asyncio
from forgestream.emotion.buffer import AudioRingBuffer
from forgestream.emotion.features import PraatExtractor
import numpy as np

# Generate test signal
t = np.linspace(0, 1.0, 16000, endpoint=False)
signal = (10000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)

# Extract features
extractor = PraatExtractor()
result = extractor.extract(signal)
print(f'F0: {result[\"f0_mean\"]:.1f} Hz')
print(f'Energy: {result[\"energy_rms\"]:.4f}')
print(f'Jitter: {result[\"jitter_local\"]:.4f}')
print(f'HNR: {result[\"hnr\"]:.1f} dB')
print('PASS: Feature extraction working')
"
```
