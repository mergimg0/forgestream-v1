# Speaker Diarization Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `speaker_id: "unknown"` with real per-speaker identification using pyannote-audio, enabling meaningful group dynamics (TLCC, CRQA, participation parity) across actual speakers.

**Architecture:** A `SpeakerDiarizer` wraps pyannote-audio's speaker-diarization-3.1 pipeline. It receives audio chunks from the AudioRingBuffer, maintains a rolling diarization state, and labels each PROSODIC_FEATURE event with the active speaker ID. The diarizer runs asynchronously alongside the EmotionExtractor — both consume from the same ring buffer but serve different purposes.

**Tech Stack:** pyannote-audio 3.1 (HuggingFace, requires `HUGGING_FACE_HUB_TOKEN`), torch, torchaudio

---

## Why pyannote-audio (Not WhisperX)

| Factor | pyannote-audio | WhisperX |
|--------|---------------|----------|
| Diarization quality | SOTA (DER ~3-5% on AMI) | Good but relies on pyannote internally |
| Streaming support | Near-RT via `pyannote.audio.pipelines.SpeakerDiarization` on chunks | Batch-oriented, 70x real-time but not streaming |
| Independence | Standalone | Bundles Whisper ASR (redundant — we use Gemini for ASR) |
| Embedding model | Built-in speaker embeddings | Also uses pyannote embeddings |

pyannote-audio is the correct choice: it's the diarization engine WhisperX wraps anyway, and we don't need Whisper's ASR.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `forgestream/emotion/diarizer.py` | `SpeakerDiarizer` — wraps pyannote-audio, labels chunks with speaker IDs |
| `tests/emotion/test_diarizer.py` | Diarizer tests |

### Modified files

| File | Changes |
|------|---------|
| `forgestream/emotion/extractor.py` | Accept optional `SpeakerDiarizer`, pass speaker_id to payload |
| `forgestream/live_stream.py` | Initialize `SpeakerDiarizer` when config enables it |
| `forgestream/config.py` | Add `diarization_enabled`, `huggingface_token` fields |
| `pyproject.toml` | Add `diarization` optional dependency group |

---

## Task 1: Add diarization dependencies and config

**Files:**
- Modify: `pyproject.toml`
- Modify: `forgestream/config.py`

- [ ] **Step 1: Add dependency group**

```toml
diarization = [
    "pyannote.audio>=3.1",
    "torch>=2.0",
    "torchaudio>=2.0",
]
```

- [ ] **Step 2: Add config fields**

```python
# In ForgeStreamConfig
diarization_enabled: bool = False  # opt-in (requires HF token + GPU recommended)
huggingface_token: str = ""        # HF_TOKEN for pyannote model access
```

- [ ] **Step 3: Commit**

---

## Task 2: SpeakerDiarizer

**Files:**
- Create: `forgestream/emotion/diarizer.py`
- Test: `tests/emotion/test_diarizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/emotion/test_diarizer.py
class TestSpeakerDiarizer:
    def test_label_silence_returns_unknown(self): ...
    def test_single_speaker_returns_consistent_id(self): ...
    def test_diarizer_respects_max_speakers(self): ...
    def test_get_active_speaker_at_timestamp(self): ...
```

- [ ] **Step 2: Implement SpeakerDiarizer**

```python
# forgestream/emotion/diarizer.py
"""Speaker diarization using pyannote-audio.

Labels audio chunks with speaker IDs. Uses pyannote's speaker-diarization-3.1
pipeline with incremental processing: each new chunk extends the internal
audio accumulator and re-runs diarization on the recent window.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

AUTHOR = "speaker_diarizer"


class SpeakerDiarizer:
    """Identifies who is speaking in each audio chunk.

    Parameters:
        huggingface_token: HuggingFace token for pyannote model access.
        max_speakers: Maximum number of speakers to detect.
        window_seconds: Audio window to analyze for diarization.
    """

    def __init__(
        self,
        huggingface_token: str,
        max_speakers: int = 6,
        window_seconds: float = 30.0,
        sample_rate: int = 16000,
    ) -> None:
        self._max_speakers = max_speakers
        self._window_seconds = window_seconds
        self._sample_rate = sample_rate
        self._pipeline = None
        self._token = huggingface_token

        # Rolling audio buffer for diarization context
        max_samples = int(window_seconds * sample_rate)
        self._audio_buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0

        # Current diarization result: list of (start_s, end_s, speaker_id)
        self._segments: list[tuple[float, float, str]] = []

    def _ensure_pipeline(self) -> None:
        """Lazily load pyannote pipeline."""
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._token,
        )

    def add_chunk(self, chunk_pcm: bytes) -> None:
        """Add a PCM int16 chunk to the internal buffer."""
        signal = np.frombuffer(chunk_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(signal)
        buf_len = len(self._audio_buffer)

        if self._write_pos + n <= buf_len:
            self._audio_buffer[self._write_pos:self._write_pos + n] = signal
        else:
            # Shift buffer left and append
            shift = self._write_pos + n - buf_len
            self._audio_buffer[:-shift] = self._audio_buffer[shift:]
            self._audio_buffer[-n:] = signal
            self._write_pos = buf_len - n

        self._write_pos += n
        self._total_samples += n

    def update_diarization(self) -> None:
        """Re-run diarization on the current audio buffer."""
        self._ensure_pipeline()
        import torch
        import torchaudio

        # Prepare audio tensor
        valid = min(self._write_pos, len(self._audio_buffer))
        waveform = torch.tensor(
            self._audio_buffer[:valid], dtype=torch.float32
        ).unsqueeze(0)

        # Run pipeline
        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": self._sample_rate},
            max_speakers=self._max_speakers,
        )

        # Extract segments
        self._segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            self._segments.append((turn.start, turn.end, speaker))

    def get_speaker_at(self, timestamp_seconds: float) -> str:
        """Get the active speaker at a given timestamp within the buffer.

        Returns "unknown" if no speaker is active or diarization hasn't run.
        """
        # Adjust timestamp relative to buffer start
        buffer_duration = min(self._write_pos, len(self._audio_buffer)) / self._sample_rate
        buffer_start = max(0, self._total_samples / self._sample_rate - buffer_duration)
        relative_ts = timestamp_seconds - buffer_start

        for start, end, speaker in self._segments:
            if start <= relative_ts <= end:
                return speaker
        return "unknown"

    def get_current_speaker(self) -> str:
        """Get the speaker at the most recent timestamp."""
        if not self._segments:
            return "unknown"
        return self._segments[-1][2]
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

---

## Task 3: Wire into EmotionExtractor

**Files:**
- Modify: `forgestream/emotion/extractor.py`
- Modify: `forgestream/live_stream.py`

- [ ] **Step 1: Add optional diarizer to EmotionExtractor**

```python
# In EmotionExtractor.__init__, add parameter:
#   diarizer: SpeakerDiarizer | None = None

# In _emit_event, replace "unknown" with:
#   speaker_id = self._diarizer.get_current_speaker() if self._diarizer else "unknown"
```

- [ ] **Step 2: Wire in GeminiLiveStream**

```python
# In GeminiLiveStream.__init__, if diarization_enabled:
#   self._diarizer = SpeakerDiarizer(huggingface_token=config.huggingface_token)
# Pass diarizer to EmotionExtractor

# In _send_loop, after writing chunk to buffer:
#   self._diarizer.add_chunk(chunk)
#   # Periodically (every 5s): self._diarizer.update_diarization()
```

- [ ] **Step 3: Run full test suite, verify no regressions**
- [ ] **Step 4: Commit**

---

## Verification

```bash
# Unit tests (mock pyannote — don't require GPU)
python3 -m pytest tests/emotion/test_diarizer.py -v

# Integration test (requires HF token + pyannote installed)
HUGGING_FACE_HUB_TOKEN=hf_xxx python3 -c "
from forgestream.emotion.diarizer import SpeakerDiarizer
d = SpeakerDiarizer(huggingface_token='hf_xxx')
print('Pipeline loaded:', d._pipeline is not None)
"
```
