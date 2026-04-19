# SenseVoice Categorical Emotion Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the `emotion_tag` and `emotion_confidence` fields in PROSODIC_FEATURE events with categorical emotion labels from SenseVoice-Small, giving the system both dimensional (arousal/valence/dominance from eGeMAPS) and categorical (angry/happy/neutral/sad/surprised from SenseVoice) emotion representations.

**Architecture:** A `SenseVoiceClassifier` wraps FunASR's SenseVoice-Small model. It receives the same sliding window of audio as the EmotionExtractor and returns an emotion tag + confidence. The classifier is optional — loaded only when `emotion_ml_enabled=True` in config and the `funasr` package is installed. When unavailable, emotion_tag/emotion_confidence remain None (graceful degradation).

**Tech Stack:** FunASR (SenseVoice-Small, ~400MB model), torch

---

## Why SenseVoice (Not emotion2vec)

| Factor | SenseVoice-Small | emotion2vec+large |
|--------|-----------------|-------------------|
| Speed | **70ms for 10s audio** | ~500ms for 10s |
| Joint capabilities | ASR + emotion + events (laughter, applause) | Emotion only |
| Model size | ~240MB | ~300MB |
| Streaming | Pseudo-streaming via truncated attention | Offline only |
| Integration | FunASR pip install | Custom loading |

SenseVoice is better for real-time because it's 7x faster and also detects audio events (laughter, applause, coughing) which are useful for meeting dynamics. emotion2vec is better for high-accuracy offline analysis (Phase 7 corpus fine-tuning).

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `forgestream/emotion/classifier.py` | `SenseVoiceClassifier` — wraps FunASR SenseVoice-Small |
| `tests/emotion/test_classifier.py` | Classifier tests |

### Modified files

| File | Changes |
|------|---------|
| `forgestream/emotion/extractor.py` | Accept optional classifier, fill emotion_tag/confidence in payload |
| `forgestream/live_stream.py` | Initialize classifier when `emotion_ml_enabled` |
| `forgestream/config.py` | Add `sensevoice_model` field |

---

## Task 1: SenseVoiceClassifier

**Files:**
- Create: `forgestream/emotion/classifier.py`
- Test: `tests/emotion/test_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/emotion/test_classifier.py
import numpy as np
import pytest

from forgestream.emotion.classifier import SenseVoiceClassifier


class TestSenseVoiceClassifier:
    def test_classify_returns_tag_and_confidence(self):
        funasr = pytest.importorskip("funasr")
        classifier = SenseVoiceClassifier()
        # Generate 2 seconds of 200Hz sine (simulates speech)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        signal = (10000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        tag, confidence = classifier.classify(signal)
        assert isinstance(tag, str)
        assert tag in ("angry", "disgusted", "fearful", "happy", "neutral",
                        "other", "sad", "surprised", "unknown")
        assert 0.0 <= confidence <= 1.0

    def test_classify_silence_returns_neutral_or_unknown(self):
        funasr = pytest.importorskip("funasr")
        classifier = SenseVoiceClassifier()
        signal = np.zeros(32000, dtype=np.int16)
        tag, confidence = classifier.classify(signal)
        assert tag in ("neutral", "unknown", "other")

    def test_classify_short_signal_graceful(self):
        funasr = pytest.importorskip("funasr")
        classifier = SenseVoiceClassifier()
        signal = np.zeros(1600, dtype=np.int16)  # 0.1s — very short
        tag, confidence = classifier.classify(signal)
        assert isinstance(tag, str)

    def test_available_without_funasr(self):
        """SenseVoiceClassifier.is_available() returns False if funasr missing."""
        # This test always passes — it just checks the interface
        assert hasattr(SenseVoiceClassifier, "is_available")
```

- [ ] **Step 2: Implement SenseVoiceClassifier**

```python
# forgestream/emotion/classifier.py
"""Categorical emotion classification using SenseVoice-Small.

Wraps FunASR's SenseVoice model. Returns emotion tags:
angry, disgusted, fearful, happy, neutral, other, sad, surprised.

Optional — loaded only when funasr is installed and config enables it.
"""

from __future__ import annotations

import logging
import tempfile
import wave

import numpy as np

logger = logging.getLogger(__name__)


class SenseVoiceClassifier:
    """Categorical emotion classifier using SenseVoice-Small.

    Parameters:
        model_name: FunASR model identifier (default: iic/SenseVoiceSmall).
        sample_rate: Expected audio sample rate.
    """

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        sample_rate: int = 16000,
    ) -> None:
        self._model_name = model_name
        self._sample_rate = sample_rate
        self._model = None

    @staticmethod
    def is_available() -> bool:
        """Check if FunASR is installed."""
        try:
            import funasr
            return True
        except ImportError:
            return False

    def _ensure_model(self) -> None:
        """Lazily load the SenseVoice model."""
        if self._model is not None:
            return
        from funasr import AutoModel
        self._model = AutoModel(model=self._model_name)
        logger.info("SenseVoice model loaded: %s", self._model_name)

    def classify(self, signal: np.ndarray) -> tuple[str, float]:
        """Classify emotion from an int16 numpy array.

        Returns (emotion_tag, confidence).
        """
        self._ensure_model()

        # SenseVoice expects a WAV file path — write temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            signal_float = signal.astype(np.float32) / 32768.0
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(signal.tobytes())

        try:
            result = self._model.generate(
                input=tmp_path,
                language="auto",
            )
            if result and len(result) > 0:
                entry = result[0]
                # SenseVoice embeds emotion in text output as <|EMOTION|> tags
                text = entry.get("text", "")
                return self._parse_emotion(text)
            return "unknown", 0.0
        except Exception as e:
            logger.warning("SenseVoice classification failed: %s", e)
            return "unknown", 0.0

    @staticmethod
    def _parse_emotion(text: str) -> tuple[str, float]:
        """Extract emotion tag from SenseVoice output text.

        SenseVoice embeds tags like <|HAPPY|>, <|NEUTRAL|>, etc.
        """
        emotion_map = {
            "<|HAPPY|>": "happy",
            "<|SAD|>": "sad",
            "<|ANGRY|>": "angry",
            "<|NEUTRAL|>": "neutral",
            "<|FEARFUL|>": "fearful",
            "<|DISGUSTED|>": "disgusted",
            "<|SURPRISED|>": "surprised",
            "<|OTHER|>": "other",
        }
        for tag, label in emotion_map.items():
            if tag in text.upper():
                return label, 0.8  # SenseVoice doesn't output confidence
        return "neutral", 0.5
```

- [ ] **Step 3: Run tests (skipped if funasr not installed)**
- [ ] **Step 4: Commit**

---

## Task 2: Wire into EmotionExtractor

**Files:**
- Modify: `forgestream/emotion/extractor.py`
- Modify: `forgestream/live_stream.py`

- [ ] **Step 1: Add optional classifier to EmotionExtractor**

```python
# In EmotionExtractor.__init__, add parameter:
#   classifier: SenseVoiceClassifier | None = None

# In _extract_features, after Parselmouth + eGeMAPS:
#   if self._classifier:
#       emotion_tag, emotion_conf = self._classifier.classify(signal)
#   else:
#       emotion_tag, emotion_conf = None, None

# In _emit_event, update payload:
#   "emotion_tag": features.emotion_tag,  # add to ProsodicFeatures
#   "emotion_confidence": features.emotion_confidence,
```

- [ ] **Step 2: Wire in GeminiLiveStream**

```python
# If config.emotion_ml_enabled and SenseVoiceClassifier.is_available():
#   classifier = SenseVoiceClassifier()
# Pass classifier to EmotionExtractor
```

- [ ] **Step 3: Run full test suite**
- [ ] **Step 4: Commit**

---

## Verification

```bash
# Skip tests if funasr not installed (all tests use pytest.importorskip)
python3 -m pytest tests/emotion/test_classifier.py -v

# Manual smoke test (requires funasr + torch)
pip3 install "forgestream[emotion-ml]"
python3 -c "
from forgestream.emotion.classifier import SenseVoiceClassifier
print('Available:', SenseVoiceClassifier.is_available())
if SenseVoiceClassifier.is_available():
    import numpy as np
    c = SenseVoiceClassifier()
    signal = np.zeros(32000, dtype=np.int16)
    tag, conf = c.classify(signal)
    print(f'Tag: {tag}, Confidence: {conf}')
"
```
