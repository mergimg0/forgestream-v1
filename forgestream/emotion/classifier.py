"""Categorical emotion classification using SenseVoice-Small.

Wraps FunASR's SenseVoice model. Returns emotion tags:
angry, disgusted, fearful, happy, neutral, other, sad, surprised.

Optional — loaded only when funasr is installed and config enables it.
Falls back gracefully when funasr is unavailable.
"""

from __future__ import annotations

import logging
import os
import tempfile
import wave

import numpy as np

logger = logging.getLogger(__name__)

_EMOTION_MAP: dict[str, str] = {
    "<|HAPPY|>": "happy",
    "<|SAD|>": "sad",
    "<|ANGRY|>": "angry",
    "<|NEUTRAL|>": "neutral",
    "<|FEARFUL|>": "fearful",
    "<|DISGUSTED|>": "disgusted",
    "<|SURPRISED|>": "surprised",
    "<|OTHER|>": "other",
}


class SenseVoiceClassifier:
    """Categorical emotion classifier using SenseVoice-Small.

    Lazy model loading — the FunASR AutoModel is not loaded until the first
    call to ``classify()``. This keeps startup time low when the feature is
    optionally enabled.

    Parameters:
        model_name: FunASR model identifier (default: iic/SenseVoiceSmall).
        sample_rate: Expected audio sample rate in Hz (default: 16000).
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
        """Return True if the funasr package is importable."""
        try:
            import funasr  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self) -> None:
        """Lazily load the SenseVoice model on first use."""
        if self._model is not None:
            return
        from funasr import AutoModel

        self._model = AutoModel(model=self._model_name)
        logger.info("SenseVoice model loaded: %s", self._model_name)

    def classify(self, signal: np.ndarray) -> tuple[str, float]:
        """Classify emotion from an int16 numpy array at ``sample_rate`` Hz.

        Writes a temporary WAV file because FunASR expects a file path, then
        parses the ``<|TAG|>`` tokens embedded in SenseVoice's output text.

        Parameters:
            signal: 1-D int16 numpy array of raw PCM samples.

        Returns:
            A ``(emotion_tag, confidence)`` tuple.  ``emotion_tag`` is one of
            angry / disgusted / fearful / happy / neutral / other / sad /
            surprised / unknown.  ``confidence`` is 0.8 for a recognised tag
            or 0.5 for the neutral fallback.
        """
        self._ensure_model()

        tmp_path: str | None = None
        try:
            # Write int16 PCM to a temporary WAV file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as fh:
                tmp_path = fh.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes = int16
                wf.setframerate(self._sample_rate)
                wf.writeframes(signal.tobytes())

            result = self._model.generate(
                input=tmp_path,
                language="auto",
            )
            if result and len(result) > 0:
                text = result[0].get("text", "")
                return self._parse_emotion(text)
            return "unknown", 0.0

        except Exception as exc:
            logger.warning("SenseVoice classification failed: %s", exc)
            return "unknown", 0.0
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _parse_emotion(text: str) -> tuple[str, float]:
        """Extract emotion tag from SenseVoice output text.

        SenseVoice embeds tags like ``<|HAPPY|>``, ``<|NEUTRAL|>``, etc. in
        the ASR transcript.  If no recognised tag is found, falls back to
        ``("neutral", 0.5)``.

        Parameters:
            text: Raw output string from SenseVoice's generate() call.

        Returns:
            ``(emotion_label, confidence)`` tuple.
        """
        upper = text.upper()
        for tag, label in _EMOTION_MAP.items():
            if tag in upper:
                return label, 0.8
        return "neutral", 0.5
