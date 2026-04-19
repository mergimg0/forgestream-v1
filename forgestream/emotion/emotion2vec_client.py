"""Client for RunPod emotion2vec+large offline classification.

Sends audio segments to the RunPod serverless endpoint for high-accuracy
emotion labeling using emotion2vec+large. Unlike SenseVoice (hardcoded
0.8/0.5 confidence), emotion2vec returns real probability distributions.

Used in PostMeetingSynthesis to re-label corpus audio after a meeting.
"""

from __future__ import annotations

import base64
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class Emotion2VecClient:
    """Client for RunPod emotion2vec classification endpoint.

    Parameters:
        runpod_endpoint: RunPod serverless base URL (e.g. https://api.runpod.ai/v2/<id>).
        runpod_api_key: API key for auth. Falls back to env/secrets file.
        timeout: Request timeout in seconds (emotion2vec is slower than CRQA).
    """

    def __init__(
        self,
        runpod_endpoint: str = "",
        runpod_api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = runpod_endpoint
        self._api_key = runpod_api_key or self._load_api_key()
        self._timeout = timeout

    @staticmethod
    def _load_api_key() -> str:
        key = os.environ.get("RUNPOD_API_KEY", "")
        if key:
            return key
        try:
            secrets_path = os.path.join(
                os.path.dirname(__file__), "..", "..", ".secrets", "runpod_api_key.txt"
            )
            with open(secrets_path) as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    async def classify(self, signal: np.ndarray) -> dict:
        """Classify emotion from int16 PCM audio via RunPod emotion2vec.

        Parameters:
            signal: 1-D int16 numpy array of raw PCM samples (16kHz mono).

        Returns:
            {"tag": "happy", "confidence": 0.87, "scores": {...}, "compute_ms": 42}
            or {"tag": "unknown", "confidence": 0.0, "error": "..."} on failure.
        """
        if not self._endpoint or not self._api_key:
            return {"tag": "unknown", "confidence": 0.0, "error": "no endpoint configured"}

        import httpx

        audio_b64 = base64.b64encode(signal.tobytes()).decode("ascii")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._endpoint}/runsync",
                    json={
                        "input": {
                            "operation": "emotion_classify",
                            "audio_b64": audio_b64,
                        }
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                envelope = response.json()

                if envelope.get("status") == "FAILED":
                    return {"tag": "unknown", "confidence": 0.0, "error": envelope.get("error")}

                return envelope.get("output", {"tag": "unknown", "confidence": 0.0})

        except Exception as exc:
            logger.warning("emotion2vec classify failed: %s", exc)
            return {"tag": "unknown", "confidence": 0.0, "error": str(exc)}

    async def classify_segments(
        self,
        audio_bytes: bytes,
        segment_duration_s: float = 3.0,
        sample_rate: int = 16000,
    ) -> list[dict]:
        """Classify emotion for each segment of a full meeting audio.

        Parameters:
            audio_bytes: Raw int16 PCM bytes (16kHz mono).
            segment_duration_s: Duration of each segment in seconds.
            sample_rate: Audio sample rate.

        Returns:
            List of {"offset_ms": int, "tag": str, "confidence": float, "scores": dict}
        """
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        segment_samples = int(segment_duration_s * sample_rate)
        results = []

        for i in range(0, len(samples), segment_samples):
            segment = samples[i : i + segment_samples]
            if len(segment) < sample_rate:  # skip < 1s segments
                continue

            offset_ms = int(i / sample_rate * 1000)
            result = await self.classify(segment)
            result["offset_ms"] = offset_ms
            results.append(result)

        return results
