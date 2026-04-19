"""CRQA compute router with RunPod GPU and local CPU fallback.

Routes CRQA computation to a RunPod serverless endpoint for surrogate-
validated results. Falls back to local numpy CRQA (without surrogates)
when RunPod is unavailable. Circuit breaker prevents hammering a dead endpoint.

Supports both RunPod serverless (api.runpod.ai/v2/<id>/runsync) and
direct pod endpoints (<pod-id>-8000.proxy.runpod.net/crqa/validate).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CIRCUIT_OPEN_THRESHOLD = 3
CIRCUIT_SKIP_CYCLES = 5


@dataclass
class CRQAResult:
    """Result of a CRQA computation."""

    det: float
    tt: float
    entr: float
    lam: float
    rr: float
    surrogate_validated: bool = False


class CRQAComputeRouter:
    """Routes CRQA computation to RunPod or local fallback.

    Parameters:
        runpod_endpoint: URL of the RunPod CRQA endpoint. Empty = local only.
            Supports both serverless (api.runpod.ai) and direct pod URLs.
        timeout: Per-request timeout in seconds.
        runpod_api_key: API key for RunPod serverless auth. Falls back to
            RUNPOD_API_KEY env var or .secrets/runpod_api_key.txt.
    """

    def __init__(
        self,
        runpod_endpoint: str = "",
        timeout: float = 4.0,
        runpod_api_key: str = "",
    ) -> None:
        self._endpoint = runpod_endpoint
        self._timeout = timeout
        self._api_key = runpod_api_key or self._load_api_key()
        self._is_serverless = "api.runpod.ai" in runpod_endpoint
        self._consecutive_failures = 0
        self._circuit_open = False
        self._skip_remaining = 0

    @staticmethod
    def _load_api_key() -> str:
        """Load RunPod API key from env or secrets file."""
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
                    logger.info(
                        "Circuit breaker opened after %d failures",
                        self._consecutive_failures,
                    )

        return self._local_compute(f0_a, f0_b, radius)

    def _should_skip(self) -> bool:
        """Check if we should skip RunPod due to circuit breaker."""
        if not self._circuit_open:
            return False
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return True
        return False

    async def _runpod_compute(
        self,
        f0_a: list[float],
        f0_b: list[float],
        embedding_dim: int,
        time_delay: int,
        radius: float,
    ) -> CRQAResult:
        """Send CRQA request to RunPod endpoint.

        Supports two modes:
        - Serverless: POST to /runsync with {"input": {...}} wrapper
        - Direct pod: POST to /crqa/validate with flat payload
        """
        import httpx

        payload = {
            "f0_a": f0_a,
            "f0_b": f0_b,
            "params": {
                "embedding_dim": embedding_dim,
                "time_delay": time_delay,
                "radius": radius,
                "n_surrogates": 20,
            },
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if self._is_serverless:
                url = f"{self._endpoint}/runsync"
                headers = {"Authorization": f"Bearer {self._api_key}"}
                response = await client.post(
                    url,
                    json={"input": payload},
                    headers=headers,
                )
                response.raise_for_status()
                envelope = response.json()
                if envelope.get("status") == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {envelope.get('error')}")
                data = envelope["output"]
            else:
                response = await client.post(
                    f"{self._endpoint}/crqa/validate",
                    json=payload,
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

        if a.std() > 0:
            a = (a - a.mean()) / a.std()
        if b.std() > 0:
            b = (b - b.mean()) / b.std()

        combined_std = np.std(np.concatenate([a, b]))
        if combined_std < 1e-10:
            return CRQAResult(det=0.0, tt=0.0, entr=0.0, lam=0.0, rr=0.0)

        threshold = radius * combined_std * 3
        dist = np.abs(a[:, None] - b[None, :])
        recurrence = dist < threshold

        rr = float(np.mean(recurrence))
        total = recurrence.sum()

        if total == 0:
            return CRQAResult(det=0.0, tt=0.0, entr=0.0, lam=0.0, rr=0.0)

        det = self._compute_determinism(recurrence, total)
        lam = self._compute_laminarity(recurrence, total)

        return CRQAResult(
            det=det, tt=0.0, entr=0.0, lam=lam, rr=rr,
            surrogate_validated=False,
        )

    @staticmethod
    def _compute_determinism(recurrence: np.ndarray, total: int) -> float:
        """Compute %DET: proportion of recurrence points on diagonal lines >= 2."""
        n = recurrence.shape[0]
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

        return min(1.0, diagonal_points / total)

    @staticmethod
    def _compute_laminarity(recurrence: np.ndarray, total: int) -> float:
        """Compute %LAM: proportion of recurrence points on vertical lines >= 2."""
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

        return min(1.0, vertical_points / total)
