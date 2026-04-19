"""Tests for CRQAComputeRouter — circuit breaker + RunPod/local routing."""

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
        assert result.surrogate_validated is False

    @pytest.mark.asyncio
    async def test_local_with_short_signals(self):
        router = CRQAComputeRouter(runpod_endpoint="", timeout=4.0)
        result = await router.compute([1.0, 2.0], [3.0, 4.0])
        assert result.det == 0.0


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_3_failures(self):
        router = CRQAComputeRouter(
            runpod_endpoint="http://fake:8000", timeout=0.1
        )
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 50))]
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
        router._skip_remaining = 0
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 50))]
        result = await router.compute(signal, signal)
        assert isinstance(result, CRQAResult)
