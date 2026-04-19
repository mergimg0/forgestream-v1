"""RunPod CRQA endpoint — accepts F0 pairs, returns surrogate-validated metrics.

Deploy as a RunPod serverless endpoint with GPU (A4000 recommended).

Usage:
    uvicorn crqa_endpoint:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
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
    total = int(recurrence.sum())

    if total == 0:
        return {"det": 0.0, "tt": 0.0, "entr": 0.0, "lam": 0.0, "rr": 0.0}

    # Determinism: diagonal lines >= 2
    n = len(a)
    diag_points = 0
    diag_lengths: list[int] = []
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
    vert_lengths: list[int] = []
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
    entr = 0.0
    if diag_lengths:
        total_diag = sum(diag_lengths)
        for ll in set(diag_lengths):
            p = diag_lengths.count(ll) * ll / total_diag
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
