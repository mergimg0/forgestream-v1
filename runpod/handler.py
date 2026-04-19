"""RunPod serverless handler — multi-operation: CRQA + emotion2vec.

Supports two operations via the "operation" field in input:
  - "crqa" (default): surrogate-validated CRQA metrics
  - "emotion_classify": emotion2vec+large categorical emotion classification

RunPod serverless calls handler(event) with event["input"].
"""

from __future__ import annotations

import base64
import math
import time

import numpy as np
import runpod

# Lazy-loaded emotion2vec model (heavy, only load on first classify call)
_emotion_model = None


def _get_emotion_model():
    """Lazily load emotion2vec+large via FunASR."""
    global _emotion_model
    if _emotion_model is None:
        from funasr import AutoModel
        _emotion_model = AutoModel(model="iic/emotion2vec_plus_large")
    return _emotion_model


# ---------------------------------------------------------------------------
# CRQA computation (existing)
# ---------------------------------------------------------------------------

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


def _handle_crqa(inp: dict) -> dict:
    """Handle CRQA validation request."""
    start = time.monotonic()

    a = np.array(inp["f0_a"], dtype=np.float64)
    b = np.array(inp["f0_b"], dtype=np.float64)
    params = inp.get("params", {})
    radius = params.get("radius", 0.25)
    n_surrogates = params.get("n_surrogates", 20)

    if a.std() > 0:
        a = (a - a.mean()) / a.std()
    if b.std() > 0:
        b = (b - b.mean()) / b.std()

    real = _compute_crqa(a, b, radius)

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

    return {
        "real": real,
        "surrogate_stats": {
            "det_mean": round(det_mean, 6),
            "det_std": round(det_std, 6),
            "tt_mean": round(tt_mean, 4),
            "tt_std": round(tt_std, 4),
        },
        "significant": {
            "det": real["det"] > det_mean + 2 * det_std if det_std > 0 else False,
            "tt": real["tt"] > tt_mean + 2 * tt_std if tt_std > 0 else False,
        },
        "compute_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Emotion classification via emotion2vec+large
# ---------------------------------------------------------------------------

def _handle_emotion_classify(inp: dict) -> dict:
    """Classify emotion from base64-encoded audio using emotion2vec+large.

    Input: {"audio_b64": "<base64 int16 PCM 16kHz>", "sample_rate": 16000}
    Output: {"tag": "happy", "confidence": 0.87, "scores": {...}, "compute_ms": 42}

    Unlike SenseVoice (hardcoded 0.8/0.5), emotion2vec outputs real
    probability distributions — confidence values are actual model output.
    """
    start = time.monotonic()

    audio_bytes = base64.b64decode(inp["audio_b64"])
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    model = _get_emotion_model()
    result = model.generate(audio, output_dir=None, granularity="utterance")

    # emotion2vec returns scores for each emotion category
    # Labels: angry, disgusted, fearful, happy, neutral, other, sad, surprised
    labels = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised"]

    if result and len(result) > 0:
        scores_raw = result[0].get("scores", [])
        if scores_raw and len(scores_raw) > 0:
            probs = scores_raw[0] if isinstance(scores_raw[0], list) else scores_raw
            scores = {labels[i]: round(float(p), 4) for i, p in enumerate(probs) if i < len(labels)}
            best_idx = int(np.argmax(probs[:len(labels)]))
            tag = labels[best_idx]
            confidence = round(float(probs[best_idx]), 4)
        else:
            scores = {}
            tag = "unknown"
            confidence = 0.0
    else:
        scores = {}
        tag = "unknown"
        confidence = 0.0

    elapsed = int((time.monotonic() - start) * 1000)

    return {
        "tag": tag,
        "confidence": confidence,
        "scores": scores,
        "compute_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Main handler (operation router)
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """RunPod serverless entry point.

    Routes to the appropriate handler based on input["operation"]:
      - "crqa" (default): CRQA validation
      - "emotion_classify": emotion2vec classification
    """
    inp = event["input"]
    operation = inp.pop("operation", "crqa")

    if operation == "crqa":
        return _handle_crqa(inp)
    elif operation == "emotion_classify":
        return _handle_emotion_classify(inp)
    else:
        return {"error": f"Unknown operation: {operation}"}


runpod.serverless.start({"handler": handler})
