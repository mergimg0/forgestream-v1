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

    TE(source -> target) = H(target_future | target_past) - H(target_future | target_past, source_past)

    Returns transfer entropy in bits (>= 0).
    """
    if len(source) < lag + 2 or len(target) < lag + 2:
        return 0.0

    src = np.array(source, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)

    n = min(len(src), len(tgt))
    src = src[:n]
    tgt = tgt[:n]

    src_bins = _discretize(src, n_bins)
    tgt_bins = _discretize(tgt, n_bins)

    tgt_future = tgt_bins[lag:]
    tgt_past = tgt_bins[:-lag]
    src_past = src_bins[:-lag]

    m = len(tgt_future)
    if m < 10:
        return 0.0

    h_tf_tp = _conditional_entropy(tgt_future, tgt_past, n_bins)
    joint_past = tgt_past * n_bins + src_past
    h_tf_tp_sp = _conditional_entropy(tgt_future, joint_past, n_bins)

    return max(0.0, h_tf_tp - h_tf_tp_sp)


def compute_symmetry(te_a_to_b: float, te_b_to_a: float) -> float:
    """Compute symmetry score from bidirectional transfer entropy.

    Returns 1.0 for perfect symmetry, 0.0 for fully one-directional.
    """
    total = te_a_to_b + te_b_to_a
    if total < 1e-10:
        return 1.0
    asymmetry = abs(te_a_to_b - te_b_to_a) / total
    return 1.0 - asymmetry


def _discretize(series: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize a continuous series into integer bins."""
    if series.std() < 1e-10:
        return np.zeros(len(series), dtype=np.int64)
    z = (series - series.mean()) / series.std()
    clipped = np.clip(z, -3, 3)
    bins = ((clipped + 3) / 6 * (n_bins - 1)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def _conditional_entropy(
    x: np.ndarray, condition: np.ndarray, n_bins: int
) -> float:
    """Compute H(X | condition) using histogram counts."""
    unique_conditions = np.unique(condition)
    total = len(x)
    h = 0.0
    for c in unique_conditions:
        mask = condition == c
        p_c = mask.sum() / total
        if p_c == 0:
            continue
        x_given_c = x[mask]
        counts = np.bincount(x_given_c, minlength=n_bins)
        probs = counts / counts.sum()
        for p in probs:
            if p > 0:
                h -= p_c * p * math.log2(p)
    return h
