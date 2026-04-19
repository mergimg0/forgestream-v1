"""Tests for transfer entropy computation."""

import numpy as np
import pytest

from forgestream.emotion.transfer_entropy import compute_transfer_entropy, compute_symmetry


class TestTransferEntropy:
    def test_self_predictive_signal_has_low_te(self):
        """A signal that only depends on itself has low TE from another."""
        np.random.seed(42)
        a = np.random.randn(200).tolist()
        b = np.random.randn(200).tolist()
        te = compute_transfer_entropy(a, b, lag=1)
        assert te >= 0.0
        assert te < 0.5

    def test_causal_signal_has_higher_te(self):
        """B caused by A should show higher TE(A->B) than TE(B->A)."""
        np.random.seed(42)
        a = np.random.randn(200).tolist()
        b = [0.0] + [0.8 * a[i] + 0.2 * np.random.randn() for i in range(199)]
        te_a_to_b = compute_transfer_entropy(a, b, lag=1)
        te_b_to_a = compute_transfer_entropy(b, a, lag=1)
        assert te_a_to_b > te_b_to_a

    def test_short_signals_return_zero(self):
        te = compute_transfer_entropy([1.0, 2.0], [3.0, 4.0], lag=1)
        assert te == 0.0

    def test_identical_signals(self):
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 100))]
        te = compute_transfer_entropy(signal, signal, lag=1)
        assert te >= 0.0


class TestComputeSymmetry:
    def test_symmetric_mutual_influence(self):
        sym = compute_symmetry(0.3, 0.3)
        assert sym == pytest.approx(1.0)

    def test_asymmetric(self):
        asym = compute_symmetry(0.5, 0.1)
        assert asym < 0.5

    def test_zero_te(self):
        zero = compute_symmetry(0.0, 0.0)
        assert zero == 1.0
