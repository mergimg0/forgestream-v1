"""Tests for prosodic feature extraction using openSMILE and Parselmouth."""

import numpy as np
import pytest

from forgestream.emotion.features import (
    EGeMAPSExtractor,
    PraatExtractor,
    ProsodicFeatures,
    derive_dimensional_emotion,
)


def _make_sine_wave(
    freq: float = 200.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 10000.0,
) -> np.ndarray:
    """Generate a sine wave as int16 samples (simulates voiced speech)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal


class TestProsodicFeatures:
    def test_dataclass_fields(self):
        f = ProsodicFeatures(
            f0_mean=185.0,
            f0_std=30.0,
            f0_contour=[180.0, 185.0, 190.0],
            energy_rms=0.04,
            jitter_local=0.01,
            shimmer_local=0.03,
            hnr=18.0,
            spectral_centroid=2300.0,
            egemaps_vector=[0.1] * 88,
            arousal=0.5,
            valence=0.5,
            dominance=0.5,
        )
        assert f.f0_mean == 185.0
        assert len(f.egemaps_vector) == 88

    def test_to_payload(self):
        f = ProsodicFeatures(
            f0_mean=185.0,
            f0_std=30.0,
            f0_contour=[180.0],
            energy_rms=0.04,
            jitter_local=0.01,
            shimmer_local=0.03,
            hnr=18.0,
            spectral_centroid=2300.0,
            egemaps_vector=[0.1] * 88,
            arousal=0.6,
            valence=0.7,
            dominance=0.4,
        )
        p = f.to_payload()
        assert p["f0_mean"] == 185.0
        assert p["arousal"] == 0.6
        assert len(p["egemaps_vector"]) == 88


class TestPraatExtractor:
    def test_extract_from_sine_wave(self):
        signal = _make_sine_wave(freq=200.0, duration=1.0)
        extractor = PraatExtractor(sample_rate=16000)
        result = extractor.extract(signal)
        # A 200Hz sine wave should give F0 near 200
        assert 150.0 < result["f0_mean"] < 250.0
        assert result["f0_std"] >= 0.0
        assert len(result["f0_contour"]) > 0
        assert result["energy_rms"] > 0.0
        assert isinstance(result["jitter_local"], float)
        assert isinstance(result["shimmer_local"], float)
        assert isinstance(result["hnr"], float)

    def test_extract_from_silence(self):
        signal = np.zeros(16000, dtype=np.int16)
        extractor = PraatExtractor(sample_rate=16000)
        result = extractor.extract(signal)
        # Silence: F0 should be 0 or NaN-clamped to 0
        assert result["f0_mean"] == 0.0


class TestEGeMAPSExtractor:
    def test_extract_returns_88_features(self):
        signal = _make_sine_wave(freq=200.0, duration=1.0)
        extractor = EGeMAPSExtractor(sample_rate=16000)
        vector = extractor.extract(signal)
        assert len(vector) == 88

    def test_extract_from_silence(self):
        signal = np.zeros(16000, dtype=np.int16)
        extractor = EGeMAPSExtractor(sample_rate=16000)
        vector = extractor.extract(signal)
        assert len(vector) == 88


class TestDeriveDimensionalEmotion:
    def test_high_energy_high_f0_gives_high_arousal(self):
        a, v, d = derive_dimensional_emotion(
            f0_mean=300.0, f0_std=60.0, energy_rms=0.12,
            hnr=20.0, spectral_centroid=3000.0,
        )
        assert a > 0.6

    def test_silence_gives_low_arousal(self):
        a, v, d = derive_dimensional_emotion(
            f0_mean=0.0, f0_std=0.0, energy_rms=0.0,
            hnr=0.0, spectral_centroid=0.0,
        )
        assert a == 0.0

    def test_all_values_in_zero_one_range(self):
        a, v, d = derive_dimensional_emotion(
            f0_mean=400.0, f0_std=100.0, energy_rms=0.3,
            hnr=35.0, spectral_centroid=5000.0,
        )
        assert 0.0 <= a <= 1.0
        assert 0.0 <= v <= 1.0
        assert 0.0 <= d <= 1.0
