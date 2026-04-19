"""Prosodic feature extraction using openSMILE eGeMAPS and Parselmouth.

Two extractors:
- PraatExtractor: F0, jitter, shimmer, HNR, energy, spectral centroid
- EGeMAPSExtractor: Full 88-dimensional eGeMAPS feature vector

Both accept numpy int16 arrays at 16kHz and return plain Python types
(no pandas/numpy in the output — safe for JSON serialization in event payloads).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProsodicFeatures:
    """Complete prosodic feature set for one analysis window."""

    # Pitch
    f0_mean: float
    f0_std: float
    f0_contour: list[float]

    # Energy
    energy_rms: float

    # Voice quality
    jitter_local: float
    shimmer_local: float
    hnr: float
    spectral_centroid: float

    # Full eGeMAPS vector
    egemaps_vector: list[float] = field(default_factory=list)

    # Dimensional emotion (derived)
    arousal: float = 0.5
    valence: float = 0.5
    dominance: float = 0.5

    # Categorical emotion (SenseVoice — Phase 1.5)
    emotion_tag: str | None = None
    emotion_confidence: float | None = None

    def to_payload(self) -> dict:
        """Convert to ECEF event payload dict."""
        return {
            "f0_mean": self.f0_mean,
            "f0_std": self.f0_std,
            "f0_contour": self.f0_contour,
            "energy_rms": self.energy_rms,
            "jitter_local": self.jitter_local,
            "shimmer_local": self.shimmer_local,
            "hnr": self.hnr,
            "spectral_centroid": self.spectral_centroid,
            "egemaps_vector": self.egemaps_vector,
            "arousal": self.arousal,
            "valence": self.valence,
            "dominance": self.dominance,
            "emotion_tag": self.emotion_tag,
            "emotion_confidence": self.emotion_confidence,
        }


class PraatExtractor:
    """Extract F0, jitter, shimmer, HNR, energy, spectral centroid via Parselmouth."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate

    def extract(self, signal: "numpy.ndarray") -> dict:
        """Extract voice quality features from an int16 numpy array.

        Returns a dict with: f0_mean, f0_std, f0_contour, energy_rms,
        jitter_local, shimmer_local, hnr, spectral_centroid.
        """
        import numpy as np
        import parselmouth
        from parselmouth import praat

        # Convert int16 to float64 in [-1, 1] range for Parselmouth
        signal_float = signal.astype(np.float64) / 32768.0
        snd = parselmouth.Sound(signal_float, sampling_frequency=self._sample_rate)

        # Pitch extraction
        pitch = snd.to_pitch(time_step=0.01)  # 10ms frames
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        if len(voiced) == 0:
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_contour": [],
                "energy_rms": float(np.sqrt(np.mean(signal_float**2))),
                "jitter_local": 0.0,
                "shimmer_local": 0.0,
                "hnr": 0.0,
                "spectral_centroid": 0.0,
            }

        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_contour = [float(v) for v in f0_values]

        # Energy (RMS)
        energy_rms = float(np.sqrt(np.mean(signal_float**2)))

        # Jitter and Shimmer via PointProcess
        point_process = praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 600
        )
        jitter_local = praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = praat.call(
            [snd, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        # Clamp NaN to 0
        if math.isnan(jitter_local):
            jitter_local = 0.0
        if math.isnan(shimmer_local):
            shimmer_local = 0.0

        # Harmonics-to-Noise Ratio
        harmonicity = snd.to_harmonicity()
        hnr = praat.call(harmonicity, "Get mean", 0, 0)
        if math.isnan(hnr):
            hnr = 0.0

        # Spectral centroid
        spectrum = snd.to_spectrum()
        spectral_centroid = praat.call(
            spectrum, "Get centre of gravity...", 1
        )
        if math.isnan(spectral_centroid):
            spectral_centroid = 0.0

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_contour": f0_contour,
            "energy_rms": energy_rms,
            "jitter_local": float(jitter_local),
            "shimmer_local": float(shimmer_local),
            "hnr": float(hnr),
            "spectral_centroid": float(spectral_centroid),
        }


class EGeMAPSExtractor:
    """Extract the 88-dimensional eGeMAPS feature vector via openSMILE."""

    def __init__(self, sample_rate: int = 16000) -> None:
        import opensmile

        self._sample_rate = sample_rate
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, signal: "numpy.ndarray") -> list[float]:
        """Extract eGeMAPS functionals from an int16 numpy array.

        Returns a list of 88 floats.
        """
        import numpy as np

        # openSMILE expects float32 in [-1, 1]
        signal_float = signal.astype(np.float32) / 32768.0
        result = self._smile.process_signal(signal_float, self._sample_rate)
        return [float(v) for v in result.values[0]]


def derive_dimensional_emotion(
    f0_mean: float,
    f0_std: float,
    energy_rms: float,
    hnr: float,
    spectral_centroid: float,
) -> tuple[float, float, float]:
    """Derive arousal, valence, dominance from prosodic features.

    Uses established psychoacoustic mappings:
    - Arousal: energy + F0 variability (high energy + high F0 range = high arousal)
    - Valence: spectral centroid + HNR (bright + clear = positive)
    - Dominance: energy + F0 mean (loud + low pitch = dominant)

    Returns (arousal, valence, dominance) each in [0, 1].
    """
    # Normalize features to approximate [0, 1] using typical speech ranges
    norm_f0_std = min(1.0, f0_std / 80.0)
    norm_energy = min(1.0, energy_rms / 0.15)
    norm_hnr = min(1.0, max(0.0, hnr / 30.0))
    norm_centroid = min(1.0, max(0.0, (spectral_centroid - 500) / 4500))
    norm_f0_mean = min(1.0, max(0.0, (f0_mean - 75) / 525))

    arousal = 0.6 * norm_energy + 0.4 * norm_f0_std
    valence = 0.5 * norm_centroid + 0.5 * norm_hnr
    dominance = 0.6 * norm_energy + 0.4 * (1.0 - norm_f0_mean)

    return (
        max(0.0, min(1.0, arousal)),
        max(0.0, min(1.0, valence)),
        max(0.0, min(1.0, dominance)),
    )
