"""Disengagement detection from prosodic feature trends.

Monitors per-speaker energy and F0 variability over a sliding window.
Flags disengagement when energy drops + pitch flattens + the change
is one-sided (other speakers maintain normal levels).
"""

from __future__ import annotations

from collections import deque


ENERGY_THRESHOLD = 0.6
F0_STD_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.7
WINDOW_SIZE = 10
BASELINE_WINDOW = 15
EMA_ALPHA = 0.01


class DisengagementDetector:
    """Detects per-speaker disengagement from prosodic trends.

    Parameters:
        damping_factor: Multiplier applied to rapport composite when disengaged.
    """

    def __init__(self, damping_factor: float = 0.3) -> None:
        self._damping_factor = damping_factor
        self._windows: dict[str, deque[dict]] = {}
        self._baselines: dict[str, dict[str, float]] = {}
        self._update_counts: dict[str, int] = {}
        self._flags: dict[str, bool] = {}

    def update(self, speaker_id: str, features: dict) -> None:
        """Add a prosodic feature snapshot for a speaker."""
        if speaker_id not in self._windows:
            self._windows[speaker_id] = deque(maxlen=WINDOW_SIZE)
            self._baselines[speaker_id] = {"energy_rms": 0.0, "f0_std": 0.0}
            self._update_counts[speaker_id] = 0
            self._flags[speaker_id] = False

        self._windows[speaker_id].append(features)
        self._update_counts[speaker_id] += 1
        count = self._update_counts[speaker_id]

        energy = features.get("energy_rms", 0.0)
        f0_std = features.get("f0_std", 0.0)

        if count <= BASELINE_WINDOW:
            bl = self._baselines[speaker_id]
            bl["energy_rms"] = (bl["energy_rms"] * (count - 1) + energy) / count
            bl["f0_std"] = (bl["f0_std"] * (count - 1) + f0_std) / count
        else:
            bl = self._baselines[speaker_id]
            bl["energy_rms"] = (1 - EMA_ALPHA) * bl["energy_rms"] + EMA_ALPHA * energy
            bl["f0_std"] = (1 - EMA_ALPHA) * bl["f0_std"] + EMA_ALPHA * f0_std

        if count >= BASELINE_WINDOW + WINDOW_SIZE:
            self._check_disengagement(speaker_id)

    def _check_disengagement(self, speaker_id: str) -> None:
        """Check if a speaker is disengaged based on current window vs baseline."""
        bl = self._baselines[speaker_id]
        if bl["energy_rms"] < 1e-6 or bl["f0_std"] < 1e-6:
            return

        window = list(self._windows[speaker_id])
        mean_energy = sum(f.get("energy_rms", 0.0) for f in window) / len(window)
        mean_f0_std = sum(f.get("f0_std", 0.0) for f in window) / len(window)

        energy_ratio = mean_energy / bl["energy_rms"]
        f0_ratio = mean_f0_std / bl["f0_std"]

        if self._flags[speaker_id]:
            if energy_ratio >= RECOVERY_THRESHOLD and f0_ratio >= RECOVERY_THRESHOLD:
                self._flags[speaker_id] = False
        else:
            energy_low = energy_ratio < ENERGY_THRESHOLD
            pitch_flat = f0_ratio < F0_STD_THRESHOLD
            one_sided = self._is_one_sided(speaker_id)

            if energy_low and pitch_flat and one_sided:
                self._flags[speaker_id] = True

    def _is_one_sided(self, speaker_id: str) -> bool:
        """Check if other speakers maintain normal levels."""
        for other_id, bl in self._baselines.items():
            if other_id == speaker_id:
                continue
            if bl["energy_rms"] < 1e-6:
                continue
            window = list(self._windows.get(other_id, deque()))
            if not window:
                continue
            other_energy = sum(f.get("energy_rms", 0.0) for f in window) / len(window)
            if other_energy / bl["energy_rms"] >= ENERGY_THRESHOLD:
                return True
        return False

    def is_disengaged(self, speaker_id: str) -> bool:
        """Check if a speaker is currently flagged as disengaged."""
        return self._flags.get(speaker_id, False)

    def disengaged_speakers(self) -> list[str]:
        """Return list of currently disengaged speaker IDs."""
        return [sid for sid, flagged in self._flags.items() if flagged]

    def get_pair_damping(self, speaker_a: str, speaker_b: str) -> float:
        """Get the damping factor for a speaker pair."""
        if self._flags.get(speaker_a, False) or self._flags.get(speaker_b, False):
            return self._damping_factor
        return 1.0
