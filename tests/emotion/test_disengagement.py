"""Tests for DisengagementDetector."""

from forgestream.emotion.disengagement import DisengagementDetector


class TestDisengagementDetector:
    def test_no_disengagement_on_normal_features(self):
        det = DisengagementDetector(damping_factor=0.3)
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.09, "f0_std": 28.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is False
        assert det.is_disengaged("sp1") is False
        assert det.disengaged_speakers() == []

    def test_detects_energy_drop_with_pitch_flattening(self):
        det = DisengagementDetector(damping_factor=0.3)
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is True
        assert det.is_disengaged("sp1") is False
        assert "sp0" in det.disengaged_speakers()

    def test_both_quiet_is_not_disengagement(self):
        det = DisengagementDetector(damping_factor=0.3)
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
        assert det.is_disengaged("sp0") is False
        assert det.is_disengaged("sp1") is False

    def test_recovery_clears_flag(self):
        det = DisengagementDetector(damping_factor=0.3)
        for i in range(15):
            det.update("sp0", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.03, "f0_std": 8.0, "arousal": 0.2})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is True
        for i in range(12):
            det.update("sp0", {"energy_rms": 0.08, "f0_std": 25.0, "arousal": 0.4})
            det.update("sp1", {"energy_rms": 0.10, "f0_std": 30.0, "arousal": 0.5})
        assert det.is_disengaged("sp0") is False

    def test_get_damping_for_pair(self):
        det = DisengagementDetector(damping_factor=0.3)
        assert det.get_pair_damping("sp0", "sp1") == 1.0
        det._flags["sp0"] = True
        assert det.get_pair_damping("sp0", "sp1") == 0.3
        assert det.get_pair_damping("sp1", "sp2") == 1.0
