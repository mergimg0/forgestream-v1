"""Tests for SpeakerTimeSeries — per-speaker prosodic time series accumulator."""

from forgestream.emotion.speaker import SpeakerTimeSeries


class TestSpeakerTimeSeries:
    def test_add_and_retrieve_f0(self):
        ts = SpeakerTimeSeries()
        ts.add_feature("sp0", 1000, {"f0_mean": 200.0, "energy_rms": 0.1})
        ts.add_feature("sp0", 2000, {"f0_mean": 210.0, "energy_rms": 0.12})
        ts.add_feature("sp1", 1500, {"f0_mean": 150.0, "energy_rms": 0.08})

        f0_a, f0_b = ts.get_f0_series("sp0", "sp1")
        assert len(f0_a) == 2
        assert len(f0_b) == 1
        assert f0_a == [200.0, 210.0]
        assert f0_b == [150.0]

    def test_speaker_ids(self):
        ts = SpeakerTimeSeries()
        ts.add_feature("alice", 100, {"f0_mean": 180.0})
        ts.add_feature("bob", 200, {"f0_mean": 120.0})
        assert set(ts.speaker_ids()) == {"alice", "bob"}

    def test_speaking_durations(self):
        ts = SpeakerTimeSeries(stride_seconds=1.0)
        ts.add_feature("sp0", 1000, {"f0_mean": 200.0})
        ts.add_feature("sp0", 2000, {"f0_mean": 200.0})
        ts.add_feature("sp0", 3000, {"f0_mean": 200.0})
        ts.add_feature("sp1", 1500, {"f0_mean": 150.0})
        durations = ts.speaking_durations()
        assert durations["sp0"] == 3.0
        assert durations["sp1"] == 1.0

    def test_energy_series(self):
        ts = SpeakerTimeSeries()
        ts.add_feature("sp0", 1000, {"f0_mean": 200.0, "energy_rms": 0.05})
        ts.add_feature("sp0", 2000, {"f0_mean": 210.0, "energy_rms": 0.10})
        energies = ts.get_energy_series("sp0")
        assert energies == [0.05, 0.10]

    def test_eviction_after_max_duration(self):
        ts = SpeakerTimeSeries(max_duration_seconds=2.0, stride_seconds=1.0)
        # Add 4 entries spanning 4 seconds — first 2 should be evicted
        for i in range(4):
            ts.add_feature("sp0", i * 1000, {"f0_mean": float(i)})
        f0, _ = ts.get_f0_series("sp0", "sp0")
        # max_duration=2s, stride=1s → keep last 2 entries
        assert len(f0) == 2
        assert f0 == [2.0, 3.0]

    def test_empty_speaker_returns_empty_lists(self):
        ts = SpeakerTimeSeries()
        f0_a, f0_b = ts.get_f0_series("sp0", "sp1")
        assert f0_a == []
        assert f0_b == []
        assert ts.get_energy_series("sp0") == []
