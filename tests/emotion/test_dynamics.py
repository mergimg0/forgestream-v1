"""Tests for GroupDynamicsEngine — TLCC, CRQA, participation parity, entropy."""

import math
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from forgestream.emotion.dynamics import GroupDynamicsEngine
from forgestream.events.schema import Event, EventType


class TestTLCC:
    def test_identical_signals_have_correlation_one_lag_zero(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 100))]
        corr, lag = engine.compute_tlcc(signal, signal)
        assert corr == pytest.approx(1.0, abs=0.01)
        assert lag == 0

    def test_shifted_signal_has_nonzero_lag(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        base = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 100))]
        # Shift by 5 samples: b is a delayed copy of a
        shifted = [0.0] * 5 + base[:-5]
        corr, lag = engine.compute_tlcc(base, shifted, max_lag=10)
        assert corr > 0.7
        assert lag != 0  # should detect a lag

    def test_uncorrelated_signals_have_low_correlation(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        np.random.seed(42)
        a = [float(x) for x in np.random.randn(100)]
        b = [float(x) for x in np.random.randn(100)]
        corr, _ = engine.compute_tlcc(a, b)
        assert abs(corr) < 0.5

    def test_short_signals_return_zero(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        corr, lag = engine.compute_tlcc([1.0], [1.0])
        assert corr == 0.0
        assert lag == 0


class TestCRQA:
    def test_identical_signals_have_higher_recurrence_than_random(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        signal = [float(x) for x in np.sin(np.linspace(0, 4 * np.pi, 50))]
        rr_identical = engine.compute_crqa_recurrence_rate(signal, signal, radius=0.3)
        np.random.seed(42)
        random_b = [float(x) for x in np.random.randn(50)]
        rr_random = engine.compute_crqa_recurrence_rate(signal, random_b, radius=0.3)
        assert rr_identical > rr_random

    def test_random_signals_have_low_recurrence(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        np.random.seed(42)
        a = [float(x) for x in np.random.randn(50)]
        b = [float(x) for x in np.random.randn(50)]
        rr = engine.compute_crqa_recurrence_rate(a, b)
        assert rr < 0.5

    def test_empty_signals_return_zero(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        assert engine.compute_crqa_recurrence_rate([], []) == 0.0


class TestParticipationParity:
    def test_equal_durations_returns_one(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        parity = engine.compute_participation_parity(
            {"a": 10.0, "b": 10.0, "c": 10.0}
        )
        assert parity == pytest.approx(1.0, abs=0.01)

    def test_unequal_durations_returns_lower_parity(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        equal = engine.compute_participation_parity({"a": 10.0, "b": 10.0})
        unequal = engine.compute_participation_parity({"a": 100.0, "b": 1.0})
        assert unequal < equal

    def test_empty_returns_one(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        assert engine.compute_participation_parity({}) == 1.0


class TestTurnTakingEntropy:
    def test_uniform_turns_have_higher_entropy_than_single(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        # Perfectly alternating: a, b, a, b, ...
        alternating = ["a", "b"] * 20
        entropy_alt = engine.compute_turn_taking_entropy(alternating)
        # Three speakers alternating should have even higher entropy
        three_way = ["a", "b", "c"] * 15
        entropy_three = engine.compute_turn_taking_entropy(three_way)
        assert entropy_alt > 0.0
        assert entropy_three > entropy_alt

    def test_single_speaker_has_zero_entropy(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        seq = ["a"] * 20
        entropy = engine.compute_turn_taking_entropy(seq)
        assert entropy == 0.0

    def test_empty_returns_zero(self):
        engine = GroupDynamicsEngine.__new__(GroupDynamicsEngine)
        assert engine.compute_turn_taking_entropy([]) == 0.0


class TestGroupDynamicsEngine:
    @pytest.mark.asyncio
    async def test_emits_snapshot_after_interval(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = GroupDynamicsEngine(orchestrator=orch)

        sid = orch.session_id
        bid = uuid4()

        # Feed prosodic features for two speakers across 61 seconds
        for i in range(62):
            event = Event(
                event_type=EventType.PROSODIC_FEATURE,
                session_id=sid,
                branch_id=bid,
                author="emotion_extractor",
                evaluator=0.0,
                payload={
                    "speaker_id": "sp0" if i % 2 == 0 else "sp1",
                    "timestamp_ms": i * 1000,
                    "f0_mean": 200.0 + i,
                    "energy_rms": 0.1,
                    "arousal": 0.5,
                    "f0_std": 20.0,
                },
            )
            await engine.on_event(event)

        # Should have emitted at least one ENTRAINMENT_SNAPSHOT
        snapshot_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.ENTRAINMENT_SNAPSHOT
        ]
        assert len(snapshot_calls) >= 1
        payload = snapshot_calls[0][0][0].payload
        assert "speaker_pairs" in payload
        assert "group_metrics" in payload
        assert "participation_parity" in payload["group_metrics"]
