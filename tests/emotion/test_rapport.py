"""Tests for RapportEngine — multi-component rapport scoring."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from forgestream.emotion.rapport import RapportEngine, interpolate_weights
from forgestream.events.schema import Event, EventType


def _make_prosodic(session_id, branch_id, speaker, ts, arousal=0.5, valence=0.5):
    return Event(
        event_type=EventType.PROSODIC_FEATURE,
        session_id=session_id, branch_id=branch_id,
        author="emotion_extractor", evaluator=0.0,
        payload={
            "speaker_id": speaker, "timestamp_ms": ts,
            "arousal": arousal, "valence": valence,
            "f0_mean": 200.0, "f0_std": 30.0, "energy_rms": 0.1,
        },
    )


def _make_snapshot(session_id, branch_id, ts):
    return Event(
        event_type=EventType.ENTRAINMENT_SNAPSHOT,
        session_id=session_id, branch_id=branch_id,
        author="dynamics_engine", evaluator=0.0,
        payload={
            "timestamp_ms": ts,
            "speaker_pairs": [
                {"speaker_a": "sp0", "speaker_b": "sp1",
                 "f0_correlation": 0.6, "energy_correlation": 0.5},
            ],
            "group_metrics": {"participation_parity": 0.8},
        },
    )


class TestInterpolateWeights:
    def test_meeting_1_favors_attentiveness_positivity(self):
        w = interpolate_weights(1)
        assert w["attentiveness"] > w["coordination"]
        assert w["positivity"] > w["coordination"]

    def test_meeting_5_favors_coordination(self):
        w = interpolate_weights(5)
        assert w["coordination"] > w["attentiveness"]
        assert w["coordination"] > w["positivity"]

    def test_weights_sum_to_one(self):
        for mc in [1, 2, 3, 5, 10]:
            w = interpolate_weights(mc)
            assert sum(w.values()) == pytest.approx(1.0, abs=0.001)

    def test_meeting_3_is_midpoint(self):
        w = interpolate_weights(3)
        assert 0.20 < w["attentiveness"] < 0.35
        assert 0.15 < w["coordination"] < 0.40


class TestRapportEngine:
    @pytest.mark.asyncio
    async def test_emits_rapport_score_on_snapshot(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch, meeting_count=3)
        sid = orch.session_id
        bid = uuid4()

        for i in range(30):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000, arousal=0.6, valence=0.5)
            )
            await engine.on_event(
                _make_prosodic(sid, bid, "sp1", i * 1000 + 500, arousal=0.5, valence=0.6)
            )

        await engine.on_event(_make_snapshot(sid, bid, 30000))

        rapport_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.RAPPORT_SCORE
        ]
        assert len(rapport_calls) >= 1

        payload = rapport_calls[0][0][0].payload
        assert "group_composite" in payload
        assert "pair_scores" in payload
        assert "group_trend" in payload
        assert "weights_applied" in payload
        assert 0.0 <= payload["group_composite"] <= 1.0

    @pytest.mark.asyncio
    async def test_ignores_self_authored(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch)
        self_event = Event(
            event_type=EventType.RAPPORT_SCORE,
            session_id=orch.session_id, branch_id=uuid4(),
            author="rapport_engine", evaluator=0.0,
            payload={"test": True},
        )
        await engine.on_event(self_event)
        assert orch.process_event.call_count == 0

    @pytest.mark.asyncio
    async def test_single_speaker_no_rapport(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch)
        sid = orch.session_id
        bid = uuid4()

        # Only one speaker
        for i in range(15):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000)
            )

        await engine.on_event(_make_snapshot(sid, bid, 15000))

        rapport_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.RAPPORT_SCORE
        ]
        assert len(rapport_calls) == 0  # need >= 2 speakers

    @pytest.mark.asyncio
    async def test_pair_scores_have_all_components(self):
        orch = MagicMock()
        orch.session_id = uuid4()
        orch.process_event = AsyncMock(return_value=True)

        engine = RapportEngine(orchestrator=orch, meeting_count=3)
        sid = orch.session_id
        bid = uuid4()

        for i in range(30):
            await engine.on_event(
                _make_prosodic(sid, bid, "sp0", i * 1000, arousal=0.7, valence=0.6)
            )
            await engine.on_event(
                _make_prosodic(sid, bid, "sp1", i * 1000 + 500, arousal=0.6, valence=0.5)
            )

        await engine.on_event(_make_snapshot(sid, bid, 30000))

        rapport_calls = [
            c for c in orch.process_event.call_args_list
            if c[0][0].event_type == EventType.RAPPORT_SCORE
        ]
        pair = rapport_calls[0][0][0].payload["pair_scores"][0]
        for field in ["attentiveness", "positivity", "coordination",
                       "symmetry", "composite", "disengagement_damped",
                       "surrogate_validated"]:
            assert field in pair, f"Missing field: {field}"
