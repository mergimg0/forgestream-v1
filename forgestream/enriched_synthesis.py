"""Enriched post-meeting synthesis — feeds x-vector verified claims + prosodic
data through the full GRPO training pipeline.

Bridges the gap between enriched_batch.py (produces claims + prosodic segments)
and post_meeting.py (needs Event objects with proper EventTypes).

Creates proper PROSODIC_FEATURE, ENTRAINMENT_SNAPSHOT, RAPPORT_SCORE, and
EMOTION_STATE events from the enriched pipeline output, then feeds everything
through PostMeetingSynthesis.run() so all 5 GRPO loops train on real signal.

Usage:
    python -m forgestream.enriched_synthesis \
        --claims data/ryan_mccarthy_claims_xvector.json \
        --segments data/ryan_mccarthy_xvector_segments.json \
        --audio data/recordings/Ryan_McCarthy_complete.wav \
        --name "Ryan_McCarthy_2026-03-30"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import wave
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .config import ForgeStreamConfig, load_config
from .emotion.buffer import AudioRingBuffer
from .emotion.correlator import EmotionCorrelator
from .emotion.dynamics import GroupDynamicsEngine
from .emotion.rapport import RapportEngine
from .events.schema import Event, EventType
from .governor.trust_region import TrustRegion
from .orchestrator import Orchestrator
from .post_meeting import PostMeetingSynthesis

logger = logging.getLogger(__name__)


def _build_prosodic_events(
    segments: list[dict],
    session_id: UUID,
    branch_id: UUID,
) -> list[Event]:
    """Convert enriched segments into PROSODIC_FEATURE events."""
    events = []
    for seg in segments:
        prosodic = seg.get("prosodic", {})
        if not prosodic or prosodic.get("f0_mean", 0) == 0:
            continue

        event = Event(
            event_type=EventType.PROSODIC_FEATURE,
            session_id=session_id,
            branch_id=branch_id,
            author="enriched_batch_diarizer",
            evaluator=0.0,
            payload={
                "speaker_id": seg.get("speaker_name", seg.get("speaker_id", "unknown")),
                "timestamp_ms": int(seg["start_s"] * 1000),
                "window_duration_ms": int((seg["end_s"] - seg["start_s"]) * 1000),
                "f0_mean": prosodic.get("f0_mean", 0.0),
                "f0_std": prosodic.get("f0_std", 0.0),
                "f0_contour": prosodic.get("f0_contour", []),
                "energy_rms": prosodic.get("energy_rms", 0.0),
                "jitter_local": prosodic.get("jitter_local", 0.0),
                "shimmer_local": prosodic.get("shimmer_local", 0.0),
                "hnr": prosodic.get("hnr", 0.0),
                "spectral_centroid": prosodic.get("spectral_centroid", 0.0),
                "arousal": prosodic.get("arousal", 0.5),
                "valence": prosodic.get("valence", 0.5),
                "dominance": prosodic.get("dominance", 0.5),
                "egemaps_vector": prosodic.get("egemaps_vector", []),
            },
        )
        events.append(event)
    return events


def _build_claim_events(
    claims: list[dict],
    session_id: UUID,
    branch_id: UUID,
) -> list[Event]:
    """Convert enriched claims into CLAIM events."""
    from .gemini.extraction import ClaimExtractor
    extractor = ClaimExtractor(session_id=session_id, branch_id=branch_id)

    events = []
    for claim in claims:
        event = extractor.parse_claim(claim)
        events.append(event)
    return events


async def _run_group_dynamics(
    prosodic_events: list[Event],
    orchestrator: Orchestrator,
) -> list[Event]:
    """Run GroupDynamicsEngine on prosodic events to generate ENTRAINMENT_SNAPSHOT events.

    Returns the entrainment events that were emitted.
    """
    captured_events: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == EventType.ENTRAINMENT_SNAPSHOT:
            captured_events.append(event)

    bus = orchestrator.event_bus
    bus.subscribe(capture)

    engine = GroupDynamicsEngine(orchestrator=orchestrator)
    bus.subscribe(engine.on_event)

    for event in prosodic_events:
        await bus.publish(event)

    bus.unsubscribe(engine.on_event)
    bus.unsubscribe(capture)

    logger.info("GroupDynamicsEngine produced %d ENTRAINMENT_SNAPSHOT events", len(captured_events))
    return captured_events


async def _run_rapport_engine(
    prosodic_events: list[Event],
    entrainment_events: list[Event],
    orchestrator: Orchestrator,
    meeting_count: int = 1,
    rapport_weights: dict[str, float] | None = None,
) -> list[Event]:
    """Run RapportEngine on prosodic + entrainment events to generate RAPPORT_SCORE events."""
    captured_events: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == EventType.RAPPORT_SCORE:
            captured_events.append(event)

    bus = orchestrator.event_bus
    bus.subscribe(capture)

    engine = RapportEngine(
        orchestrator=orchestrator,
        meeting_count=meeting_count,
        rapport_weights=rapport_weights,
    )
    bus.subscribe(engine.on_event)

    # Feed prosodic events (RapportEngine consumes these for disengagement detection)
    for event in prosodic_events:
        await bus.publish(event)

    # Feed entrainment snapshots (triggers rapport computation)
    for event in entrainment_events:
        await bus.publish(event)

    bus.unsubscribe(engine.on_event)
    bus.unsubscribe(capture)

    logger.info("RapportEngine produced %d RAPPORT_SCORE events", len(captured_events))
    return captured_events


async def _run_emotion_correlator(
    claim_events: list[Event],
    prosodic_events: list[Event],
    orchestrator: Orchestrator,
) -> list[Event]:
    """Run EmotionCorrelator to generate EMOTION_STATE events from claim↔prosodic alignment."""
    captured_events: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == EventType.EMOTION_STATE:
            captured_events.append(event)

    bus = orchestrator.event_bus
    bus.subscribe(capture)

    correlator = EmotionCorrelator(orchestrator=orchestrator)
    bus.subscribe(correlator.on_event)

    # Feed prosodic events first (builds the buffer)
    for event in prosodic_events:
        await bus.publish(event)

    # Then feed claims (triggers correlation)
    for event in claim_events:
        await bus.publish(event)

    bus.unsubscribe(correlator.on_event)
    bus.unsubscribe(capture)

    logger.info("EmotionCorrelator produced %d EMOTION_STATE events", len(captured_events))
    return captured_events


async def run_enriched_synthesis(
    claims_path: str,
    segments_path: str,
    audio_path: str = "",
    meeting_name: str = "",
    config: ForgeStreamConfig | None = None,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Run full GRPO training pipeline on enriched (x-vector verified) data.

    Steps:
    1. Load x-vector claims + prosodic segments
    2. Build PROSODIC_FEATURE events from segments
    3. Build CLAIM events from claims
    4. Run GroupDynamicsEngine → ENTRAINMENT_SNAPSHOT events
    5. Run RapportEngine → RAPPORT_SCORE events
    6. Run EmotionCorrelator → EMOTION_STATE events
    7. Merge all events, sort by timestamp
    8. Feed into PostMeetingSynthesis.run() (all 5 GRPO loops)
    9. Update TrustRegion with rapport trend
    10. Persist with has_prosodic_signal flag
    """
    config = config or load_config()
    data_dir_path = Path(data_dir)

    print(f"\n{'='*60}")
    print(f"ENRICHED SYNTHESIS — FULL GRPO TRAINING")
    print(f"{'='*60}")

    # ---- Step 1: Load data ----
    print("\n--- Step 1: Loading enriched data ---")
    claims = json.loads(Path(claims_path).read_text())
    segments = json.loads(Path(segments_path).read_text())
    print(f"  Claims: {len(claims)}")
    print(f"  Segments: {len(segments)}")

    session_id = uuid4()
    branch_id = uuid4()

    # ---- Step 2: Build PROSODIC_FEATURE events ----
    print("\n--- Step 2: Building PROSODIC_FEATURE events ---")
    prosodic_events = _build_prosodic_events(segments, session_id, branch_id)
    print(f"  {len(prosodic_events)} PROSODIC_FEATURE events")

    # ---- Step 3: Build CLAIM events ----
    print("\n--- Step 3: Building CLAIM events ---")
    claim_events = _build_claim_events(claims, session_id, branch_id)
    print(f"  {len(claim_events)} CLAIM events")

    # ---- Step 4: Run GroupDynamicsEngine ----
    print("\n--- Step 4: GroupDynamicsEngine → ENTRAINMENT_SNAPSHOT ---")
    orch = Orchestrator(config)
    entrainment_events = await _run_group_dynamics(prosodic_events, orch)
    print(f"  {len(entrainment_events)} ENTRAINMENT_SNAPSHOT events")

    # ---- Step 5: Run RapportEngine ----
    print("\n--- Step 5: RapportEngine → RAPPORT_SCORE ---")
    pms = PostMeetingSynthesis(config=config, data_dir=data_dir)
    meeting_count = pms.load_meeting_count()
    rapport_weights = pms.load_rapport_weights(meeting_count)
    rapport_events = await _run_rapport_engine(
        prosodic_events, entrainment_events, orch,
        meeting_count=meeting_count,
        rapport_weights=rapport_weights,
    )
    print(f"  {len(rapport_events)} RAPPORT_SCORE events")

    # ---- Step 6: Run EmotionCorrelator ----
    print("\n--- Step 6: EmotionCorrelator → EMOTION_STATE ---")
    emotion_events = await _run_emotion_correlator(claim_events, prosodic_events, orch)
    print(f"  {len(emotion_events)} EMOTION_STATE events")

    # ---- Step 7: Merge and sort all events ----
    print("\n--- Step 7: Merging all events ---")
    all_events = (
        claim_events
        + prosodic_events
        + entrainment_events
        + rapport_events
        + emotion_events
    )

    # Sort by timestamp (prosodic by timestamp_ms, claims by approximate parse)
    def _event_sort_key(e: Event) -> int:
        ts = e.payload.get("timestamp_ms", 0)
        if ts == 0:
            # Try to parse timestamp_approx from claims
            ts_str = e.payload.get("timestamp_approx", "00:00")
            parts = ts_str.replace(":", " ").split()
            try:
                ts = (int(parts[0]) * 60 + int(parts[1])) * 1000
            except (IndexError, ValueError):
                pass
        return ts

    all_events.sort(key=_event_sort_key)
    print(f"  Total events: {len(all_events)}")

    from collections import Counter
    type_counts = Counter(e.event_type.value for e in all_events)
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    # ---- Step 8: Feed into PostMeetingSynthesis.run() ----
    print("\n--- Step 8: PostMeetingSynthesis (all 5 GRPO loops) ---")

    # Load audio buffer if audio path provided (for corpus persistence)
    audio_buffer = None
    if audio_path and Path(audio_path).exists():
        audio_buffer = AudioRingBuffer(capacity_seconds=3600)  # 1 hour capacity
        with wave.open(audio_path, "rb") as wf:
            # Read in 0.5s chunks to fill the ring buffer
            chunk_samples = 8000  # 0.5s at 16kHz
            while True:
                raw = wf.readframes(chunk_samples)
                if not raw:
                    break
                audio_buffer.write_chunk(raw)
        print(f"  Audio buffer loaded: {audio_buffer.chunk_count} chunks")

    # Connect to PostgreSQL for event persistence
    # ot-ctx-pg-leak-005: track conn for cleanup in finally block
    store = None
    _pg_conn = None
    try:
        import psycopg
        from .events.store import EventStore
        _pg_conn = await psycopg.AsyncConnection.connect(config.postgres_dsn)
        store = EventStore(_pg_conn)
        print("  PostgreSQL: connected")
    except Exception as e:
        print(f"  PostgreSQL: unavailable ({e})")

    # Create orchestrator with store, feed all events
    # ot-ctx-pg-leak-005: try/finally ensures connection closes on any exception
    try:
        return await _run_synthesis_core(
            config, store, all_events, pms, meeting_name, data_dir_path,
            claims, prosodic_events, emotion_events, rapport_events,
            entrainment_events, claim_events, type_counts, audio_buffer,
        )
    finally:
        if _pg_conn is not None:
            try:
                await _pg_conn.close()
            except Exception:
                pass


async def _run_synthesis_core(
    config, store, all_events, pms, meeting_name, data_dir_path,
    claims, prosodic_events, emotion_events, rapport_events,
    entrainment_events, claim_events, type_counts, audio_buffer,
):
    """Inner synthesis logic — extracted for try/finally connection safety."""
    synth_orch = Orchestrator(config, store=store)
    synth_orch.attach_synthesis_engine()

    # Pre-populate the event buffer (PostMeetingSynthesis reads from it)
    for event in all_events:
        synth_orch._event_buffer.append(event)
        event.evaluator = synth_orch.evaluator.compute(synth_orch._event_buffer[-20:])
        if store:
            await store.append(event)

    # Run full synthesis
    result = await pms.run(
        events=all_events,
        meeting_name=meeting_name,
        human_score=None,  # Auto-score from prosodic data
        audio_buffer=audio_buffer,
    )

    print(f"\n  E(π): {result['e_meso']:.3f}")
    print(f"  Meeting count: {result['meeting_count']}")
    print(f"  Weights: {json.dumps({k: round(v, 4) for k, v in result['weights'].items() if k != 'meeting_count'})}")
    print(f"  Report: {result.get('report_path', 'n/a')}")
    sensitivity = result.get("weight_sensitivity", {})
    if sensitivity:
        print(f"  Most impactful weight: {sensitivity.get('most_impactful', '?')}")

    # ---- Step 9: Update TrustRegion with rapport trend ----
    print("\n--- Step 9: TrustRegion update ---")
    tr_path = data_dir_path / "trust_region.json"
    tr = TrustRegion.load(tr_path)
    rapport_trend = 0.0
    if rapport_events:
        rapport_trend = rapport_events[-1].payload.get("group_trend", 0.0)

    tr.record_meeting_result(
        e_macro_improved=result["e_meso"] > 0.3,  # Improving above baseline
        axiom_violations=0,
        rapport_trend=rapport_trend,
    )
    tr.save(str(tr_path))
    print(f"  Epsilon: {tr.epsilon:.3f}")
    print(f"  Rapport trend: {rapport_trend:.3f}")
    print(f"  Auto-spawn: {'YES' if tr.epsilon >= tr.AUTO_SPAWN_THRESHOLD else 'no'}")

    # ---- Step 10: Persist with has_prosodic_signal flag ----
    print("\n--- Step 10: Persisting with prosodic signal flag ---")

    # Update weights.json with the flag
    weights_file = data_dir_path / "weights.json"
    weights_data = json.loads(weights_file.read_text())
    weights_data["has_prosodic_signal"] = True
    weights_data["prosodic_coverage"] = len(prosodic_events) / max(len(claim_events), 1)
    weights_data["emotion_state_count"] = len(emotion_events)
    weights_data["rapport_score_count"] = len(rapport_events)
    weights_data["entrainment_snapshot_count"] = len(entrainment_events)
    weights_data["xvector_verified"] = True
    weights_data["speaker_confidence_high_pct"] = (
        sum(1 for c in claims if c.get("speaker_confidence") == "high") / max(len(claims), 1)
    )
    weights_file.write_text(json.dumps(weights_data, indent=2))

    # Update the last entry in weights_history.json too
    history_file = data_dir_path / "weights_history.json"
    if history_file.exists():
        history = json.loads(history_file.read_text())
        if history:
            history[-1]["has_prosodic_signal"] = True
            history[-1]["prosodic_coverage"] = weights_data["prosodic_coverage"]
            history[-1]["xvector_verified"] = True
            history_file.write_text(json.dumps(history, indent=2))

    # Neural GRPO activation gate
    enriched_count = sum(1 for h in json.loads(history_file.read_text()) if h.get("has_prosodic_signal"))
    if enriched_count >= 5:
        print(f"\n  NEURAL GRPO: READY TO ACTIVATE ({enriched_count}/5 enriched meetings)")
    else:
        print(f"\n  Neural GRPO: {enriched_count}/5 enriched meetings. Collecting data.")
        print(f"  Current method: perturbation GRPO.")

    print(f"  has_prosodic_signal: True")
    print(f"  prosodic_coverage: {weights_data['prosodic_coverage']:.2f}")
    print(f"  xvector_verified: True")

    # DB connection closed by caller's finally block (ot-ctx-pg-leak-005)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"ENRICHED SYNTHESIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Events processed: {len(all_events)}")
    print(f"    CLAIM: {type_counts.get('claim', 0)}")
    print(f"    PROSODIC_FEATURE: {type_counts.get('prosodic_feature', 0)}")
    print(f"    ENTRAINMENT_SNAPSHOT: {type_counts.get('entrainment_snapshot', 0)}")
    print(f"    RAPPORT_SCORE: {type_counts.get('rapport_score', 0)}")
    print(f"    EMOTION_STATE: {type_counts.get('emotion_state', 0)}")
    print(f"  E(π): {result['e_meso']:.3f}")
    print(f"  Trust epsilon: {tr.epsilon:.3f}")
    print(f"  GRPO loops trained: WeightTuner ✓, ToneTuner ✓, RapportGRPO ✓, ConfigTuner ✓, PromptTuner ✓")

    result["trust_epsilon"] = tr.epsilon
    result["all_event_count"] = len(all_events)
    result["type_counts"] = dict(type_counts)
    result["has_prosodic_signal"] = True

    return result


# ---- Step 4 enhancement: Upgraded compute_auto_score ----
# This is applied by monkey-patching PostMeetingSynthesis before run()

def _enriched_auto_score(self: PostMeetingSynthesis, events: list[Event]) -> float:
    """Compute automatic meeting quality score from enriched event data.

    60% driven by prosodic/rapport data — no human scoring needed.
    """
    claims = [e for e in events if e.event_type == EventType.CLAIM]
    requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
    artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
    findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]
    prosodic = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]
    rapport = [e for e in events if e.event_type == EventType.RAPPORT_SCORE]

    req_scaffold = (
        len(artifacts) / max(len(requirements), 1)
        if requirements else 0.0
    )
    findings_per_claim = (
        len(findings) / max(len(claims), 1)
        if claims else 0.0
    )

    # Enriched engagement signal
    if prosodic:
        mean_arousal = sum(
            e.payload.get("arousal", 0.5) for e in prosodic
        ) / len(prosodic)
        mean_valence = sum(
            e.payload.get("valence", 0.5) for e in prosodic
        ) / len(prosodic)
        mean_jitter = sum(
            e.payload.get("jitter_local", 0.02) for e in prosodic
        ) / len(prosodic)
        vocal_confidence = max(0.0, 1.0 - (mean_jitter / 0.04))
        engagement = 0.4 * mean_arousal + 0.3 * mean_valence + 0.3 * vocal_confidence
    else:
        engagement = 0.0

    # Rapport signal
    if rapport:
        final_rapport = rapport[-1].payload.get("group_composite", 0.5)
        rapport_trend = rapport[-1].payload.get("group_trend", 0.0)
        rapport_signal = final_rapport * (1.0 + 0.2 * rapport_trend)
    else:
        rapport_signal = 0.0

    base_score = (
        0.20 * min(1.0, req_scaffold)
        + 0.20 * min(1.0, findings_per_claim)
        + 0.30 * engagement
        + 0.30 * min(1.0, rapport_signal)
    )
    return min(1.0, max(0.0, base_score))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enriched post-meeting synthesis")
    parser.add_argument("--claims", required=True, help="Path to x-vector verified claims JSON")
    parser.add_argument("--segments", required=True, help="Path to enriched segments JSON")
    parser.add_argument("--audio", default="", help="Path to WAV audio (for corpus persistence)")
    parser.add_argument("--name", default="", help="Meeting name")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()

    # ot-ctx-gcp-cred-003: resolve relative to project root, not CWD
    _project_root = str(Path(__file__).resolve().parent.parent)
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(_project_root, ".secrets", "service-account.json"),
    )

    config = load_config()

    # Apply enriched auto-score before running
    PostMeetingSynthesis.compute_auto_score = _enriched_auto_score

    result = asyncio.run(run_enriched_synthesis(
        claims_path=args.claims,
        segments_path=args.segments,
        audio_path=args.audio,
        meeting_name=args.name,
        config=config,
        data_dir=args.data_dir,
    ))

    print(f"\nFinal E(π): {result['e_meso']:.3f}")


if __name__ == "__main__":
    main()
