"""REST API endpoints for the dashboard."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)


def create_router(firestore_db: Any = None) -> APIRouter:
    """Create API router, optionally connected to Firestore."""
    router = APIRouter()
    db = firestore_db

    def _get_events() -> list[dict]:
        """Fetch events from Firestore."""
        if db is None:
            return []
        try:
            docs = db.collection("events").order_by("timestamp").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning("Firestore query failed: %s", e)
            return []

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "firestore": db is not None}

    @router.get("/graph")
    async def get_graph() -> dict:
        """Return the knowledge graph for visualization."""
        events = _get_events()
        concepts: dict[str, dict] = {}
        edges: list[dict] = []

        for event in events:
            if event.get("event_type") != "claim":
                continue
            payload = event.get("payload", {})
            keywords = payload.get("topic_keywords", [])
            confidence = payload.get("confidence", 0.5)

            for kw in keywords:
                if kw not in concepts:
                    concepts[kw] = {"name": kw, "confidence": confidence, "count": 1}
                else:
                    concepts[kw]["confidence"] = max(concepts[kw]["confidence"], confidence)
                    concepts[kw]["count"] += 1

            for i, kw_a in enumerate(keywords):
                for kw_b in keywords[i + 1:]:
                    edges.append({"source": kw_a, "target": kw_b, "weight": confidence})

        requirements = [
            {"description": e.get("payload", {}).get("description", ""), "status": "detected"}
            for e in events if e.get("event_type") == "requirement"
        ]
        artifacts = [
            {"compiles": e.get("payload", {}).get("compiles", False),
             "tests_pass": e.get("payload", {}).get("tests_pass", False)}
            for e in events if e.get("event_type") == "artifact"
        ]

        return {
            "concepts": list(concepts.values()),
            "requirements": requirements,
            "artifacts": artifacts,
            "edges": edges,
        }

    @router.get("/evaluator")
    async def get_evaluator() -> dict:
        """Return the evaluator trajectory."""
        events = _get_events()
        trajectory = [
            {"evaluator": e.get("evaluator", 0.0), "event_type": e.get("event_type", "")}
            for e in events
        ]
        current_e = trajectory[-1]["evaluator"] if trajectory else 0.0

        return {
            "trajectory": trajectory,
            "current": {
                "E_micro": current_e,
                "E_meso": current_e,
                "E_macro": current_e,
            },
            "axioms": {
                "monotone": True,
                "bounded_step": True,
                "constraint": True,
            },
        }

    @router.get("/branches")
    async def get_branches() -> dict:
        """Return branch tree data from BRANCH_POINT events."""
        events = _get_events()
        branches = []
        for e in events:
            if e.get("event_type") != "branch_point":
                continue
            p = e.get("payload", {})
            branches.append({
                "id": p.get("new_branch_id", ""),
                "new_branch_id": p.get("new_branch_id", ""),
                "parent_branch_id": p.get("parent_branch_id", None),
                "potential_score": p.get("potential_score", 0.0),
                "description": p.get("description", ""),
                "timestamp": e.get("timestamp", ""),
            })
        return {"branches": branches}

    @router.get("/seeds")
    async def get_seeds() -> dict:
        """Return SEED events with computed status."""
        from datetime import datetime, timezone
        events = _get_events()

        # Collect REQUIREMENT keywords for "promoted" detection
        req_keywords: set[str] = set()
        for e in events:
            if e.get("event_type") == "requirement":
                desc = e.get("payload", {}).get("description", "").lower()
                req_keywords.update(desc.split())

        # Collect meeting timestamps to establish "meetings old" approximation.
        # Use index position as a proxy: events in earliest third = old.
        total = len(events)
        seeds_raw = [e for e in events if e.get("event_type") == "seed"]

        seeds = []
        for i, e in enumerate(seeds_raw):
            p = e.get("payload", {})
            cluster_nodes: list[str] = p.get("cluster_nodes", [])

            # Determine status
            keywords_lower = {kw.lower() for kw in cluster_nodes}
            if keywords_lower & req_keywords:
                status = "promoted"
            elif total > 0 and events.index(e) < total // 3:
                status = "dormant"
            else:
                status = "active"

            seeds.append({
                "cluster_nodes": cluster_nodes,
                "novelty_score": p.get("novelty_score", 0.0),
                "avg_confidence": p.get("avg_confidence", 0.0),
                "domain_guess": p.get("domain_guess", ""),
                "description": p.get("description", ""),
                "status": status,
                "timestamp": e.get("timestamp", ""),
            })
        return {"seeds": seeds}

    @router.get("/trust-region")
    async def get_trust_region() -> dict:
        """Return SOS trust region state: epsilon, consecutive improvements, axiom status."""
        from forgestream.config import ForgeStreamConfig
        cfg = ForgeStreamConfig()
        events = _get_events()

        # Compute consecutive improvements from evaluator snapshots
        evaluator_values = [
            e.get("evaluator", 0.0)
            for e in events
            if e.get("event_type") in ("claim", "evaluator_snapshot")
        ]
        consecutive = 0
        for i in range(1, len(evaluator_values)):
            if evaluator_values[i] > evaluator_values[i - 1]:
                consecutive += 1
            else:
                consecutive = 0

        # Epsilon decays as improvements accumulate
        base_eps = cfg.trust_region_epsilon_base
        epsilon = max(0.05, base_eps * (0.95 ** consecutive))

        # Axiom status — computed from recent evaluator trajectory
        if len(evaluator_values) >= 2:
            diffs = [evaluator_values[i] - evaluator_values[i - 1]
                     for i in range(1, len(evaluator_values))]
            monotone = all(d >= -0.05 for d in diffs[-10:])  # allow small dips
            bounded_step = all(abs(d) <= 0.3 for d in diffs[-10:])
            constraint_ok = all(0.0 <= v <= 1.0 for v in evaluator_values[-10:])
        else:
            monotone = True
            bounded_step = True
            constraint_ok = True

        return {
            "epsilon": round(epsilon, 4),
            "consecutive_improvements": consecutive,
            "axiom_status": {
                "monotone": monotone,
                "bounded_step": bounded_step,
                "constraint": constraint_ok,
            },
        }

    @router.get("/autonomy-progression")
    async def get_autonomy_progression() -> dict:
        """Return epsilon history and predicted meeting when autonomy threshold is reached."""
        import json
        from pathlib import Path

        AUTO_SPAWN_THRESHOLD = 0.6
        BRANCH_AUTO_ALLOCATE = 0.7

        # Load history file (data/trust_region_history.json relative to CWD)
        history_path = Path("data") / "trust_region_history.json"
        if history_path.exists():
            try:
                history: list[dict] = json.loads(history_path.read_text())
                if not isinstance(history, list):
                    history = []
                # Sort by meeting number ascending
                history = sorted(history, key=lambda e: e.get("meeting", 0))
            except (json.JSONDecodeError, OSError):
                history = []
        else:
            history = []

        current_epsilon = history[-1]["epsilon"] if history else 0.525

        # Linear extrapolation: fit slope over last N points (or all if fewer)
        predicted_auto_spawn: int | None = None
        predicted_branch_auto: int | None = None
        slope: float = 0.0

        if len(history) >= 2:
            # Use at most last 5 points for slope
            window = history[-5:]
            n = len(window)
            xs = [e["meeting"] for e in window]
            ys = [e["epsilon"] for e in window]
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            numerator = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
            denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))
            if denominator > 0:
                slope = numerator / denominator
                # Predict meeting number when ε crosses each threshold
                last_meeting = xs[-1]
                last_epsilon = ys[-1]
                if slope > 0:
                    if last_epsilon < AUTO_SPAWN_THRESHOLD:
                        meetings_needed = (AUTO_SPAWN_THRESHOLD - last_epsilon) / slope
                        predicted_auto_spawn = int(last_meeting + meetings_needed) + 1
                    if last_epsilon < BRANCH_AUTO_ALLOCATE:
                        meetings_needed = (BRANCH_AUTO_ALLOCATE - last_epsilon) / slope
                        predicted_branch_auto = int(last_meeting + meetings_needed) + 1

        return {
            "history": history,
            "current_epsilon": round(current_epsilon, 6),
            "auto_spawn_threshold": AUTO_SPAWN_THRESHOLD,
            "branch_auto_allocate_threshold": BRANCH_AUTO_ALLOCATE,
            "slope_per_meeting": round(slope, 6),
            "predicted_auto_spawn_meeting": predicted_auto_spawn,
            "predicted_branch_auto_meeting": predicted_branch_auto,
        }

    @router.get("/contradictions")
    async def get_contradictions() -> dict:
        """Return CONTRADICTION events for the contradictions panel."""
        events = _get_events()
        contradictions = []
        for e in events:
            if e.get("event_type") != "contradiction":
                continue
            p = e.get("payload", {})
            contradictions.append({
                "concept_a": p.get("concept_a", ""),
                "concept_b": p.get("concept_b", ""),
                "explanation": p.get("explanation", ""),
                "probing_questions": p.get("probing_questions", []),
                "timestamp": e.get("timestamp", ""),
            })
        return {"contradictions": contradictions}

    @router.get("/agents")
    async def get_agents() -> dict:
        return {"agents": []}

    @router.get("/emotion/timeline")
    async def get_emotion_timeline() -> dict:
        """Return prosodic feature timeline for emotion visualization."""
        events = _get_events()
        timeline = []
        for e in events:
            if e.get("event_type") != "prosodic_feature":
                continue
            p = e.get("payload", {})
            timeline.append({
                "timestamp_ms": p.get("timestamp_ms", 0),
                "speaker_id": p.get("speaker_id", "unknown"),
                "arousal": p.get("arousal", 0.5),
                "valence": p.get("valence", 0.5),
                "dominance": p.get("dominance", 0.5),
                "emotion_tag": p.get("emotion_tag"),
            })
        return {"timeline": timeline}

    @router.get("/emotion/entrainment")
    async def get_emotion_entrainment() -> dict:
        """Return entrainment snapshots for group dynamics visualization."""
        events = _get_events()
        snapshots = [
            e.get("payload", {})
            for e in events
            if e.get("event_type") == "entrainment_snapshot"
        ]
        return {"snapshots": snapshots}

    @router.get("/emotion/speakers")
    async def get_emotion_speakers() -> dict:
        """Return per-speaker emotion summary."""
        events = _get_events()
        speaker_data: dict[str, dict] = {}
        for e in events:
            if e.get("event_type") != "prosodic_feature":
                continue
            p = e.get("payload", {})
            sid = p.get("speaker_id", "unknown")
            if sid not in speaker_data:
                speaker_data[sid] = {
                    "speaker_id": sid,
                    "arousal_sum": 0.0,
                    "valence_sum": 0.0,
                    "count": 0,
                }
            speaker_data[sid]["arousal_sum"] += p.get("arousal", 0.5)
            speaker_data[sid]["valence_sum"] += p.get("valence", 0.5)
            speaker_data[sid]["count"] += 1

        speakers = []
        for sd in speaker_data.values():
            c = sd["count"]
            speakers.append({
                "speaker_id": sd["speaker_id"],
                "mean_arousal": sd["arousal_sum"] / c if c > 0 else 0.5,
                "mean_valence": sd["valence_sum"] / c if c > 0 else 0.5,
                "count": c,
            })
        return {"speakers": speakers}

    @router.get("/emotion/rapport")
    async def get_emotion_rapport() -> dict:
        """Return rapport scores for visualization."""
        events = _get_events()
        scores = [
            e.get("payload", {})
            for e in events
            if e.get("event_type") == "rapport_score"
        ]
        latest_trend = scores[-1].get("group_trend", 0.0) if scores else 0.0
        return {"scores": scores, "latest_trend": latest_trend}

    @router.get("/suggestions")
    async def get_suggestions() -> dict:
        events = _get_events()
        suggestions = [
            {"text": e.get("payload", {}).get("text", ""),
             "priority": e.get("payload", {}).get("priority", 0.5)}
            for e in events if e.get("event_type") == "suggestion"
        ]
        return {"suggestions": suggestions}

    @router.get("/proof-obligations")
    async def get_proof_obligations() -> dict:
        """Return PROOF_OBLIGATION events for the proof queue panel."""
        events = _get_events()
        obligations = []
        for e in events:
            if e.get("event_type") != "proof_obligation":
                continue
            p = e.get("payload", {})
            obligations.append({
                "claim_id": p.get("claim_id", ""),
                "claim_text": p.get("claim_text", ""),
                "speaker": p.get("speaker", "unknown"),
                "confidence": p.get("confidence", 0.0),
                "lean4_stub": p.get("lean4_stub", ""),
                "conclusion": p.get("conclusion", ""),
                "status": p.get("status", "pending"),
                "formalization_confidence": p.get("formalization_confidence", 0.6),
                "timestamp": e.get("timestamp", ""),
            })
        return {"obligations": obligations}

    return router
