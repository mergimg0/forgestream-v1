# ForgeStream Rapport Tracking System — Design Spec

**Date:** 2026-03-29
**Status:** Approved (brainstorming complete)
**Depends on:** Emotion Pipeline (Phases 0-7, implemented), Speaker Diarization (planned)

---

## Overview

A multi-component rapport tracking system that measures interpersonal dynamics in meetings from prosodic features, producing a composite Rapport Score enriched by surrogate-validated CRQA via RunPod GPU. Integrates with the SOS governor evaluator, axiom checker, and trust region.

## Design Decisions (From Brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Exposure model | Composite + drill-down (C) | Matches existing E(π) pattern. Single number for TUI/evaluator, components for dashboard/reports |
| Disengagement handling | Hard override with damping (B) | Prevents GRPO from learning to ignore disengagement. Damping factor (default 0.3) is GRPO-tunable |
| Meeting history weights | Sigmoid interpolation + GRPO fallback (B+C) | Encodes Tickle-Degnen as Bayesian prior; GRPO overrides if wrong for this user |
| Surrogate validation | RunPod per-window with local fallback | Gold standard CRQA validation at ~$0.50/meeting. Circuit breaker for seamless recovery |

## Psychological Foundation

Based on Tickle-Degnen & Rosenthal (1990) 3-component model of rapport:

| Component | Definition | Acoustic Proxy | Weight (Early) | Weight (Established) |
|-----------|-----------|----------------|----------------|---------------------|
| Mutual Attentiveness | Focused interest | Arousal correlation between speakers | 0.35 | 0.20 |
| Positivity | Warmth, affective tone | Valence proximity between speakers | 0.30 | 0.15 |
| Coordination | Behavioral synchrony | CRQA %DET + TT (surrogate-validated) | 0.15 | 0.40 |
| Symmetry | Equitable interaction | 1 - |TE(A→B) - TE(B→A)| | 0.20 | 0.25 |

Weights interpolate via sigmoid based on `meeting_count`. Meeting 1 ≈ 90% early profile. Meeting 5 ≈ 90% established profile. GRPO operates on top of the interpolated weights.

## Architecture

### Event Flow

```
PROSODIC_FEATURE (every ~1s, from EmotionExtractor)
    │
    ├──► GroupDynamicsEngine (existing)
    │       └── ENTRAINMENT_SNAPSHOT (every 60s)
    │               │
    │               ▼
    │           RapportEngine (NEW)
    │               ├── Component 1: Attentiveness (local, Pearson on arousal)
    │               ├── Component 2: Positivity (local, valence distance)
    │               ├── Component 3: Coordination (RunPod CRQA + surrogates)
    │               ├── Component 4: Symmetry (local, transfer entropy)
    │               ├── Disengagement damping
    │               ├── Meeting-count sigmoid weights
    │               └── RAPPORT_SCORE (every 60s)
    │
    └──► DisengagementDetector (inside RapportEngine)
            └── Per-speaker sliding window (last 10 events)
            └── Flags when energy↓ + F0 flattening + one-sided
```

### RAPPORT_SCORE Event Payload

```python
{
    "timestamp_ms": 120000,
    "window_duration_ms": 30000,
    "pair_scores": [
        {
            "speaker_a": "sp0",
            "speaker_b": "sp1",
            "attentiveness": 0.72,
            "positivity": 0.65,
            "coordination": 0.58,
            "symmetry": 0.81,
            "composite": 0.68,
            "disengagement_damped": false,
            "surrogate_validated": true,
        }
    ],
    "group_composite": 0.68,
    "group_trend": 0.03,
    "disengaged_speakers": [],
    "weights_applied": {
        "attentiveness": 0.28,
        "positivity": 0.22,
        "coordination": 0.30,
        "symmetry": 0.20,
    },
    "meeting_count": 5,
}
```

## Component Algorithms

### Component 1: Attentiveness

Pearson correlation of arousal time series between speakers over the 30-second window. High correlation = speakers are emotionally "tracking" each other.

```
attentiveness = max(0, pearson_r(arousal_a, arousal_b))
```

Clamped to [0, 1]. Negative correlation (anti-phase arousal) is treated as 0 attentiveness, not negative rapport.

### Component 2: Positivity

Inverted valence distance. Speakers with similar valence (both positive or both neutral) score high. One positive + one negative scores low.

```
mean_valence_a = mean(valence_a_series)
mean_valence_b = mean(valence_b_series)
positivity = 1.0 - |mean_valence_a - mean_valence_b|
```

### Component 3: Coordination (RunPod CRQA)

CRQA %DET (determinism) is the primary coordination metric, validated by surrogate testing. %TT (trapping time) is secondary.

**Pipeline:**
1. Z-score normalize log-F0 per speaker
2. Estimate embedding params (dim, delay, radius) at session start via AMI + FNN on first 60s
3. Send F0 pairs + params to RunPod endpoint
4. RunPod computes CRQA on real data + 20 shuffled surrogates
5. Return metrics + significance flags
6. If `significant.det == true`: coordination = normalized %DET (range [0, 1])
7. If not significant: coordination = 0.0 (the synchrony is noise)

**Fallback chain:**
- RunPod available → surrogate-validated CRQA (gold standard)
- RunPod down → local CPU CRQA, compared against session noise floor
- CRQA fails entirely → use TLCC peak correlation from ENTRAINMENT_SNAPSHOT

### Component 4: Symmetry

Transfer entropy asymmetry. Mutual influence (both speakers affect each other) = high symmetry. One-directional influence = low symmetry.

```
te_a_to_b = transfer_entropy(f0_a, f0_b, lag=1)
te_b_to_a = transfer_entropy(f0_b, f0_a, lag=1)
asymmetry = |te_a_to_b - te_b_to_a| / max(te_a_to_b + te_b_to_a, epsilon)
symmetry = 1.0 - asymmetry
```

## Disengagement Detector

### Detection Criteria (All Three Must Co-occur)

1. **Energy declining:** RMS energy trend over last 10 PROSODIC_FEATURE events has negative slope AND current energy < 60% of speaker's session baseline
2. **Pitch flattening:** F0 std in current window < 40% of speaker's session mean F0 std
3. **One-sided:** At least one other speaker maintains normal energy/pitch levels

### What It Does NOT Flag

- Both speakers quieting together (natural ebb)
- Attentive silence (high arousal, low energy)
- Naturally quiet speakers (per-speaker baseline normalization)

### Behavior When Flagged

- Damping factor (default 0.3, GRPO-tunable) multiplies composite rapport for all pairs involving flagged speaker
- Speaker added to `disengaged_speakers` list in RAPPORT_SCORE
- Flag persists until speaker's energy AND F0 variability return to 70% of baseline (hysteresis)

### Per-Speaker Baseline

- Computed from first 60 seconds of meeting
- Updated with slow EMA (alpha=0.01) to adapt to genuine shifts without reacting to momentary dips

## Meeting-Count Weight Interpolation

Sigmoid interpolation between early and established weight profiles:

```python
def interpolate_weights(meeting_count: int) -> dict[str, float]:
    # Sigmoid: meeting 1 → ~0.1, meeting 3 → ~0.5, meeting 5 → ~0.9
    t = 1.0 / (1.0 + math.exp(-(meeting_count - 3)))

    early = {"attentiveness": 0.35, "positivity": 0.30, "coordination": 0.15, "symmetry": 0.20}
    established = {"attentiveness": 0.20, "positivity": 0.15, "coordination": 0.40, "symmetry": 0.25}

    return {k: early[k] * (1 - t) + established[k] * t for k in early}
```

GRPO operates on top: after interpolation, GRPO perturbations are applied and the best-performing variant is blended in (70/30 conservative update, same as existing weight tuning).

## RunPod CRQA Endpoint

### Deployment

Stateless FastAPI app deployed on RunPod serverless (A4000 GPU, $0.44/hr).

### API Contract

```
POST /crqa/validate
Request:
{
    "f0_a": [float, ...],
    "f0_b": [float, ...],
    "params": {
        "embedding_dim": 3,
        "time_delay": 2,
        "radius": 0.25,
        "n_surrogates": 20
    }
}

Response:
{
    "real": {"det": 0.58, "tt": 4.2, "entr": 1.8, "lam": 0.45, "rr": 0.035,
             "dcrp": [...]},
    "surrogate_stats": {"det_mean": 0.12, "det_std": 0.04, "tt_mean": 1.1, "tt_std": 0.3},
    "significant": {"det": true, "tt": true},
    "compute_ms": 1200
}
```

### Circuit Breaker

| Failure Count | Behavior |
|---------------|----------|
| 0 | Try RunPod every cycle |
| 1-2 | Still try (transient) |
| 3+ | Skip for 5 cycles (5 min), then retry |
| Success after skip | Reset to 0, full resumption |

Every 60-second cycle always tries RunPod first (unless circuit is open). The actual CRQA request serves as the health check — no separate ping endpoint needed. Transition back is seamless and automatic.

### Warm-Up

During meeting initialization, send a dummy CRQA request with small synthetic signals. The 10-30s cold start overlaps with Gemini connection setup and first 30s of audio accumulation.

## SOS Governor Integration

### Evaluator

The existing `_emotional_engagement` metric is enriched:

```
emotional_engagement = (
    0.4 × group_rapport_composite     # from RAPPORT_SCORE (when available)
    + 0.3 × mean_arousal              # from PROSODIC_FEATURE (existing)
    + 0.3 × f0_variability            # from PROSODIC_FEATURE (existing)
)
```

Falls back to current formula (arousal + F0 var + energy) when no RAPPORT_SCORE events exist.

### Axiom Checker

Axiom 1 (Monotone) gains an advisory: if `group_rapport_composite` has 3 consecutive declining windows AND `disengaged_speakers` is non-empty, emit a "rapport degradation" warning in EVALUATOR_SNAPSHOT. This is advisory only — NOT an axiom violation (productive disagreement naturally dips rapport).

Axiom 2 (Bounded Step): No change.
Axiom 3 (Constraint): No change.

### Trust Region

`record_meeting_result` accepts optional `rapport_trend`. Meetings with building rapport (trend > 0.1) AND improving E(π) get a fractional trust boost (+0.5 consecutive improvements). The 0.5 boost value is GRPO-tunable.

## File Structure

### New Files

| File | Responsibility | Est. Lines |
|------|---------------|-----------|
| `forgestream/emotion/rapport.py` | RapportEngine — orchestrates components, weights, damping, emits RAPPORT_SCORE | ~250 |
| `forgestream/emotion/disengagement.py` | DisengagementDetector — per-speaker sliding window, baseline, co-occurrence | ~120 |
| `forgestream/emotion/crqa_router.py` | CRQAComputeRouter — circuit breaker, RunPod client, local fallback | ~130 |
| `forgestream/emotion/transfer_entropy.py` | compute_transfer_entropy() — scipy-based for symmetry component | ~60 |
| `runpod/crqa_endpoint.py` | Standalone FastAPI for RunPod — CRQA + surrogates | ~80 |
| `tests/emotion/test_rapport.py` | RapportEngine tests | ~150 |
| `tests/emotion/test_disengagement.py` | Disengagement detector tests | ~100 |
| `tests/emotion/test_crqa_router.py` | Router + circuit breaker tests | ~80 |

### Modified Files

| File | Changes |
|------|---------|
| `forgestream/events/schema.py` | Add `RAPPORT_SCORE` to EventType |
| `forgestream/governor/evaluator.py` | Enrich `_emotional_engagement` with rapport composite |
| `forgestream/governor/axioms.py` | Add rapport degradation advisory |
| `forgestream/governor/trust_region.py` | Accept `rapport_trend` in `record_meeting_result` |
| `forgestream/orchestrator.py` | Add `attach_rapport_engine()` |
| `forgestream/live_stream.py` | Wire RapportEngine in init |
| `forgestream/config.py` | Add RunPod + rapport config fields |
| `forgestream/dashboard/api.py` | Add `/emotion/rapport` endpoint |
| `pyproject.toml` | Add `scipy` to emotion deps |

## Edge Cases

**Single speaker (monologue):** No pair scores computed. `group_composite` defaults to 0.5 (neutral). Disengagement detector still runs (monitors the sole speaker's energy/pitch for signs of fatigue or disengagement).

**Speaker joins late:** Their baseline is computed from their first 60 seconds of speech, not the meeting's first 60 seconds. Pair scores involving them begin when both speakers have at least 30 seconds of data.

**More than 6 speakers:** Pairwise computation is O(n²). With 6 speakers = 15 pairs. Each pair's CRQA goes to RunPod in a batch request. At 7+ speakers, batch the pairs into groups of 10 to stay within RunPod timeout.

## Research Sources

- Kruyt et al. (2023), JSLHR — 12 entrainment methods compared
- Tickle-Degnen & Rosenthal (1990) — 3-component rapport model
- Fusaroli & Tylén (2016), Cognitive Science — CRQA DET/LAM predict task performance
- Coco & Dale (2014/2021) — crqa R package, CRQA metrics definitions
- Levitan & Hirschberg (2011/2012) — Entrainment taxonomy + social variable validation
- Benus, Gravano, Levitan et al. (2014) — Power-deference convergence (Supreme Court)
- Healey, Purver & Howes (2014) — Syntactic divergence as engagement signal
- Pérez & Gálvez (2016) — Unsigned synchrony (disentrainment is also coordination)
- Gravano et al. (2015) — Backward mimicry vs forward influence
- Marwan — Surrogate validation for CRQA (recurrence-plot.tk)
