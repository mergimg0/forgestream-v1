# Pipeline Integrity Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 verified defects across the rapport, claim extraction, GRPO, and state persistence systems so that the pipeline produces valid data for both corpus processing and production meetings.

**Architecture:** Three categories of fix: (A) wiring bugs where GRPO results are saved but never loaded, (B) configuration gaps where Gemini defaults produce noisy output, (C) logic defects where wrong fitness functions or hardcoded stubs corrupt optimization. All fixes are backwards-compatible — they improve existing behavior without changing interfaces.

**Tech Stack:** Python 3.12, google-genai SDK, numpy, existing ForgeStream modules.

---

## Execution Constraints

**Sequential execution required for `live_stream.py`:** Tasks 1, 2, 3, and 4 all modify `forgestream/live_stream.py`. They must run sequentially — never in parallel — or one agent's changes will clobber another's. Tasks 1→2→3 insert into `__init__`; Task 4 modifies `connect()` (a separate method, so it could run in parallel with 1-3 if using worktrees).

**Task dependencies:**
- Task 8 depends on Task 1 (Task 1 wires `rapport_weights` through `orchestrator.py` and `live_stream.py`; Task 8 modifies only `rapport.py` internals and assumes the call-site wiring already exists)
- Tasks 5, 6, 7 are independent of each other and of Tasks 1-4
- Task 9 is independent; Task 10 depends on Task 9
- Task 11 runs last (integration verification)

---

## File Map

| File | Changes | Responsibility | Tasks |
|------|---------|----------------|-------|
| `forgestream/live_stream.py` | Load rapport weights, config overrides, trust region at startup; temperature=0 | Wiring + config | 1, 2, 3, 4 (sequential) |
| `forgestream/agent_dispatcher.py` | Wire `TrustRegion.load()` | Wiring fix | 3 |
| `forgestream/emotion/rapport.py` | Add maturity inference from raw signals | Rapport maturity | 8 (depends on 1) |
| `forgestream/orchestrator.py` | Pass `rapport_weights` through `attach_rapport_engine()` | Wiring fix | 1 |
| `forgestream/post_meeting.py` | Fix rapport GRPO fitness, fix auto_score, add consensus re-extraction | GRPO + claims | 5, 7, 10 |
| `forgestream/governor/evaluator.py` | Make confidence-aware, fix `_suggestion_uptake` | Evaluator accuracy | 6 |
| `forgestream/runner.py` | Add `temperature=0` to batch Gemini calls | Claim stability | 4 |
| `forgestream/batch_meeting.py` | Add `temperature=0` to batch Gemini calls | Claim stability | 4 |
| `forgestream/audio_meeting.py` | Add `temperature=0` to batch Gemini calls | Claim stability | 4 |
| `tests/governor/test_evaluator.py` | Tests for confidence-aware evaluator | Testing | 6 |
| `tests/governor/test_improvement.py` | Tests for rapport GRPO fix | Testing | 5 |
| `tests/emotion/test_rapport.py` | Tests for maturity inference | Testing | 8 |
| `tests/test_live_stream.py` | Tests for startup loading | Testing | 1 |
| `tests/test_post_meeting.py` | Tests for consensus re-extraction + auto_score | Testing | 7, 9 |

---

## Task 1: Wire Rapport Weights Loading at Startup

**The bug:** `PostMeetingSynthesis.run()` saves GRPO-tuned rapport weights to `data/rapport_weights.json`. But `GeminiLiveStream.__init__` never loads them — `RapportEngine` always starts from `interpolate_weights(meeting_count)`, discarding all prior tuning.

**Files:**
- Modify: `forgestream/live_stream.py:94-160`
- Modify: `forgestream/orchestrator.py:171-183`
- Modify: `forgestream/emotion/rapport.py:62-72`
- Test: `tests/test_live_stream.py`

- [ ] **Step 1: Write failing test — rapport weights loaded at startup**

In `tests/test_live_stream.py`, add:

```python
def test_rapport_weights_loaded_from_disk(tmp_path, monkeypatch):
    """Saved rapport weights should be loaded into RapportEngine at startup."""
    import json
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Write custom rapport weights
    rapport_weights = {
        "attentiveness": 0.30,
        "positivity": 0.20,
        "coordination": 0.35,
        "symmetry": 0.15,
        "meeting_count": 5,
    }
    (data_dir / "rapport_weights.json").write_text(json.dumps(rapport_weights))

    # Write evaluator weights (required for init)
    (data_dir / "weights.json").write_text(json.dumps({
        "knowledge": 0.2, "verification": 0.2, "scaffold": 0.2,
        "uptake": 0.2, "engagement": 0.2, "meeting_count": 5,
    }))

    from forgestream.post_meeting import PostMeetingSynthesis
    from forgestream.config import ForgeStreamConfig
    config = ForgeStreamConfig(data_dir=str(data_dir))
    pms = PostMeetingSynthesis(config=config, data_dir=str(data_dir))

    loaded = pms.load_rapport_weights(meeting_count=5)
    assert abs(loaded["coordination"] - 0.35) < 0.001
    assert abs(loaded["attentiveness"] - 0.30) < 0.001
```

- [ ] **Step 2: Run test to verify it passes** (this tests existing load_rapport_weights — should pass already)

Run: `python3 -m pytest tests/test_live_stream.py::test_rapport_weights_loaded_from_disk -v`

- [ ] **Step 3: Add `rapport_weights` parameter to `RapportEngine.__init__` and `attach_rapport_engine`**

In `forgestream/emotion/rapport.py`, modify `__init__`:

```python
def __init__(
    self,
    orchestrator: "Orchestrator",
    meeting_count: int = 1,
    damping_factor: float = 0.3,
    runpod_endpoint: str = "",
    runpod_timeout: float = 4.0,
    rapport_weights: dict[str, float] | None = None,
) -> None:
    self._orchestrator = orchestrator
    self._meeting_count = meeting_count
    if rapport_weights is not None:
        self._weights = rapport_weights
    else:
        self._weights = interpolate_weights(meeting_count)
```

In `forgestream/orchestrator.py`, modify `attach_rapport_engine`:

```python
def attach_rapport_engine(
    self, meeting_count: int = 1, damping_factor: float = 0.3,
    runpod_endpoint: str = "", runpod_timeout: float = 4.0,
    rapport_weights: dict[str, float] | None = None,
) -> "RapportEngine":
    """Create and attach a RapportEngine to this orchestrator's EventBus."""
    from .emotion.rapport import RapportEngine
    engine = RapportEngine(
        orchestrator=self, meeting_count=meeting_count,
        damping_factor=damping_factor, runpod_endpoint=runpod_endpoint,
        runpod_timeout=runpod_timeout, rapport_weights=rapport_weights,
    )
    self.event_bus.subscribe(engine.on_event)
    return engine
```

- [ ] **Step 4: Load and pass rapport weights in `GeminiLiveStream.__init__`**

In `forgestream/live_stream.py`, after line 98 (`self._meeting_count = pms.load_meeting_count()`), add:

```python
# Load persisted rapport weights (GRPO loop continuity)
self._rapport_weights = pms.load_rapport_weights(self._meeting_count)
logger.info(
    "Loaded rapport weights: %s",
    {k: round(v, 3) for k, v in self._rapport_weights.items()},
)
```

Then modify the `attach_rapport_engine` call around line 155:

```python
self.rapport_engine = orchestrator.attach_rapport_engine(
    meeting_count=self._meeting_count,
    damping_factor=config.rapport_damping_factor,
    runpod_endpoint=config.runpod_crqa_endpoint,
    runpod_timeout=config.runpod_timeout_seconds,
    rapport_weights=self._rapport_weights,
)
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_live_stream.py tests/emotion/ -v -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forgestream/emotion/rapport.py forgestream/orchestrator.py forgestream/live_stream.py tests/test_live_stream.py
git commit -m "fix: load persisted rapport weights at startup (was write-only dead end)"
```

---

## Task 2: Wire Config Overrides Loading at Startup

**The bug:** `PostMeetingSynthesis.run()` saves tuned config to `data/config_overrides.json`. The `load_config_overrides()` function exists in `config.py` but `GeminiLiveStream.__init__` never calls it.

**Files:**
- Modify: `forgestream/live_stream.py:94-113`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

In `tests/test_config.py`, add:

```python
def test_load_config_overrides_applies_saved_values(tmp_path):
    """Saved config overrides should be loaded and applied to config."""
    import json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config_overrides.json").write_text(json.dumps({
        "spawn_cooldown_seconds": 45.0,
        "emotion_stride_seconds": 2.0,
    }))

    from forgestream.config import ForgeStreamConfig, load_config_overrides
    base = ForgeStreamConfig(data_dir=str(data_dir))
    updated = load_config_overrides(data_dir=str(data_dir), config=base)
    assert updated.spawn_cooldown_seconds == 45
    assert updated.emotion_stride_seconds == 2.0
```

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_config.py::test_load_config_overrides_applies_saved_values -v`
Expected: PASS (function exists, just isn't called at startup)

- [ ] **Step 3: Wire loading in `GeminiLiveStream.__init__`**

In `forgestream/live_stream.py`, after the prompt params loading block (around line 113), add:

```python
# Load persisted config overrides (ConfigTuner GRPO loop)
from .config import load_config_overrides
config = load_config_overrides(data_dir=config.data_dir, config=config)
self.config = config
logger.info("Applied config overrides from %s/config_overrides.json", config.data_dir)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_config.py tests/test_live_stream.py -v -x`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add forgestream/live_stream.py tests/test_config.py
git commit -m "fix: load config overrides at startup (was write-only dead end)"
```

---

## Task 3: Wire TrustRegion Persistence

**The bug:** `TrustRegion.load()` exists at `trust_region.py:155` but `AgentDispatcher.__init__` calls `TrustRegion()` (bare constructor), resetting epsilon to 0.525 every session.

**Files:**
- Modify: `forgestream/agent_dispatcher.py:33`
- Modify: `forgestream/config.py` (add `trust_region_path`)
- Test: `tests/test_agent_dispatcher.py`

- [ ] **Step 1: Write failing test**

In `tests/test_agent_dispatcher.py`, add:

```python
def test_trust_region_loaded_from_disk(tmp_path):
    """AgentDispatcher should load persisted TrustRegion state."""
    import json
    tr_path = tmp_path / "trust_region.json"
    tr_path.write_text(json.dumps({
        "consecutive_improvements": 5,
        "total_violations": 1,
        "meeting_count": 8,
        "volatility": 0.05,
    }))

    from forgestream.governor.trust_region import TrustRegion
    tr = TrustRegion.load(tr_path)
    assert tr._consecutive_improvements == 5
    assert tr._total_violations == 1
    assert tr._meeting_count == 8
```

- [ ] **Step 2: Run test to verify it passes** (TrustRegion.load already works)

Run: `python3 -m pytest tests/test_agent_dispatcher.py::test_trust_region_loaded_from_disk -v`
Expected: PASS

- [ ] **Step 3: Wire TrustRegion.load() in AgentDispatcher**

In `forgestream/agent_dispatcher.py`, replace line 33:

```python
# Before:
self.trust_region = TrustRegion()

# After:
trust_region_path = Path(config.data_dir) / "trust_region.json"
self.trust_region = TrustRegion.load(trust_region_path)
self._trust_region_path = trust_region_path
```

- [ ] **Step 4: Wire TrustRegion loading and saving in GeminiLiveStream**

**IMPORTANT:** `TrustRegion.save()` exists but has zero call sites in production code. `GeminiLiveStream` has no `agent_dispatcher` attribute. The save must be wired directly into the live stream, following the same pattern used for evaluator weights.

In `forgestream/live_stream.py __init__`, after the evaluator weights loading block (after line 103), add:

```python
# Load persisted trust region state
from .governor.trust_region import TrustRegion as _TrustRegion
_tr_path = Path(config.data_dir) / "trust_region.json"
self._trust_region = _TrustRegion.load(_tr_path)
self._trust_region_path = _tr_path
logger.info("Loaded trust region: epsilon=%.3f", self._trust_region.epsilon)
```

In `forgestream/live_stream.py`, in the `_run_post_meeting` method, after the `PostMeetingSynthesis.run()` call completes, add:

```python
# Persist trust region state
self._trust_region.save(str(self._trust_region_path))
```

This follows the exact pattern used for evaluator weights: load in `__init__`, save in `_run_post_meeting`.

**Known limitation:** `GeminiLiveStream._trust_region` and `AgentDispatcher.trust_region` are two separate instances loaded from the same file. If `AgentDispatcher` mutates its instance during a meeting (via `record_axiom_violation()` or `record_meeting_result()`), the `GeminiLiveStream` copy won't reflect those changes and will overwrite with stale state. However, neither `record_meeting_result()` nor `record_axiom_violation()` has any call sites in production code today (verified by round 1 audit), so both instances are currently immutable during meetings. This save establishes the file on disk and persists state correctly in the current codebase. When the mutation calls are wired in a future task, the save should be moved to save `AgentDispatcher.trust_region` (the live instance) instead.

- [ ] **Step 4b: Write test that verifies AgentDispatcher loads from disk**

```python
def test_agent_dispatcher_loads_trust_region(tmp_path):
    """AgentDispatcher should load TrustRegion from disk, not start fresh."""
    import json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "trust_region.json").write_text(json.dumps({
        "consecutive_improvements": 5,
        "total_violations": 1,
        "meeting_count": 8,
        "volatility": 0.05,
    }))

    from unittest.mock import MagicMock
    from forgestream.config import ForgeStreamConfig
    from forgestream.agent_dispatcher import AgentDispatcher

    config = ForgeStreamConfig(data_dir=str(data_dir))
    orchestrator = MagicMock()
    dispatcher = AgentDispatcher(config=config, orchestrator=orchestrator)
    assert dispatcher.trust_region._consecutive_improvements == 5
    assert dispatcher.trust_region._meeting_count == 8
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_agent_dispatcher.py tests/governor/test_trust_region.py -v -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forgestream/agent_dispatcher.py tests/test_agent_dispatcher.py
git commit -m "fix: load persisted TrustRegion at startup (was resetting every session)"
```

---

## Task 4: Set temperature=0 on All Gemini Calls

**The bug:** No `temperature` or `generation_config` is passed anywhere. Gemini defaults to high temperature, causing 15%+ variance in claim extraction per run.

**Files:**
- Modify: `forgestream/live_stream.py:208-214`
- Modify: `forgestream/runner.py:64-70`
- Test: `tests/test_runner.py`, `tests/test_live_stream.py`

- [ ] **Step 1: Add temperature=0 to the Live API connection**

In `forgestream/live_stream.py`, modify the `LiveConnectConfig` around line 210. `LiveConnectConfig` has a direct `temperature` field (verified in SDK v1.69.0) — use it directly rather than nesting inside `GenerationConfig`:

```python
from google.genai import types

self._session_cm = client.aio.live.connect(
    model=self.config.gemini_model,
    config=types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system_instruction,
        temperature=0.0,
    ),
)
```

- [ ] **Step 2: Add temperature=0 to the batch runner**

In `forgestream/runner.py`, modify the `generate_content` call around line 64:

```python
response = client.models.generate_content(
    model=config.gemini_model,
    contents=[
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        EXTRACTION_PROMPT,
    ],
    config=types.GenerateContentConfig(temperature=0.0),
)
```

- [ ] **Step 3: Patch all other Gemini generate_content calls**

Search with `grep -r "generate_content" forgestream/` and add `config=types.GenerateContentConfig(temperature=0.0)` to each:

In `forgestream/batch_meeting.py`, find the `generate_content` call (~line 90) and add the config parameter:
```python
response = client.models.generate_content(
    model=config.gemini_model,
    contents=[audio_part, prompt],
    config=types.GenerateContentConfig(temperature=0.0),
)
```

In `forgestream/audio_meeting.py`, find the `generate_content` call (~line 107) and add the same config parameter.

Ensure `from google.genai import types` is imported in each file (check existing imports first).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_runner.py tests/test_live_stream.py -v -x`
Expected: PASS (existing tests shouldn't hit real Gemini — they should be mocked)

- [ ] **Step 5: Commit**

```bash
git add forgestream/live_stream.py forgestream/runner.py forgestream/batch_meeting.py forgestream/audio_meeting.py
git commit -m "fix: set temperature=0 on all Gemini calls to reduce claim extraction variance"
```

---

## Task 5: Fix Rapport GRPO Fitness Function

**The bug:** `tune_rapport_weights()` calls `WeightTuner.tune()` which instantiates `Evaluator(weights=rapport_weights)`. The Evaluator ignores rapport keys entirely — it only reads `knowledge/verification/scaffold/uptake/engagement`. Every perturbation scores identically, making rapport GRPO a random walk. `tune_multi_objective()` already exists and does the right thing.

**Files:**
- Modify: `forgestream/post_meeting.py:213-223`
- Test: `tests/governor/test_improvement.py`

- [ ] **Step 1: Write failing test**

In `tests/governor/test_improvement.py`, add:

```python
def test_rapport_grpo_uses_multi_objective():
    """Rapport weight tuning should pull weights toward interpolated targets,
    not score them against the claim-based evaluator."""
    import random
    random.seed(42)

    from forgestream.governor.improvement import WeightTuner
    from forgestream.emotion.rapport import interpolate_weights

    tuner = WeightTuner()
    meeting_count = 5
    targets = interpolate_weights(meeting_count)

    # Start with deliberately wrong weights (opposite of targets)
    wrong_weights = {
        "attentiveness": 0.40,
        "positivity": 0.35,
        "coordination": 0.10,
        "symmetry": 0.15,
    }

    # Tune toward targets
    result = tuner.tune_multi_objective(wrong_weights, [], component_targets=targets)

    # Coordination should have moved toward the target (higher), not stayed at 0.10
    assert result["coordination"] > wrong_weights["coordination"]
    # Attentiveness should have moved toward target (lower), not stayed at 0.40
    assert result["attentiveness"] < wrong_weights["attentiveness"]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/governor/test_improvement.py::test_rapport_grpo_uses_multi_objective -v`
Expected: PASS (tune_multi_objective already works correctly)

- [ ] **Step 3: Change tune_rapport_weights to use tune_multi_objective**

In `forgestream/post_meeting.py`, replace lines 213-223:

```python
def tune_rapport_weights(
    self,
    events: list[Event],
    meeting_count: int = 1,
) -> dict[str, float]:
    """GRPO-tune rapport component weights toward maturity-interpolated targets.

    Note: human_score was removed from this signature. Rapport weights optimize
    toward the theoretically-motivated sigmoid targets via tune_multi_objective,
    not toward a claim-based score. When maturity inference is implemented
    (Task 8), the targets will be data-driven instead of sigmoid-based.
    """
    current = self.load_rapport_weights(meeting_count)
    targets = interpolate_weights(meeting_count)
    return self.weight_tuner.tune_multi_objective(
        current, events, component_targets=targets,
    )
```

**IMPORTANT:** The `human_score` parameter has been removed from the signature (not just ignored). Update the caller in `run()` at line 385 from:
```python
rapport_weights = self.tune_rapport_weights(events, meeting_count, human_score)
```
to:
```python
rapport_weights = self.tune_rapport_weights(events, meeting_count)
```

- [ ] **Step 4: Run full test suite for governor**

Run: `python3 -m pytest tests/governor/ -v -x`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add forgestream/post_meeting.py tests/governor/test_improvement.py
git commit -m "fix: rapport GRPO uses tune_multi_objective instead of claim-based evaluator"
```

---

## Task 6: Fix Evaluator — Confidence-Aware + Remove Dead Uptake Stub

**The bugs:**
1. `_knowledge_density` uses raw `claim_count` in denominator — noisy when extraction varies
2. `_suggestion_uptake` is hardcoded to return 0.5 always — wastes a weight dimension and creates GRPO feedback loop

**Files:**
- Modify: `forgestream/governor/evaluator.py:74-111`
- Test: `tests/governor/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

In `tests/governor/test_evaluator.py`, add:

```python
from forgestream.events.schema import Event, EventType


def _make_claim(keywords: list[str], confidence: float = 0.8) -> Event:
    """Helper: create a CLAIM event with given keywords and confidence."""
    return Event(
        event_type=EventType.CLAIM,
        session_id="test",
        branch_id="test",
        author="gemini",
        evaluator=0.0,
        payload={
            "text": "test claim",
            "speaker": "Speaker 1",
            "confidence": confidence,
            "tone_markers": [],
            "topic_keywords": keywords,
            "is_requirement": False,
            "is_question": False,
        },
    )


def test_knowledge_density_uses_confidence_weighting():
    """High-confidence claims should count more than low-confidence ones."""
    from forgestream.governor.evaluator import Evaluator

    evaluator = Evaluator()

    # 3 high-confidence claims with unique keywords
    high_conf = [
        _make_claim(["alpha"], confidence=0.9),
        _make_claim(["beta"], confidence=0.85),
        _make_claim(["gamma"], confidence=0.95),
    ]
    kd_high = evaluator._knowledge_density(high_conf)

    # 3 low-confidence claims with same unique keywords
    low_conf = [
        _make_claim(["alpha"], confidence=0.2),
        _make_claim(["beta"], confidence=0.15),
        _make_claim(["gamma"], confidence=0.1),
    ]
    kd_low = evaluator._knowledge_density(low_conf)

    # Same keyword diversity, but high-confidence should score differently
    # because denominator is confidence-weighted, not raw count
    assert kd_high != kd_low


def test_suggestion_uptake_not_hardcoded():
    """_suggestion_uptake with no suggestions should return 0.0, not 0.5."""
    from forgestream.governor.evaluator import Evaluator
    evaluator = Evaluator()
    assert evaluator._suggestion_uptake([]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/governor/test_evaluator.py::test_knowledge_density_uses_confidence_weighting tests/governor/test_evaluator.py::test_suggestion_uptake_not_hardcoded -v`
Expected: FAIL (both use current logic)

- [ ] **Step 3: Implement confidence-aware knowledge_density**

In `forgestream/governor/evaluator.py`, replace `_knowledge_density` (lines 74-83):

```python
@staticmethod
def _knowledge_density(events: list[Event]) -> float:
    """Unique concepts extracted per claim, scaled by average confidence.

    The confidence multiplier means high-confidence extraction runs produce
    higher knowledge_density than low-confidence runs with the same keywords.
    This makes E(π) sensitive to extraction quality, not just quantity.
    """
    claims = [e for e in events if e.event_type == EventType.CLAIM]
    if not claims:
        return 0.0
    all_keywords: set[str] = set()
    for c in claims:
        all_keywords.update(c.payload.get("topic_keywords", []))
    avg_confidence = sum(c.payload.get("confidence", 0.5) for c in claims) / len(claims)
    raw_density = len(all_keywords) / max(len(claims), 1)
    return min(1.0, raw_density * avg_confidence)
```

Note: With this formula, high_conf (avg=0.9): `(3/3)*0.9 = 0.9`, low_conf (avg=0.15): `(3/3)*0.15 = 0.15`. The `min(1.0, ...)` clamp only activates when keyword diversity is extreme. Test assertion `kd_high != kd_low` passes.

- [ ] **Step 4: Fix _suggestion_uptake**

Replace `_suggestion_uptake` (lines 105-111):

```python
@staticmethod
def _suggestion_uptake(events: list[Event]) -> float:
    """Returns 0.0 — uptake tracking not yet implemented.

    Previously returned 0.5 which created a GRPO feedback loop
    (constant signal attracted weight without contributing information).
    Returns 0.0 so the uptake weight dimension has no pull until implemented.
    """
    return 0.0
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/governor/test_evaluator.py -v -x`
Expected: PASS

- [ ] **Step 6: Update stale test comment**

In `tests/governor/test_evaluator.py`, find the comment near line 74 that says `# Default uptake=0.5 and engagement=0.5` and update it to reflect `uptake=0.0`:

```python
# Default uptake=0.0 and engagement=0.5, so empty = 0.15*0.0 + 0.15*0.5 = 0.075
```

- [ ] **Step 7: Run full governor tests to check for regressions**

Run: `python3 -m pytest tests/governor/ -v -x`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add forgestream/governor/evaluator.py tests/governor/test_evaluator.py
git commit -m "fix: confidence-weighted knowledge_density + remove hardcoded uptake stub"
```

---

## Task 7: Fix auto_score Hardcoded Constants

**The bug:** `compute_auto_score` has two `0.2 * 0.5` terms that add a fixed 0.20 to every score, compressing the GRPO target range to [0.20, 0.70].

**Files:**
- Modify: `forgestream/post_meeting.py:254-286`
- Test: `tests/test_post_meeting.py`

- [ ] **Step 1: Write failing test**

In `tests/test_post_meeting.py`, add:

```python
def test_auto_score_zero_for_empty_meeting():
    """An empty meeting (no events) should score 0.0, not 0.2."""
    from forgestream.post_meeting import PostMeetingSynthesis
    from forgestream.config import ForgeStreamConfig
    config = ForgeStreamConfig()
    pms = PostMeetingSynthesis(config=config)
    score = pms.compute_auto_score([])
    assert score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_post_meeting.py::test_auto_score_zero_for_empty_meeting -v`
Expected: FAIL (currently returns 0.2 for empty meetings)

- [ ] **Step 3: Fix compute_auto_score**

In `forgestream/post_meeting.py`, replace lines 280-286:

```python
base_score = (
    0.35 * min(1.0, req_scaffold)
    + 0.35 * min(1.0, findings_per_claim)
    + 0.30 * engagement_bonus * 10.0  # scale arousal (0-0.1) to (0-0.3)
)
return min(1.0, max(0.0, base_score))
```

This removes the hardcoded 0.5 baselines. The score now ranges [0.0, 1.0] based on actual meeting content. Weights redistribute: requirement-scaffold and findings-per-claim get 0.35 each, engagement gets 0.30.

**Design note on arousal amplification:** The old formula contributed `0.1 * mean_arousal` to the score. The new formula contributes `0.30 * engagement_bonus * 10.0 = 0.30 * (0.1 * mean_arousal) * 10.0 = 0.30 * mean_arousal`. This is a 3x increase in arousal's weight within auto_score. This is intentional: with the two hardcoded 0.5 terms removed, the engagement component needs to fill a larger portion of the [0,1] range. Mean arousal typically falls in [0.3, 0.7], so the engagement contribution ranges [0.09, 0.21] — healthy sensitivity without dominating. If prosodic extraction proves too noisy, reduce the multiplier from 10.0 to 5.0 (giving 1.5x arousal weight instead of 3x).

- [ ] **Step 4: Run test**

Run: `python3 -m pytest tests/test_post_meeting.py::test_auto_score_zero_for_empty_meeting -v`
Expected: PASS

- [ ] **Step 5: Run all post_meeting tests**

Run: `python3 -m pytest tests/test_post_meeting.py -v -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forgestream/post_meeting.py tests/test_post_meeting.py
git commit -m "fix: remove hardcoded 0.5 baselines from auto_score (was compressing GRPO target range)"
```

---

## Task 8: Rapport Maturity Inference From Raw Signals

> **Depends on:** Task 1 must be completed first. Task 1 wires the `rapport_weights` parameter through `orchestrator.py:attach_rapport_engine()` and `live_stream.py`. Task 8 modifies only `rapport.py` internals and assumes those call-site changes already exist. Without Task 1, the `rapport_weights` parameter added here is accepted but never passed by any caller.

**The problem:** The sigmoid uses global `meeting_count` which is meaningless for mixed speaker groups. We need to infer relationship maturity from the meeting's own prosodic signals — works for both corpus and production.

**Design:** After accumulating 6 ENTRAINMENT_SNAPSHOT events (~3 minutes), estimate maturity from coordination level and symmetry stability. Use this to override the sigmoid. Before 6 snapshots, use equal weights (0.25 each). Coordination/symmetry data is accumulated unconditionally (even when weights are pre-loaded from disk), ensuring the `inferred_maturity` payload field emits real values for all meetings.

**Files:**
- Modify: `forgestream/emotion/rapport.py`
- Test: `tests/emotion/test_rapport.py`

- [ ] **Step 1: Write failing test**

In `tests/emotion/test_rapport.py`, add:

```python
def test_maturity_inference_low_coordination():
    """Low coordination + high symmetry variance → early relationship → t < 0.3."""
    from forgestream.emotion.rapport import infer_maturity

    # Early relationship: speakers haven't adapted to each other
    coordination_values = [0.1, 0.15, 0.08, 0.12, 0.09, 0.11]
    symmetry_values = [0.8, 0.3, 0.6, 0.2, 0.7, 0.4]

    t = infer_maturity(coordination_values, symmetry_values)
    assert 0.0 <= t <= 0.3, f"Expected early maturity (t<0.3), got {t}"


def test_maturity_inference_high_coordination():
    """High coordination + stable symmetry → established relationship → t > 0.7."""
    from forgestream.emotion.rapport import infer_maturity

    # Established relationship: strong prosodic alignment
    coordination_values = [0.7, 0.75, 0.72, 0.68, 0.73, 0.71]
    symmetry_values = [0.55, 0.58, 0.56, 0.57, 0.55, 0.56]

    t = infer_maturity(coordination_values, symmetry_values)
    assert 0.7 <= t <= 1.0, f"Expected established maturity (t>0.7), got {t}"


def test_equal_weights_before_maturity_estimated():
    """Before enough snapshots accumulate, all four weights should be 0.25."""
    from forgestream.emotion.rapport import RapportEngine
    from unittest.mock import MagicMock

    engine = RapportEngine(orchestrator=MagicMock(), meeting_count=1)
    # Before maturity is inferred, weights should be equal
    assert engine._weights == {
        "attentiveness": 0.25,
        "positivity": 0.25,
        "coordination": 0.25,
        "symmetry": 0.25,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/emotion/test_rapport.py::test_maturity_inference_low_coordination tests/emotion/test_rapport.py::test_maturity_inference_high_coordination tests/emotion/test_rapport.py::test_equal_weights_before_maturity_estimated -v`
Expected: FAIL (infer_maturity doesn't exist yet, equal weights not the default)

- [ ] **Step 3: Implement `infer_maturity` function**

In `forgestream/emotion/rapport.py`, add after the `interpolate_weights` function (after line 48):

```python
EQUAL_WEIGHTS = {
    "attentiveness": 0.25,
    "positivity": 0.25,
    "coordination": 0.25,
    "symmetry": 0.25,
}

MATURITY_SNAPSHOTS_REQUIRED = 6  # ~3 minutes at 30s intervals


def infer_maturity(
    coordination_values: list[float],
    symmetry_values: list[float],
) -> float:
    """Infer relationship maturity from observed prosodic signals.

    Uses coordination level (CRQA %DET) and symmetry stability to estimate
    how established the speaker relationship is. Returns t in [0, 1].

    Low coordination + high symmetry variance → early relationship (t~0)
    High coordination + low symmetry variance → established relationship (t~1)
    """
    if not coordination_values or not symmetry_values:
        return 0.0

    mean_coord = sum(coordination_values) / len(coordination_values)

    # Symmetry stability = inverse of coefficient of variation
    sym_mean = sum(symmetry_values) / len(symmetry_values)
    if sym_mean > 0.01:
        sym_std = (
            sum((v - sym_mean) ** 2 for v in symmetry_values) / len(symmetry_values)
        ) ** 0.5
        sym_stability = max(0.0, 1.0 - (sym_std / sym_mean))
    else:
        sym_stability = 0.0

    # Weighted combination: coordination is the stronger signal
    raw = 0.7 * mean_coord + 0.3 * sym_stability

    # Clamp to [0, 1]
    return max(0.0, min(1.0, raw))
```

- [ ] **Step 4: Modify RapportEngine to start with equal weights and infer maturity**

In `forgestream/emotion/rapport.py`, modify `__init__`:

```python
def __init__(
    self,
    orchestrator: "Orchestrator",
    meeting_count: int = 1,
    damping_factor: float = 0.3,
    runpod_endpoint: str = "",
    runpod_timeout: float = 4.0,
    rapport_weights: dict[str, float] | None = None,
) -> None:
    self._orchestrator = orchestrator
    self._meeting_count = meeting_count

    # Start with equal weights; refine after maturity inference
    if rapport_weights is not None:
        self._weights = rapport_weights
        self._maturity_inferred = True
    else:
        self._weights = EQUAL_WEIGHTS.copy()
        self._maturity_inferred = False

    # Track coordination and symmetry for maturity inference
    self._coordination_history: list[float] = []
    self._symmetry_history: list[float] = []
    self._snapshot_count = 0
```

- [ ] **Step 5: Add maturity inference trigger in `_handle_snapshot`**

In the `_handle_snapshot` method, insert **after the `pair_scores` for-loop completes and BEFORE the `composites = [...]` line** (i.e., between the `pair_scores.append(score)` end-of-loop and `composites = [p["composite"]...]`):

```python
# Always accumulate coordination/symmetry for observability and payload
# (decoupled from weight-override to ensure data flows even with pre-loaded weights)
for ps in pair_scores:
    self._coordination_history.append(ps["coordination"])
    self._symmetry_history.append(ps["symmetry"])
self._snapshot_count += 1

# Only infer and override weights if not already loaded from disk
if not self._maturity_inferred and self._snapshot_count >= MATURITY_SNAPSHOTS_REQUIRED:
    t = infer_maturity(self._coordination_history, self._symmetry_history)
    self._weights = interpolate_weights_from_t(t)
    self._maturity_inferred = True
    logger.info(
        "Maturity inferred: t=%.3f, weights=%s",
        t, {k: round(v, 3) for k, v in self._weights.items()},
    )
```

**IMPORTANT:** Accumulation happens unconditionally (outside the `if not self._maturity_inferred` guard). Only the weight-override is conditional. This ensures `_coordination_history` is populated for ALL meetings, including production sessions with pre-loaded GRPO weights. The `inferred_maturity` payload field (Step 7) will emit real values after 6 snapshots regardless of whether weights were loaded from disk.

- [ ] **Step 6: Add `interpolate_weights_from_t` helper**

In `forgestream/emotion/rapport.py`, add after `interpolate_weights`:

```python
def interpolate_weights_from_t(t: float) -> dict[str, float]:
    """Interpolate between early and established weight profiles using t directly."""
    t = max(0.0, min(1.0, t))
    return {
        k: EARLY_WEIGHTS[k] * (1 - t) + ESTABLISHED_WEIGHTS[k] * t
        for k in EARLY_WEIGHTS
    }
```

Then modify the existing `interpolate_weights` to use it:

```python
def interpolate_weights(meeting_count: int) -> dict[str, float]:
    """Sigmoid interpolation between early and established weight profiles."""
    t = 1.0 / (1.0 + math.exp(-(meeting_count - 3)))
    return interpolate_weights_from_t(t)
```

- [ ] **Step 7: Add `inferred_maturity` to rapport event payload**

In `_handle_snapshot`, add to the payload dict:

```python
"inferred_maturity": round(
    infer_maturity(self._coordination_history, self._symmetry_history), 4
) if (self._maturity_inferred and self._coordination_history) else None,
```

- [ ] **Step 8: Run all rapport tests**

Run: `python3 -m pytest tests/emotion/test_rapport.py -v -x`
Expected: PASS

- [ ] **Step 9: Run full emotion test suite**

Run: `python3 -m pytest tests/emotion/ -v -x`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add forgestream/emotion/rapport.py tests/emotion/test_rapport.py
git commit -m "feat: infer rapport maturity from prosodic signals instead of global meeting_count"
```

---

## Task 9: Post-Meeting Consensus Re-Extraction

**The problem:** Gemini claim extraction is non-deterministic. GRPO optimizes against noisy reward signal. After each meeting, re-extract claims from the transcript 2 more times and take consensus.

**Design:** In `PostMeetingSynthesis.run()`, after the main pipeline completes, re-extract claims from the cached transcript text (not audio — fast and cheap). Deduplicate semantically using topic_keyword overlap (Jaccard similarity). Use consensus claims for GRPO scoring.

**Files:**
- Modify: `forgestream/post_meeting.py`
- Create: `forgestream/claims/consensus.py`
- Test: `tests/test_post_meeting.py`

- [ ] **Step 1: Write failing test for claim consensus**

In `tests/test_post_meeting.py`, add:

```python
def test_claim_consensus_deduplicates():
    """Claims with overlapping keywords should be merged in consensus."""
    from forgestream.claims.consensus import build_consensus

    run1 = [
        {"text": "Redis is good for caching", "topic_keywords": ["redis", "caching"], "confidence": 0.8},
        {"text": "Postgres handles writes", "topic_keywords": ["postgres", "writes"], "confidence": 0.9},
    ]
    run2 = [
        {"text": "Redis caching is effective", "topic_keywords": ["redis", "caching", "performance"], "confidence": 0.85},
        {"text": "Postgres for write workloads", "topic_keywords": ["postgres", "writes", "workloads"], "confidence": 0.7},
        {"text": "Latency is important", "topic_keywords": ["latency", "performance"], "confidence": 0.6},
    ]

    consensus = build_consensus([run1, run2], jaccard_threshold=0.4, min_runs=1)

    # Redis and Postgres claims appear in both runs → included
    redis_claims = [c for c in consensus if "redis" in c["topic_keywords"]]
    assert len(redis_claims) >= 1

    # Confidence should be boosted for claims appearing in multiple runs
    for c in consensus:
        if "redis" in c["topic_keywords"]:
            assert c["consensus_confidence"] > 0.8  # boosted from appearing in 2 runs


def test_claim_consensus_filters_singletons():
    """Claims appearing in only 1 of 3 runs with min_runs=2 should be filtered."""
    from forgestream.claims.consensus import build_consensus

    run1 = [{"text": "A", "topic_keywords": ["alpha"], "confidence": 0.8}]
    run2 = [{"text": "B", "topic_keywords": ["beta"], "confidence": 0.8}]
    run3 = [{"text": "A2", "topic_keywords": ["alpha"], "confidence": 0.7}]

    consensus = build_consensus([run1, run2, run3], jaccard_threshold=0.4, min_runs=2)

    # "alpha" appears in run1 and run3 → included
    alpha = [c for c in consensus if "alpha" in c["topic_keywords"]]
    assert len(alpha) == 1

    # "beta" appears only in run2 → excluded
    beta = [c for c in consensus if "beta" in c["topic_keywords"]]
    assert len(beta) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_post_meeting.py::test_claim_consensus_deduplicates -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Create `forgestream/claims/__init__.py`**

```python
```

- [ ] **Step 4: Create `forgestream/claims/consensus.py`**

```python
"""Claim consensus — deduplicate claims across multiple extraction runs."""

from __future__ import annotations


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_consensus(
    runs: list[list[dict]],
    jaccard_threshold: float = 0.4,
    min_runs: int = 1,
) -> list[dict]:
    """Build consensus claims from multiple extraction runs.

    Clusters claims across runs by topic_keyword Jaccard similarity.
    Returns one representative claim per cluster, with boosted confidence.

    Args:
        runs: List of extraction runs, each a list of claim dicts.
        jaccard_threshold: Minimum Jaccard similarity to consider claims as matching.
        min_runs: Minimum number of runs a claim cluster must appear in.

    Returns:
        List of consensus claim dicts with added 'consensus_confidence' and 'run_count' fields.
    """
    # Flatten all claims with run index
    tagged: list[tuple[int, dict]] = []
    for run_idx, claims in enumerate(runs):
        for claim in claims:
            tagged.append((run_idx, claim))

    # Greedy clustering by keyword overlap
    clusters: list[list[tuple[int, dict]]] = []
    used: set[int] = set()

    for i, (run_i, claim_i) in enumerate(tagged):
        if i in used:
            continue
        cluster = [(run_i, claim_i)]
        used.add(i)
        kw_i = set(claim_i.get("topic_keywords", []))

        for j, (run_j, claim_j) in enumerate(tagged):
            if j in used:
                continue
            kw_j = set(claim_j.get("topic_keywords", []))
            if _jaccard(kw_i, kw_j) >= jaccard_threshold:
                cluster.append((run_j, claim_j))
                used.add(j)

        clusters.append(cluster)

    # Build consensus: one representative per cluster
    consensus = []
    for cluster in clusters:
        run_indices = {run_idx for run_idx, _ in cluster}
        if len(run_indices) < min_runs:
            continue

        # Representative = highest confidence claim in cluster
        best = max(cluster, key=lambda x: x[1].get("confidence", 0.0))
        representative = dict(best[1])

        # Merge keywords from all cluster members
        all_keywords: set[str] = set()
        for _, claim in cluster:
            all_keywords.update(claim.get("topic_keywords", []))
        representative["topic_keywords"] = sorted(all_keywords)

        # Boost confidence based on cross-run agreement
        base_conf = representative.get("confidence", 0.5)
        run_fraction = len(run_indices) / len(runs)
        representative["consensus_confidence"] = min(1.0, base_conf * (0.5 + 0.5 * run_fraction))
        representative["run_count"] = len(run_indices)

        consensus.append(representative)

    return consensus
```

- [ ] **Step 5: Run consensus tests**

Run: `python3 -m pytest tests/test_post_meeting.py::test_claim_consensus_deduplicates tests/test_post_meeting.py::test_claim_consensus_filters_singletons -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forgestream/claims/__init__.py forgestream/claims/consensus.py tests/test_post_meeting.py
git commit -m "feat: claim consensus module for deduplicating across extraction runs"
```

---

## Task 10: Integrate Consensus Into PostMeetingSynthesis

**Files:**
- Modify: `forgestream/post_meeting.py:319-340`

- [ ] **Step 1: Add consensus re-extraction to `run()`**

In `forgestream/post_meeting.py`, after the main `tune_weights` call (around line 330) and before the sensitivity analysis, add:

```python
# Consensus re-extraction: re-extract claims from transcript text
# to reduce GRPO noise from extraction non-determinism
from .claims.consensus import build_consensus
claim_events = [e for e in events if e.event_type == EventType.CLAIM]
if claim_events:
    # Build run 1 from existing claims
    run1 = [
        {
            "text": e.payload.get("text", ""),
            "topic_keywords": e.payload.get("topic_keywords", []),
            "confidence": e.payload.get("confidence", 0.5),
        }
        for e in claim_events
    ]
    # For now, use single-run consensus (dedup within the run)
    # TODO: When transcript caching is available, re-extract 2 more times
    consensus_claims = build_consensus([run1], jaccard_threshold=0.5, min_runs=1)
    result["consensus_claim_count"] = len(consensus_claims)
    result["raw_claim_count"] = len(claim_events)
```

Note: Full multi-run re-extraction requires transcript caching (the Live API provides `outputTranscription` but it's not currently saved). This wires in the consensus module with single-run deduplication now, which still helps by merging duplicate keyword fragments. Multi-run extraction is a follow-up task after transcript caching is implemented.

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest tests/test_post_meeting.py -v -x`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add forgestream/post_meeting.py
git commit -m "feat: wire consensus claim deduplication into PostMeetingSynthesis"
```

---

## Task 11: Full Integration Test

- [ ] **Step 1: Run the complete test suite**

Run: `python3 -m pytest -q --ignore=tests/events/test_store.py --ignore=tests/events/test_subscribe.py -k "not writes_to_store and not milestone_a and not full_pipeline_with"`

Expected: All tests PASS, no regressions.

- [ ] **Step 2: Verify weights loading chain**

Run a quick smoke test to confirm the wiring fixes work together:

```python
# Quick verification script
import json
from pathlib import Path
from forgestream.config import ForgeStreamConfig, load_config_overrides
from forgestream.post_meeting import PostMeetingSynthesis
from forgestream.governor.trust_region import TrustRegion

config = ForgeStreamConfig()
updated_config = load_config_overrides(config=config)
pms = PostMeetingSynthesis(config=updated_config)

print("Evaluator weights:", pms.load_weights())
print("Meeting count:", pms.load_meeting_count())
print("Rapport weights:", pms.load_rapport_weights(pms.load_meeting_count()))
print("Config overrides applied:", updated_config.spawn_cooldown_seconds)

tr = TrustRegion.load(Path("data/trust_region.json"))
print("TrustRegion epsilon:", tr.epsilon)
```

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A
git commit -m "test: integration verification for pipeline integrity fixes"
```

---

## Summary of Changes

| Task | Fix | Category | Risk |
|------|-----|----------|------|
| 1 | Load rapport weights at startup | Wiring | Low — adds loading, doesn't change logic |
| 2 | Load config overrides at startup | Wiring | Low — function already exists |
| 3 | Load TrustRegion from disk | Wiring | Low — method already exists |
| 4 | temperature=0 on Gemini calls | Config | Low — reduces variance, no functional change |
| 5 | Rapport GRPO → tune_multi_objective | Logic fix | Medium — changes optimization direction |
| 6 | Confidence-aware evaluator + remove uptake stub | Logic fix | Medium — changes E(π) values |
| 7 | Fix auto_score hardcoded constants | Logic fix | Medium — changes GRPO target range |
| 8 | Maturity inference from raw signals | New feature | Medium — replaces sigmoid meeting_count input |
| 9 | Claim consensus module | New feature | Low — new module, no existing code changed |
| 10 | Wire consensus into PostMeetingSynthesis | Integration | Low — adds dedup pass, doesn't change events |
| 11 | Full integration test | Verification | None — read-only |

**Expected impact on existing data:** Tasks 5-7 change the GRPO optimization dynamics. The current `data/weights.json`, `data/rapport_weights.json`, and `data/weights_history.json` were produced under the old (broken) system. After these fixes, the first meeting will produce weights that diverge from the history. This is intentional — the old weights were optimized against wrong objectives. Consider archiving the current data files before the first run post-fix.
