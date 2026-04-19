# Sentinel — Real-Time Cognitive Amplification for Claude Code Sessions

**Author:** Mergim Gashi + Claude Opus 4.6
**Date:** 2026-03-29
**Status:** Design approved, pending implementation plan

## 1. Vision

Sentinel is a real-time monitor system that augments Claude Code sessions through parallel analysis terminals, curated insight injection, and SOS-guaranteed self-improvement. It operates as a cognitive amplifier — a second brain that observes, challenges, enhances, and learns alongside the user.

The system consists of:
- A background **Sentinel Daemon** that captures session events and runs fast-path analysis
- Multiple **Mode Terminals** (Claude Code instances), each dedicated to a specific analysis mode
- A **Hook Pipeline** on the working terminal that feeds events and delivers injections
- An **SOS-guaranteed GRPO** optimization loop that improves each mode's effectiveness over time

### Three Timescales

| Layer | Timescale | What Runs | Where |
|-------|-----------|-----------|-------|
| **Wire** | Milliseconds | Hook event → daemon → fast-path response | Daemon (heuristics, no LLM) |
| **Intelligence** | Seconds | Mode terminal → deep analysis → insight curation | Mode terminals (full Claude) |
| **Evolution** | Every 3 turns | GRPO optimization → strategy refinement → persistence | Daemon (pure math) |

### Theoretical Foundation

Every mode's optimization loop is a Sacred Object System (SOS) instance per Definition 2.1 (Gashi, 2026). Convergence follows as a free theorem from three axioms: Monotone Improvement, Bounded Step, Constraint Preservation. The system implements the dual-SOS pattern from ProofForge: weight-space GRPO (daemon) alongside context-space pattern accumulation (terminals).

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENTINEL DAEMON                               │
│                     (FastAPI, localhost:9100)                         │
│                                                                      │
│   HTTP Server            Event Bus            Injection Queue        │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │ Receives hook │─▶│ Routes to    │─▶│ pending_injections     │   │
│   │ events from   │  │ mode queues  │  │ .jsonl                 │   │
│   │ working term  │  │              │  │ (read by hooks)        │   │
│   └──────────────┘  └──────┬───────┘  └────────────────────────┘   │
│                             │                                        │
│   GRPO Engine               │  State Manager                        │
│   ┌──────────────┐          │  ┌──────────────────┐                 │
│   │ Per-mode SOS │          │  │ session_signal    │                 │
│   │ instances    │          │  │ session_context   │                 │
│   │ (3-turn cycle│          │  │ convergence data  │                 │
│   │  convergence │          │  └──────────────────┘                 │
│   │  guaranteed) │          │                                        │
│   └──────────────┘          │                                        │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  ORANGE TEAM   │  │  PROMPT FORGE  │  │  INSIGHT MINER │
│  Claude Term.  │  │  Claude Term.  │  │  Claude Term.  │
│                │  │                │  │                │
│  Adversarial   │  │  Prompt        │  │  Learning      │
│  stress-test   │  │  enhancement   │  │  extraction    │
│  of working    │  │  and strategy  │  │  from thinking │
│  session       │  │  optimization  │  │  blocks        │
│                │  │                │  │                │
│  User: dig     │  │  User: tune,   │  │  User: review, │
│  deeper, inject│  │  inject, edit  │  │  store, recall │
└────────────────┘  └────────────────┘  └────────────────┘

┌────────────────┐
│  COACH         │
│  Claude Term.  │
│                │
│  Real-time     │
│  commentary,   │
│  prompt advice,│
│  meta-analysis │
└────────────────┘

┌──────────────────────────────────────────────────────┐
│              WORKING TERMINAL                         │
│                                                       │
│  User ↔ Claude (normal work)                          │
│                                                       │
│  Hooks:                                               │
│  - UserPromptSubmit → POST :9100/hook/user_prompt     │
│  - PreToolUse (Agent) → POST :9100/hook/pre_tool      │
│  - PostToolUse → POST :9100/hook/post_tool            │
│  - Stop → POST :9100/hook/stop                        │
│                                                       │
│  Reads: pending_injections.jsonl (on each prompt)     │
└──────────────────────────────────────────────────────┘
```

### Terminal Layout (tmux)

```
┌───────────────────────────────┬───────────────────────┐
│                               │    COACH              │
│    WORKING TERMINAL           │    (commentary +      │
│    (primary work)             │     suggestions)      │
│                               ├───────────────────────┤
│                               │    ORANGE TEAM        │
│                               │    (challenges)       │
├───────────────────────────────┼───────────────────────┤
│    PROMPT FORGE               │    INSIGHT MINER      │
│    (enhancements)             │    (learnings)        │
└───────────────────────────────┴───────────────────────┘
```

## 3. Sentinel Daemon

### 3.1 Technology

FastAPI Python server, ~400 lines. No LLM calls. Pure event routing, state management, and GRPO computation.

**Environment requirements:** `BRAINTRUST_API_KEY` must be set (for Layer 2 hydration — querying traces). Reads from `~/.claude/.env` if available.

**Graceful degradation:** If the daemon is down, HTTP hooks fail open (non-2xx = hook skipped). The working terminal proceeds normally with no injection or event capture. Events during daemon downtime are lost — this is acceptable because modes catch up via Braintrust trace replay on reconnect.

### 3.2 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/hook/user_prompt` | POST | Receive UserPromptSubmit hook events. Routes to mode queues. Returns pending injections as `additionalContext`. |
| `/hook/pre_tool` | POST | Receive PreToolUse hook events (Agent/Task). Can return `updatedInput` for subagent prompt rewriting. |
| `/hook/post_tool` | POST | Receive PostToolUse hook events. Routes to mode queues for outcome scoring. |
| `/hook/stop` | POST | Receive Stop hook events. Triggers end-of-turn processing. |
| `/context/{mode}` | GET | Progressive context bootstrap — Layer 1 snapshot. |
| `/context/{mode}/hydrate` | GET | Rich context — Layer 2, triggers Braintrust trace query. |
| `/inject` | POST | Mode terminal submits approved injection. |
| `/queue/{mode}` | GET | Pending insights for a mode. |
| `/grpo/{mode}` | GET | Current SOS state and GRPO trajectory for a mode. |
| `/grpo/{mode}/score` | POST | Submit outcome scoring data. |
| `/health` | GET | Daemon status, connected modes, active session. |

### 3.3 State Files

```
~/.claude/state/sentinel/
├── daemon.pid
├── session_signal.json                 # Layer 0: minimal signal
├── session_context.json                # Layer 1: full snapshot
├── pending_injections.jsonl            # Approved injections for working terminal
├── modes/
│   ├── prompt_forge/
│   │   ├── events.jsonl                # Event queue
│   │   ├── insights.jsonl              # Generated insights
│   │   ├── sos_state.json              # Strategy weights + orbit
│   │   └── context_library.jsonl       # Accumulated effective patterns
│   ├── orange_team/
│   │   ├── events.jsonl
│   │   ├── insights.jsonl
│   │   ├── sos_state.json
│   │   └── context_library.jsonl
│   └── insight_miner/
│       ├── events.jsonl
│       ├── insights.jsonl
│       ├── sos_state.json
│       └── context_library.jsonl
└── grpo/
    ├── trajectories.jsonl              # Historical orbit sequences a(n)
    └── convergence.json                # Per-mode convergence tracking
```

## 4. Hook Pipeline

Four HTTP hooks on the working terminal. All `"type": "http"` pointing at the daemon.

### 4.1 UserPromptSubmit Hook

```json
{
  "type": "http",
  "url": "http://localhost:9100/hook/user_prompt",
  "timeout": 5
}
```

- **Sends:** `{prompt, session_id, timestamp, cwd}`
- **Daemon actions:** routes event to all mode queues, updates `session_context.json`, checks `pending_injections.jsonl`
- **Returns:** `{additionalContext: "[SENTINEL:MODE] ..."}` if injections pending, empty otherwise
- **This is the injection delivery point.** The daemon reads pending injections, returns them, clears the file.

### 4.2 PreToolUse Hook (Agent/Task)

```json
{
  "matcher": "Task|Agent",
  "type": "http",
  "url": "http://localhost:9100/hook/pre_tool",
  "timeout": 10
}
```

- **Sends:** `{tool_name, tool_input, session_id}`
- **Returns:** optionally `{updatedInput: {...}}` if Prompt Forge has a high-confidence subagent prompt rewrite
- **This is the literal prompt rewrite point** for subagent calls. The `updatedInput` replaces the entire tool input, including the `prompt` field.

### 4.3 PostToolUse Hook

```json
{
  "matcher": "*",
  "type": "http",
  "url": "http://localhost:9100/hook/post_tool",
  "timeout": 3
}
```

- **Sends:** `{tool_name, tool_input, tool_output_summary, session_id}`
- **Daemon actions:** routes to mode queues, updates `session_context.json`
- **Returns:** empty (observational, no injection)

### 4.4 Stop Hook

```json
{
  "type": "http",
  "url": "http://localhost:9100/hook/stop",
  "timeout": 5
}
```

- **Sends:** `{session_id, transcript_path}`
- **Daemon actions:** triggers end-of-turn processing, increments turn counter, checks if GRPO cycle due (every 3 turns)
- **Returns:** empty

## 5. Mode Terminals

### 5.1 General Architecture

Each mode terminal is a Claude Code instance launched in a mode-specific directory:

```bash
cd ~/.claude/sentinel/modes/orange-team && claude
```

The directory contains a `CLAUDE.md` that defines the mode's identity, event consumption pattern, interaction commands, and SOS strategy vector. The terminal has full filesystem access to the working project for deep analysis.

Mode terminal directories (identity/config — separate from runtime state at `~/.claude/state/sentinel/`):

```
~/.claude/sentinel/
├── daemon.py                           # Sentinel daemon entry point
├── modes/
│   ├── prompt-forge/
│   │   └── CLAUDE.md
│   ├── orange-team/
│   │   └── CLAUDE.md
│   ├── insight-miner/
│   │   └── CLAUDE.md
│   └── coach/
│       └── CLAUDE.md
```

### 5.2 Prompt Forge

| Aspect | Detail |
|--------|--------|
| **Purpose** | Analyze user prompts, generate enhanced versions, learn what works |
| **Events consumed** | `user_prompt` (primary), `post_tool` + `stop` (outcome scoring) |
| **Strategy vector** | `{specificity: 0.2, constraints: 0.15, context: 0.2, examples: 0.1, anti_patterns: 0.15, success_criteria: 0.1, simplify: 0.1}` |
| **Insight format** | Original prompt → enhanced prompt → confidence → reasoning |
| **Auto-inject** | Yes, at daemon heuristic confidence ≥ 0.9 (fast-path, no terminal needed) |
| **GRPO scoring** | First-try completion (+1.0), user correction (-1.0), no feedback (0.0) |
| **User interactions** | "Enhance this prompt", "Why did you add X?", "Tune down specificity" |

**Fast-path auto-injection heuristics (daemon, no LLM):**
- Prompt < 20 chars → inject recently edited file paths (from session_context)
- No file references in prompt → suggest relevant files
- No constraint language ("don't", "must", "only") → no action
- Fires only if strategy weight for that category ≥ 0.2 AND confidence ≥ 0.9

### 5.3 Orange Team

| Aspect | Detail |
|--------|--------|
| **Purpose** | Adversarially stress-test the working Claude's reasoning and output |
| **Events consumed** | `user_prompt`, `stop` (thinking blocks via Braintrust), `post_tool` (Edit/Write for code review) |
| **Strategy vector** | `{security: 0.2, edge_cases: 0.2, scalability: 0.15, intent_drift: 0.2, pattern_violations: 0.15, error_handling: 0.1}` |
| **Insight format** | Challenge → severity (critical/warning/info) → evidence → suggested injection |
| **Auto-inject** | Never — always curated by user |
| **GRPO scoring** | Adopted + code changed (+1.0), dismissed (-0.5), adopted + bug prevented (+2.0 bonus) |
| **User interactions** | "Dig deeper on security", "What if N=10M?", "Check the error path" |

### 5.4 Insight Miner

| Aspect | Detail |
|--------|--------|
| **Purpose** | Extract decisions, patterns, solutions, failures for cross-session learning |
| **Events consumed** | ALL events, plus Braintrust traces (thinking blocks are primary source) |
| **Strategy vector** | `{decisions: 0.2, solutions: 0.2, failures: 0.2, patterns: 0.2, extraction_threshold: 0.1, novelty_bias: 0.1}` |
| **Insight format** | Learning → type (decision/solution/failure/pattern) → context → tags → confidence |
| **Auto-inject** | No injection — insights stored to memory/recall system |
| **GRPO scoring** | Recalled in future session (+1.0), rated useful (+2.0), never recalled (-0.3) |
| **User interactions** | "What have you extracted?", "Dismiss that", "Connect this to last week's auth decision" |

### 5.5 Coach

| Aspect | Detail |
|--------|--------|
| **Purpose** | Real-time commentary, prompt suggestions, meta-analysis across all modes |
| **Events consumed** | ALL mode insight queues + conversation flow + session_context |
| **Strategy vector** | None — Coach is qualitative, not GRPO-optimized |
| **Output** | Narrative coaching: prompt suggestions, thinking block summaries, cross-mode synthesis |
| **Auto-inject** | Never — Coach speaks to the user, not to the working Claude |
| **User interactions** | "What should I ask next?", "Summarize Claude's thinking", "Which insights matter most?" |

## 6. Progressive Context Hydration

When a mode terminal starts or reconnects, it bootstraps via four layers:

### Layer 0 — Signal (< 50ms)

Reads `~/.claude/state/sentinel/session_signal.json`:

```json
{
  "active": true,
  "session_id": "59e8dd8a-...",
  "working_dir": "/Users/mghome/projects/forgestream",
  "current_file": "forgestream/governor/trust_region.py",
  "turn_count": 12,
  "last_event_ts": "2026-03-29T16:51:42Z"
}
```

Enough to orient. The terminal knows a session exists, where, and how far along.

### Layer 1 — Snapshot (< 500ms)

Calls `GET /context/{mode}`. Returns:

```json
{
  "session_summary": "User is implementing trust region history persistence...",
  "files_touched": ["forgestream/governor/trust_region.py", "tests/governor/test_trust_region_history.py"],
  "recent_tools": [
    {"tool": "Read", "file": "trust_region.py", "turn": 9},
    {"tool": "Edit", "file": "trust_region.py", "turn": 10},
    {"tool": "Bash", "command": "pytest tests/governor/", "turn": 11}
  ],
  "recent_prompts": [
    {"turn": 10, "summary": "Add history tracking to trust region"},
    {"turn": 11, "summary": "Run tests"},
    {"turn": 12, "summary": "Fix failing assertion"}
  ],
  "sos_state": {"weights": {"...": "..."}, "orbit_position": 7, "last_evaluator": 0.42},
  "pending_insights": 2
}
```

Terminal is now productive — can start generating insights.

### Layer 2 — Hydration (< 3s, async)

Calls `GET /context/{mode}/hydrate`. Daemon queries Braintrust for recent traces:

```json
{
  "recent_turns": [
    {
      "turn": 12,
      "user_prompt": "fix the assertion error on line 45",
      "thinking_summary": "Claude identified mismatch between expected JSON structure...",
      "thinking_raw": "The error is on line 45 where...[truncated to 2000 chars]",
      "response_summary": "Fixed assertion, test passes",
      "tools_used": ["Read", "Edit", "Bash"],
      "token_count": 5200
    }
  ],
  "session_token_total": 48000,
  "braintrust_trace_id": "59e8dd8a-..."
}
```

Mode terminal now has thinking blocks, full prompts, response details.

### Layer 3 — Deep (on-demand)

Fetched when user or mode analysis requires it:
- Complete session history (all turns, all thinking blocks)
- Cross-session learnings (memory recall for related past sessions)
- Historical GRPO trajectories
- Related sessions by topic similarity

## 7. Injection Mechanism

### 7.1 Injection Lifecycle

```
1. Event occurs in working terminal
2. Hook POSTs to daemon → routes to mode queues
3. Mode terminal reads events, runs analysis
4. Mode terminal generates insight, displays to user
5. User reviews: approve / dismiss / edit
6. If approved: mode terminal POSTs to /inject
7. Daemon writes to pending_injections.jsonl
8. User sends next prompt in working terminal
9. UserPromptSubmit hook reads pending_injections.jsonl
10. Hook returns as additionalContext
11. Working Claude sees: [SENTINEL:MODE] content
12. Daemon clears delivered injection, logs for GRPO scoring
```

### 7.2 Auto-Injection Path (Prompt Forge Only)

For confidence ≥ 0.9, the daemon applies fast-path heuristics without waiting for the mode terminal:
- Check prompt length, file references, constraint language
- If heuristic fires AND strategy weight ≥ 0.2: inject directly
- Tagged `[SENTINEL:AUTO]` for user visibility
- If user corrects after auto-injection → confidence threshold raised

### 7.3 Injection Format

```
[SENTINEL:{MODE}] {content}
```

Tagged, short, immediately identifiable. Appears as a system-reminder to the working Claude alongside existing hook output.

### 7.4 Subagent Prompt Rewriting

When the working Claude spawns a subagent via Agent/Task tool, the PreToolUse hook can return `updatedInput` with a modified `prompt` field. This is literal prompt rewriting — the subagent receives the modified prompt, not the original.

Controlled by Prompt Forge strategy vector. Only fires at high confidence. The `updatedInput` must include ALL fields from the original `tool_input`, not just the modified `prompt`.

## 8. SOS Convergence Framework

### 8.1 SOS Definition (Per Mode)

Each mode (except Coach) is an SOS instance (Π, d, E, Δ, C, δ):

- **(Π, d)** — strategy weight vectors forming a metric simplex in ℝⁿ
- **E: Π → ℝ** — mode-specific bounded evaluator (bounded in [0, 1])
- **Δ > 0** — step bound derived from 70/30 blend ratio
- **C: Π → Prop** — constraint predicate (weights positive, sum to 1, within tunable bounds)
- **δ: Π → Π** — GRPO update operator with monotone improvement gate

By Theorem 2.5 (Gashi, 2026), the orbit sequence a(n) = E(δⁿ(π₀)) converges. By Theorem 2.9, per-step improvement vanishes — the system naturally dampens, resolving meta-instability.

### 8.2 Constraint Lifting (Safety Rails)

Per Theorem 2.11, safety constraints are added without breaking convergence:

```python
def safe_update(sos, policy, interactions):
    candidate = sos.update(policy, interactions)
    if sos.constraint(candidate):
        return candidate
    return policy  # reject, convergence preserved
```

Applied constraints:
- Auto-inject confidence floor: ≥ 0.7
- Max curated injections per turn: [1, 5]
- Max tokens per turn: [200, 800]
- Cooldown after consecutive dismissals: [3, 10] turns

### 8.3 Dual-SOS Pattern

Each mode runs two SOS instances in parallel (ported from ProofForge's `pf-cli/src/main.rs`):

**SOS-1: Weight-space (daemon)**
- Policy space: strategy weight vectors
- Update: GRPO perturbation → scoring → 70/30 blend
- Pure math, no LLM needed

**SOS-2: Context-space (mode terminal)**
- Policy space: accumulated library of effective patterns
- Update: append verified patterns (monotone by construction)
- Needs Claude reasoning for classification and curation
- **Curation preserves monotonicity:** the mode terminal may prune duplicates or merge similar patterns, but never deletes — stale patterns are archived (moved to a separate `context_library_archive.jsonl`). Information content never decreases, preserving the SOS axiom.

```
ProofForge:                    Sentinel:
GRPO on strategy logits   →   Daemon runs GRPO on mode weights
Proof memory accumulation  →   Mode terminal accumulates insight patterns
```

### 8.4 Jensen Bridge Relevance

The stochastic extension (Layer 2 of the SOS paper) applies directly: GRPO perturbations are Gaussian-random. The Jensen bridge proves that stochastic noise doesn't degrade convergence — it accelerates it via the variance term. The randomness in perturbation generation is a feature, not a risk.

The Neural Thickets connection (Layer 3): diverse perturbations produce task-specialist strategies. Each mode converges to its own optimal strategy through this diversity.

## 9. GRPO Engine

### 9.1 Location and Timing

Lives in the daemon. Pure Python math, no LLM calls.

**Trigger frequency:** Every 3 turns (configurable). Also runs at session end.

### 9.2 Algorithm

```python
class SentinelGRPO:
    def tune(self, mode: str, interactions: list[Interaction]) -> dict:
        sos = self.load_sos(mode)
        current = sos.weights
        current_score = sos.evaluator(current, interactions)

        # Generate 10 Gaussian perturbations (σ = 0.05)
        perturbations = generate_perturbations(current, n=10, sigma=0.05)

        # Score each
        scored = [(sos.evaluator(p, interactions), p) for p in perturbations]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_weights = scored[0]

        # SOS Monotone Improvement gate
        if best_score <= current_score:
            self.record_orbit(mode, current_score, current)
            return current  # no improvement, reject

        # Blend 70/30 (Bounded Step)
        blended = {k: 0.7 * current[k] + 0.3 * best_weights[k] for k in current}
        normalized = normalize_weights(blended)

        # Constraint Preservation (Theorem 2.11)
        if not sos.constraint(normalized):
            self.record_orbit(mode, current_score, current)
            return current  # constraint lifting: reject

        final_score = sos.evaluator(normalized, interactions)
        self.record_orbit(mode, final_score, normalized)
        self.save_sos(mode, normalized)
        return normalized
```

### 9.3 Scoring Signals

| Signal | Source | Detection Method |
|--------|--------|------------------|
| First-try completion | User sends new topic without correction | No "no", "wrong", "instead" in next prompt |
| User correction | User rephrases or corrects | Next prompt references same topic with different phrasing |
| Challenge adopted | Code changed after injection | File diff between pre/post injection |
| Learning recalled | Memory system matches stored insight | Recall script returns the insight in future session |
| Prompt specificity | Automated analysis | Regex + heuristic: file paths, function names, constraints present? |

### 9.4 Orbit Tracking

The daemon persists the orbit sequence a(n) for each mode:

```jsonl
{"mode": "prompt_forge", "iteration": 0, "evaluator": 0.31, "weights": {...}, "ts": "..."}
{"mode": "prompt_forge", "iteration": 1, "evaluator": 0.35, "weights": {...}, "ts": "..."}
{"mode": "prompt_forge", "iteration": 2, "evaluator": 0.38, "weights": {...}, "ts": "..."}
```

Monotone non-decreasing — guaranteed by SOS axioms. Convergence gap `M - a(n)` tracked for diagnostics.

## 10. Noise Budget

### 10.1 Per-Turn Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max auto-injections per turn | 1 | Only highest-confidence Prompt Forge |
| Max curated injections per turn | 3 | User-approved but still bounded |
| Max total injected tokens per turn | 500 | Prevents context window pollution |
| Max total injected tokens per session | 5000 | Long session budget |

### 10.2 Per-Mode Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max queued insights per mode | 10 | Prevents stale accumulation |
| Cooldown after 3 consecutive dismissals | 5 turns | Mode backs off when not useful |
| Min turns between same-category injections | 2 | Prevents repetition |

### 10.3 GRPO-Tunable Bounds

All limits are part of the SOS weight space. GRPO can tune within constraint bounds:

```python
noise_constraints = {
    "auto_inject_confidence_min": (0.7, 1.0),
    "max_curated_per_turn": (1, 5),
    "max_tokens_per_turn": (200, 800),
    "cooldown_after_dismissals": (3, 10),
}
```

Constraint lifting ensures bounds are always respected.

## 11. Data Model

### 11.1 SentinelEvent

```python
@dataclass
class SentinelEvent:
    event_id: str               # UUID
    event_type: str             # "user_prompt" | "pre_tool" | "post_tool" | "stop"
    session_id: str             # Working terminal session
    timestamp: datetime
    payload: dict               # Type-specific data
```

### 11.2 Insight

```python
@dataclass
class Insight:
    insight_id: str             # "{mode}-{date}-{seq}"
    mode: str                   # "prompt_forge" | "orange_team" | "insight_miner"
    content: str                # Human-readable insight text
    injectable_text: str        # Ready-to-inject system-reminder text
    confidence: float           # 0.0 - 1.0
    severity: str               # "critical" | "warning" | "info"
    evidence: list[str]         # What triggered this insight
    suggested_action: str       # "inject" | "coach" | "store"
    status: str                 # "pending" | "approved" | "dismissed" | "injected"
    created_at: datetime
    resolved_at: datetime | None
    outcome: str | None         # "adopted" | "ignored" | "reversed"
```

### 11.3 SOSState

```python
@dataclass
class SOSState:
    mode: str
    weights: dict[str, float]           # Current strategy vector π
    orbit: list[float]                  # a(n) sequence
    iteration: int                      # Current orbit position n
    initial_weights: dict[str, float]   # π₀
    convergence_gap: float              # M - a(n)
    last_updated: datetime
    weight_bounds: dict[str, tuple[float, float]]
    noise_bounds: dict[str, tuple[float, float]]
```

### 11.4 Interaction

```python
@dataclass
class Interaction:
    turn: int
    user_prompt: str
    injections_delivered: list[str]
    auto_injections: list[str]
    response_quality: float | None
    user_correction: bool
    task_completed: bool
    insights_adopted: list[str]
    insights_dismissed: list[str]
```

## 12. Inter-Process Communication

| Channel | Mechanism | Used For |
|---------|-----------|----------|
| Working terminal → Daemon | HTTP hooks (native `"type": "http"`) | Event delivery, injection retrieval |
| Daemon → Mode terminals | File-based event queues (`events.jsonl`) | Event distribution |
| Mode terminals → Daemon | HTTP API via `curl` from Bash | Injection submission, scoring |

**Why file-based for daemon → terminal:** Mode terminals are Claude Code instances. They cannot listen on sockets. They read files via Read tool or periodic `tail`. The `/loop` skill or background `tail -f` gives near-real-time event consumption.

**Why HTTP for terminal → daemon:** Mode terminals can make HTTP calls via Bash `curl`. Simple, reliable, no dependencies.

## 13. Lifecycle Management

### 13.1 Daemon

```bash
# Start (backgrounded, writes PID)
cd ~/.claude/sentinel && python daemon.py &

# Graceful shutdown (flushes state, runs final GRPO)
kill -TERM $(cat ~/.claude/state/sentinel/daemon.pid)
```

### 13.2 Mode Terminal Connect/Disconnect

- **Connect:** reads Layer 0, calls Layer 1, initiates Layer 2 hydration async
- **Disconnect:** events queue in `events.jsonl`. GRPO in daemon keeps running. No data lost.
- **Reconnect:** reads queued events since last-seen timestamp, catches up.

### 13.3 Session Boundary

- **Working terminal SessionStart:** daemon creates new session context, resets turn counter, keeps SOS state (persistent across sessions)
- **Working terminal SessionEnd:** daemon runs final GRPO cycle for all modes, archives session data

## 14. V2 Extension Points

| Extension | Interface | V2 Work |
|-----------|-----------|---------|
| **RunPod neural GRPO scorer** | `evaluator` is a Callable — swap heuristic for API call to RunPod endpoint | Deploy scoring model, add `/score` endpoint |
| **Idea Reactor mode** | Add mode directory + CLAUDE.md, register with daemon | Write lateral-thinking analysis logic |
| **Arch Sentinel mode** | Same pattern as other modes | Write architecture review logic |
| **Token Economist mode** | Same pattern | Write context efficiency analysis logic |
| **Dashboard visualization** | Daemon exposes `/api/orbit/{mode}` — same as ForgeStream dashboard | Add D3 panels for orbit, convergence |
| **Cross-mode meta-SOS** | SOS where Π = mode activation weights | Define evaluator for mode portfolio optimization |
| **Karpathy autoresearch integration** | Context-space SOS already structured for this | Wire to ProofForge's KarpathyLoop pattern |

## 15. Naming Conventions

| Concept | Name |
|---------|------|
| The system | **Sentinel** |
| The daemon | **Sentinel Daemon** |
| A mode terminal | **Sentinel: {Mode Name}** (e.g., "Sentinel: Orange Team") |
| An injected insight | **[SENTINEL:{MODE}]** tag |
| Auto-injected content | **[SENTINEL:AUTO]** tag |
| The coach | **Sentinel Coach** |
| The optimization loop | **SOS-GRPO** |
| The pattern library | **Context Library** |
