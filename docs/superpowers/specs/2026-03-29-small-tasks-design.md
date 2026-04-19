# ForgeStream Small Tasks — Consolidated Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Human Feedback → GRPO Wiring

**What:** The TUI `/end` command prompts the user for a 1-10 score. This score needs to be passed as `human_score` to `PostMeetingSynthesis.run()` via `GeminiLiveStream.stop()`.

**Implementation:** Add `human_score: float | None = None` parameter to `GeminiLiveStream.stop()`. Pass it through to `_run_post_meeting()`. The TUI `/end` handler calls `stream.stop(human_score=score/10.0)`.

**Files:** Modify `live_stream.py` (stop signature), modify `forgestream/tui/` (if `/end` handler exists, wire the score).

## 2. Branch Tree Visualization

**What:** Dashboard panel showing conversation branches — the tree structure from BranchTracker with branch potentials and topic drift markers.

**Implementation:** New JS module `branch-tree.js`. Fetches `/api/branches`. Renders a D3 tree layout (`d3.tree()`) with nodes = branches, edges = parent-child, node size = claim count, color = branch activity. Add to dashboard HTML grid.

**Files:** Create `dashboard/static/js/branch-tree.js`. Modify `server.py` (add panel to HTML). Modify `api.py` (flesh out `/api/branches` to return real branch data from events).

## 3. Seed Garden

**What:** Dashboard panel tracking cross-meeting seeds — promoted, dormant, archived status.

**Implementation:** New JS module `seed-garden.js`. New API endpoint `/api/seeds`. Shows seeds as cards with status badges. Seeds from SEED events, status tracked by whether they've been promoted to REQUIREMENT or gone stale.

**Files:** Create `dashboard/static/js/seed-garden.js`. Add `/api/seeds` endpoint to `api.py`. Modify `server.py` (add panel).

## 4. SOS Convergence Panel

**What:** Dashboard panel showing axiom status (3 indicators), trust region ε value, and predicted convergence meeting number.

**Implementation:** New JS module `sos-convergence.js`. Fetches `/api/evaluator` (already returns axiom status). Add `/api/trust-region` endpoint that returns current ε, consecutive improvements, total violations, and a simple convergence estimate (meetings until ε > 0.6 at current improvement rate).

**Files:** Create `dashboard/static/js/sos-convergence.js`. Add `/api/trust-region` endpoint. Modify `server.py`.

## 5. Contradiction Resolution Workflow

**What:** When contradictions are detected (CONTRADICTION events), create a structured resolution flow: present both sides, link to source claims, suggest probing questions, track resolution status.

**Implementation:** `ContradictionResolver` class that subscribes to EventBus, receives CONTRADICTION events, generates a resolution payload with both sides + suggested probing questions, and emits a SUGGESTION event with `priority: "high"` and `category: "contradiction_resolution"`. The dashboard shows contradictions in a dedicated section.

**Files:** Create `forgestream/synthesis/contradiction_resolver.py`. Add `/api/contradictions` endpoint. Create `dashboard/static/js/contradictions.js`. Wire into Orchestrator.
