# ForgeStream Medium Tasks — Consolidated Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Multi-Objective GRPO

**What:** Instead of one composite E(π) score driving GRPO, tune each evaluator component weight independently. Each component (knowledge, verification, scaffold, uptake, engagement) gets its own perturbation trajectory, scored against the human feedback signal for that specific dimension.

**Architecture:** Extend `WeightTuner.tune()` to accept a `component_scores: dict[str, float]` that maps each weight key to its individual target. After each meeting, compute per-component targets: knowledge target from claim density, verification target from findings/claims ratio, engagement target from rapport composite. GRPO perturbs each weight independently within a ±0.1 window (not global normalization) then normalizes at the end.

**Files:**
- Modify: `forgestream/governor/improvement.py` — add `tune_multi_objective()` method
- Modify: `forgestream/post_meeting.py` — compute per-component targets, call multi-objective tune
- Test: `tests/governor/test_multi_objective_grpo.py`

## 2. GRPO on Prompt Templates

**What:** The Gemini system instructions (MODE_INSTRUCTIONS in live_stream.py) should evolve across meetings. Track which prompt variant produces the best claim extraction quality. GRPO perturbs prompt parameters (instruction emphasis, extraction granularity, tone sensitivity) and selects the best-performing variant.

**Architecture:** `PromptTuner` class manages a set of prompt parameters (not the raw text — parameters that modify the template). Parameters: `extraction_granularity` (0.0=coarse → 1.0=fine), `tone_sensitivity` (0.0=ignore tone → 1.0=heavily weight tone markers), `context_injection_frequency_minutes` (5-30). Each meeting, perturb parameters, generate the prompt from template + parameters, and score by claim quality metrics post-meeting.

**Files:**
- Create: `forgestream/gemini/prompt_tuner.py` — PromptTuner + prompt template with parameter slots
- Modify: `forgestream/live_stream.py` — load tuned prompt params at startup, apply to MODE_INSTRUCTIONS
- Modify: `forgestream/post_meeting.py` — tune prompt params in run()
- Test: `tests/gemini/test_prompt_tuner.py`

## 3. Meta-Gleanings

**What:** Second-order observations about meetings themselves — not "what was discussed" but "how the discussion evolved," "what patterns recur across meetings," "which topics always lead to requirements vs stay conceptual."

**Architecture:** `MetaGleaningEngine` runs post-meeting. Analyzes the event log for structural patterns:
- Topic-to-outcome mapping: which keywords co-occur with REQUIREMENT events vs just CLAIM
- Discussion evolution: does the meeting follow explore→converge→decide or meander?
- Cross-meeting recurrence: topics that appear in 3+ meetings
- Engagement-outcome correlation: do high-engagement topics produce more artifacts?

Outputs a `meta_gleanings` section in the meeting report.

**Files:**
- Create: `forgestream/synthesis/meta_gleanings.py` — MetaGleaningEngine
- Modify: `forgestream/post_meeting.py` — call in run(), add to report
- Test: `tests/synthesis/test_meta_gleanings.py`

## 4. Meeting Preparation Mode

**What:** Before a meeting, ForgeStream reviews the knowledge graph for relevant prior knowledge, prepares suggested questions based on knowledge gaps, and pre-loads context.

**Architecture:** `MeetingPrep` class that:
1. Loads the knowledge graph from the most recent meeting's events (stored in Firestore/PostgreSQL)
2. Identifies knowledge gaps: concepts with low confidence, unresolved contradictions, dormant seeds
3. Generates suggested probing questions based on gaps
4. Outputs a prep document (markdown) the user can review before the meeting

Triggered via CLI: `python -m forgestream prep [--topic "quantum computing"]`

**Files:**
- Create: `forgestream/meeting_prep.py` — MeetingPrep class
- Modify: `forgestream/__main__.py` — add `prep` subcommand
- Test: `tests/test_meeting_prep.py`

## 5. Expert Profiles

**What:** Track what each speaker (expert) knows across multiple meetings. Build per-speaker expertise profiles: topics they're authoritative on, their communication style, when they're most engaged.

**Architecture:** `ExpertProfile` dataclass stored per speaker in `data/expert_profiles/`. Updated post-meeting from CLAIM events (topic_keywords by speaker), PROSODIC_FEATURE events (communication style per speaker), and RAPPORT_SCORE events (rapport with user). Uses EMA smoothing across meetings.

**Files:**
- Create: `forgestream/profile/expert.py` — ExpertProfile dataclass + ExpertProfileManager
- Modify: `forgestream/post_meeting.py` — update expert profiles in run()
- Test: `tests/profile/test_expert.py`

## 6. MCP Tool Integration

**What:** Expose ForgeStream's knowledge graph, event log, and agent system as MCP tools. This lets Claude Code (or any MCP client) query meeting knowledge during coding sessions.

**Architecture:** MCP server using `mcp` Python SDK. Tools:
- `forgestream_query_knowledge(topic)` — search the knowledge graph for concepts matching a topic
- `forgestream_get_requirements()` — list all detected requirements with status
- `forgestream_get_seeds()` — list all seeds with promotion status
- `forgestream_get_contradictions()` — list unresolved contradictions
- `forgestream_search_claims(query)` — full-text search across all claims
- `forgestream_get_expert(speaker)` — get expert profile for a speaker

Server runs alongside the dashboard on a configurable port. Reads from Firestore/PostgreSQL (same data as dashboard API).

**Files:**
- Create: `forgestream/mcp_server.py` — MCP server with 6 tools
- Modify: `forgestream/config.py` — add `mcp_port` config
- Modify: `forgestream/__main__.py` — add `mcp` subcommand
- Test: `tests/test_mcp_server.py`
