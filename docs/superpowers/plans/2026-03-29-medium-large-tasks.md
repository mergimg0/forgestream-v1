# Medium + Large Tasks Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement 6 medium tasks (multi-objective GRPO, prompt tuning, meta-gleanings, meeting prep, expert profiles, MCP server) and 1 large task (proof obligation pipeline).

**Architecture:** Each task is an independent module. No cross-dependencies between tasks except that all wire into PostMeetingSynthesis or Orchestrator.

---

## Medium Task 1: Multi-Objective GRPO

**Files:**
- Modify: `forgestream/governor/improvement.py`
- Modify: `forgestream/post_meeting.py`
- Test: `tests/governor/test_multi_objective_grpo.py`

**Implementation:**

Add `tune_multi_objective()` to WeightTuner:

```python
def tune_multi_objective(
    self, current_weights: dict[str, float],
    events: list[Event],
    component_targets: dict[str, float],
) -> dict[str, float]:
    """Tune each weight toward its individual component target."""
    perturbations = self.generate_perturbations(current_weights, n=10)
    scores = []
    for p_weights in perturbations:
        evaluator = Evaluator(weights=p_weights)
        metrics = evaluator.compute_metrics(events)
        # Score = sum of (1 - |component_value - component_target|) per component
        score = 0.0
        metric_map = {
            "knowledge": metrics.knowledge_density,
            "verification": metrics.verification_rate,
            "scaffold": metrics.scaffold_success,
            "uptake": metrics.suggestion_uptake,
            "engagement": metrics.emotional_engagement,
        }
        for key, target in component_targets.items():
            actual = metric_map.get(key, 0.5)
            score += 1.0 - abs(actual - target)
        scores.append((score, p_weights))
    scores.sort(key=lambda x: x[0], reverse=True)
    best = scores[0][1]
    blended = {k: 0.7 * current_weights[k] + 0.3 * best[k] for k in current_weights}
    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}
```

**Tests:** test_multi_objective_improves_toward_targets, test_normalized_output, test_backward_compatible

---

## Medium Task 2: GRPO on Prompt Templates

**Files:**
- Create: `forgestream/gemini/prompt_tuner.py`
- Modify: `forgestream/live_stream.py`
- Modify: `forgestream/post_meeting.py`
- Test: `tests/gemini/test_prompt_tuner.py`

**Implementation:**

```python
# forgestream/gemini/prompt_tuner.py
@dataclass
class PromptParams:
    extraction_granularity: float = 0.5  # 0=coarse, 1=fine
    tone_sensitivity: float = 0.5        # 0=ignore, 1=heavy
    context_injection_minutes: float = 10.0

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "PromptParams": ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "PromptParams": ...

class PromptTuner:
    def apply_params(self, base_instruction: str, params: PromptParams) -> str:
        """Modify instruction based on params."""
        # granularity: add "Extract EVERY detail" vs "Focus on key claims"
        # tone_sensitivity: add "Pay attention to tone markers" vs remove tone instruction

    def tune(self, current: PromptParams, events: list[Event], human_score: float) -> PromptParams:
        """GRPO-perturb prompt params."""
```

**Tests:** test_apply_params_modifies_instruction, test_tune_returns_valid_params, test_save_load_roundtrip

---

## Medium Task 3: Meta-Gleanings

**Files:**
- Create: `forgestream/synthesis/meta_gleanings.py`
- Modify: `forgestream/post_meeting.py`
- Test: `tests/synthesis/test_meta_gleanings.py`

**Implementation:**

```python
# forgestream/synthesis/meta_gleanings.py
@dataclass
class MetaGleaning:
    category: str  # "topic_outcome", "evolution_pattern", "recurrence", "engagement_outcome"
    description: str
    confidence: float
    supporting_data: dict

class MetaGleaningEngine:
    def analyze(self, events: list[Event], prior_gleanings: list[dict] = None) -> list[MetaGleaning]:
        gleanings = []
        gleanings.extend(self._topic_outcome_mapping(events))
        gleanings.extend(self._discussion_evolution(events))
        gleanings.extend(self._engagement_outcome(events))
        return gleanings

    def _topic_outcome_mapping(self, events): ...
        # Which keywords co-occur with REQUIREMENT vs just CLAIM
    def _discussion_evolution(self, events): ...
        # Analyze claim density over time — explore→converge→decide pattern?
    def _engagement_outcome(self, events): ...
        # High engagement topics → more artifacts?
```

Persistence: `data/meta_gleanings.json` — accumulated across meetings.

**Tests:** test_topic_outcome_mapping, test_discussion_evolution_pattern, test_empty_events

---

## Medium Task 4: Meeting Preparation Mode

**Files:**
- Create: `forgestream/meeting_prep.py`
- Modify: `forgestream/__main__.py`
- Test: `tests/test_meeting_prep.py`

**Implementation:**

```python
class MeetingPrep:
    def __init__(self, data_dir: str = "data") -> None: ...

    def prepare(self, topic: str = "") -> str:
        """Generate a prep document for the next meeting."""
        graph = self._load_latest_graph()
        gaps = self._find_knowledge_gaps(graph)
        contradictions = self._find_unresolved_contradictions(graph)
        seeds = self._find_active_seeds()
        questions = self._generate_questions(gaps, contradictions, seeds, topic)
        return self._format_prep_doc(gaps, contradictions, seeds, questions)

    def _load_latest_graph(self) -> KnowledgeGraph: ...
    def _find_knowledge_gaps(self, graph) -> list[dict]: ...
        # Concepts with confidence < 0.5
    def _find_unresolved_contradictions(self, graph) -> list[dict]: ...
    def _find_active_seeds(self) -> list[dict]: ...
    def _generate_questions(self, gaps, contradictions, seeds, topic) -> list[str]: ...
```

CLI: `python -m forgestream prep --topic "quantum computing"`

**Tests:** test_prepare_generates_markdown, test_knowledge_gaps_detected, test_empty_graph

---

## Medium Task 5: Expert Profiles

**Files:**
- Create: `forgestream/profile/expert.py`
- Modify: `forgestream/post_meeting.py`
- Test: `tests/profile/test_expert.py`

**Implementation:**

```python
@dataclass
class ExpertProfile:
    speaker_id: str
    expertise_topics: dict[str, float] = field(default_factory=dict)  # topic → confidence
    communication_style: dict[str, float] = field(default_factory=dict)  # arousal, f0_var, energy
    rapport_with_user: float = 0.5
    meetings_count: int = 0
    total_claims: int = 0

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "ExpertProfile": ...

class ExpertProfileManager:
    def __init__(self, profiles_dir: str = "data/expert_profiles") -> None: ...

    def update_from_events(self, events: list[Event]) -> list[ExpertProfile]:
        """Update all speaker profiles from meeting events."""
        # Group CLAIM events by speaker → topic frequency
        # Group PROSODIC_FEATURE by speaker → communication style
        # Group RAPPORT_SCORE → rapport with user
        # EMA smoothing across meetings

    def load_profile(self, speaker_id: str) -> ExpertProfile: ...
    def save_profile(self, profile: ExpertProfile) -> None: ...
    def get_expert_for_topic(self, topic: str) -> ExpertProfile | None: ...
```

**Tests:** test_update_from_events, test_expertise_accumulates, test_get_expert_for_topic

---

## Medium Task 6: MCP Tool Integration

**Files:**
- Create: `forgestream/mcp_server.py`
- Modify: `forgestream/config.py`
- Modify: `forgestream/__main__.py`
- Test: `tests/test_mcp_server.py`

**Implementation:**

```python
# forgestream/mcp_server.py
"""MCP server exposing ForgeStream knowledge to Claude Code."""

from mcp.server import Server
from mcp.types import Tool, TextContent

def create_mcp_server(data_dir: str = "data") -> Server:
    server = Server("forgestream")

    @server.tool()
    async def forgestream_query_knowledge(topic: str) -> list[TextContent]:
        """Search the knowledge graph for concepts matching a topic."""

    @server.tool()
    async def forgestream_get_requirements() -> list[TextContent]:
        """List all detected requirements."""

    @server.tool()
    async def forgestream_get_seeds() -> list[TextContent]:
        """List all seeds with status."""

    @server.tool()
    async def forgestream_get_contradictions() -> list[TextContent]:
        """List unresolved contradictions."""

    @server.tool()
    async def forgestream_search_claims(query: str) -> list[TextContent]:
        """Full-text search across claims."""

    @server.tool()
    async def forgestream_get_expert(speaker: str) -> list[TextContent]:
        """Get expert profile for a speaker."""

    return server
```

Add `mcp` optional dependency and CLI subcommand.

**Tests:** test_query_knowledge, test_get_requirements, test_search_claims

---

## Large Task: Proof Obligation Pipeline

**Files:**
- Create: `forgestream/synthesis/proof_obligations.py`
- Create: `forgestream/synthesis/lean_stub.py`
- Modify: `forgestream/events/schema.py` — add PROOF_OBLIGATION
- Modify: `forgestream/orchestrator.py` — add attach_proof_detector()
- Modify: `forgestream/live_stream.py` — wire detector
- Modify: `forgestream/dashboard/api.py` — add /api/proof-obligations
- Modify: `forgestream/post_meeting.py` — export proof_obligations.json
- Test: `tests/synthesis/test_proof_obligations.py`
- Test: `tests/synthesis/test_lean_stub.py`

**Implementation per spec:** See docs/superpowers/specs/2026-03-29-proof-obligation-pipeline-design.md for full detail including:
- FORMALIZABLE_PATTERNS regex list
- MATH_KEYWORDS set
- LeanStubGenerator template
- ProofObligationDetector with EventBus subscription
- PROOF_OBLIGATION event payload
- Dashboard "Proof Queue" panel
- ProofForge export format

**Tests:**
- test_detects_formalizable_claim
- test_skips_non_formalizable
- test_generates_lean_stub
- test_stub_has_sorry
- test_exports_obligations_json
- test_dedup_same_conclusion
