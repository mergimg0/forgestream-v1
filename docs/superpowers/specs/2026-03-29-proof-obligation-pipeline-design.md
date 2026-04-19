# ForgeStream Proof Obligation Pipeline — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Depends on:** Knowledge graph, claim extraction, synthesis engine

---

## Overview

Automatically extracts claims from meetings that sound formalizable as mathematical theorems, generates preliminary Lean 4 theorem statements, and queues them for human review. This bridges ForgeStream (meeting intelligence) to ProofForge (Lean 4 proof generation with GRPO).

## Why This Matters

During expert meetings on formal mathematics, SOS theory, or algorithm design, speakers make claims like:
- "Any evaluator satisfying the three axioms converges"
- "The trust region epsilon is bounded by [0.15, 0.9]"
- "If engagement exceeds 0.7, verification rate doubles"

These are formalizable. Currently they stay as text claims in the knowledge graph. This pipeline detects them, translates them to Lean 4 stubs, and feeds them to ProofForge for proof attempt.

## Architecture

```
CLAIM events (from Gemini extraction)
    │
    ▼
ProofObligationDetector (subscribes to EventBus)
    ├── Is this claim formalizable?
    │   ├── Pattern matching: "for all", "if...then", "bounded by", "converges"
    │   ├── Keyword signals: mathematical terms, quantifiers, bounds
    │   └── Confidence threshold (>= 0.7 to attempt formalization)
    │
    ├── YES → Generate Lean 4 stub
    │   ├── Extract: hypothesis, conclusion, variables, types
    │   ├── Map to Lean 4 syntax: theorem, have, sorry
    │   └── Emit PROOF_OBLIGATION event
    │
    └── NO → Skip

PROOF_OBLIGATION events
    │
    ├── Stored in event log (PostgreSQL + Firestore)
    ├── Dashboard: "Proof Queue" panel showing pending obligations
    └── Export: proof_obligations.json for ProofForge consumption
```

## PROOF_OBLIGATION Event Payload

```python
{
    "claim_id": "uuid-of-source-claim",
    "claim_text": "Any evaluator satisfying the three axioms converges",
    "speaker": "Expert A",
    "confidence": 0.85,

    # Formalization
    "lean4_stub": "theorem evaluator_converges\n  (E : Evaluator)\n  (h1 : monotone E)\n  (h2 : bounded_step E)\n  (h3 : constraint_preservation E)\n  : converges E := by\n  sorry",
    "variables": ["E : Evaluator"],
    "hypotheses": ["monotone E", "bounded_step E", "constraint_preservation E"],
    "conclusion": "converges E",

    # Status
    "status": "pending",  # pending | reviewing | proved | failed | dismissed
    "formalization_confidence": 0.6,  # how confident are we in the Lean stub

    # ProofForge integration
    "proofforge_task_id": null,  # set when exported to ProofForge
}
```

## ProofObligationDetector

### Formalizability Detection

Pattern-based detection (not ML — fast, interpretable, no model dependency):

```python
FORMALIZABLE_PATTERNS = [
    r"(?:for all|for every|for any)\s+\w+",          # universal quantification
    r"(?:if|when|whenever)\s+.+(?:then|,)\s+",       # implication
    r"(?:bounded|bounded by|within|between)\s+",     # bounds
    r"(?:converges?|diverges?|monotone|increasing|decreasing)", # analysis
    r"(?:at most|at least|exactly|no more than)\s+", # quantitative bounds
    r"(?:implies|entails|guarantees|ensures)\s+",    # logical connectives
    r"(?:iff|if and only if)\s+",                    # biconditional
    r"(?:there exists|there is)\s+",                 # existential
]

MATH_KEYWORDS = {
    "theorem", "lemma", "proof", "axiom", "convergence", "bound",
    "epsilon", "delta", "sigma", "monotone", "continuous", "finite",
    "infinite", "set", "function", "mapping", "injective", "surjective",
    "bijective", "isomorphism", "homomorphism",
}
```

A claim is formalizable if:
1. It matches >= 1 pattern AND contains >= 1 math keyword
2. Its extraction confidence >= 0.7
3. It's not a question (`is_question: false`)

### Lean 4 Stub Generation

Template-based (not LLM — deterministic, auditable):

```python
def generate_lean_stub(claim_text, variables, hypotheses, conclusion):
    """Generate a Lean 4 theorem stub from extracted components."""
    var_decls = "\n  ".join(f"({v})" for v in variables)
    hyp_decls = "\n  ".join(f"(h{i+1} : {h})" for i, h in enumerate(hypotheses))

    return f"""theorem auto_obligation
  {var_decls}
  {hyp_decls}
  : {conclusion} := by
  sorry"""
```

The stub uses `sorry` — it's an obligation, not a proof. ProofForge's GRPO loop attempts to fill the proof.

### Component Extraction

Heuristic extraction from natural language:
- **Variables:** Nouns that follow "for all/any" or are capitalized mathematical objects
- **Hypotheses:** Clauses after "if/when/given" and before "then"
- **Conclusion:** Clause after "then" or the main predicate
- **Types:** Inferred from context keywords (e.g., "evaluator" → `Evaluator`, "function" → `α → β`)

This is intentionally imperfect — the human review step catches extraction errors. The goal is to surface obligations, not to produce correct Lean code on the first try.

## ProofForge Integration

Export format: `data/proof_obligations.json` — array of obligation payloads. ProofForge reads this file to discover new proof tasks.

```json
[
    {
        "claim_id": "...",
        "lean4_stub": "theorem ...",
        "status": "pending",
        "source_meeting": "2026-03-29-quantum-meeting",
        "confidence": 0.85
    }
]
```

ProofForge writes back: updates `status` to "proved" or "failed" with proof text or error.

## File Structure

| File | Responsibility | Est. Lines |
|------|---------------|-----------|
| `forgestream/synthesis/proof_obligations.py` | `ProofObligationDetector` — detection + stub generation | ~200 |
| `forgestream/synthesis/lean_stub.py` | `LeanStubGenerator` — template-based Lean 4 generation | ~100 |
| `tests/synthesis/test_proof_obligations.py` | Detector + generator tests | ~120 |
| `tests/synthesis/test_lean_stub.py` | Lean stub generation tests | ~60 |

## Modified Files

| File | Changes |
|------|---------|
| `forgestream/events/schema.py` | Add `PROOF_OBLIGATION` EventType |
| `forgestream/orchestrator.py` | Add `attach_proof_detector()` |
| `forgestream/live_stream.py` | Wire ProofObligationDetector when math-heavy mode |
| `forgestream/dashboard/api.py` | Add `/api/proof-obligations` endpoint |
| `forgestream/dashboard/server.py` | Add "Proof Queue" panel to HTML |
| `forgestream/post_meeting.py` | Export proof_obligations.json in run() |

## Edge Cases

- Claims that SOUND formal but aren't ("the system is bounded" — by what? vague) → low formalization_confidence, still queued but flagged
- Multiple claims that formalize to the same theorem → dedup by conclusion text
- Claims in non-English → skip (pattern matching is English-only for now)
- Very long claims → truncate to first sentence for formalization attempt
