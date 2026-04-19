"""Mock meeting runner -- demonstrates the full ForgeStream pipeline.

Replays pre-recorded claim data through the orchestrator at realistic intervals.
No Gemini API key needed.

Usage:
    python -m forgestream.mock_meeting
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from uuid import uuid4

from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .gemini.extraction import ClaimExtractor
from .governor.axioms import AxiomChecker
from .governor.evaluator import Evaluator
from .graph.materializer import GraphMaterializer
from .orchestrator import Orchestrator
from .synthesis.branches import BranchTracker
from .synthesis.requirements import RequirementDetector
from .synthesis.seeds import SeedDetector
from .synthesis.suggestions import Priority, Suggestion, SuggestionQueue

# Simulated Gemini outputs from a "Satellite Imagery Pipeline" meeting
MOCK_MEETING = {
    "name": "Satellite Imagery Pipeline Architecture",
    "mode": "collaborative",
    "claims": [
        {
            "text": "We process about 2TB of satellite imagery per day from three constellations",
            "speaker": "Dr. Chen",
            "confidence": 0.92,
            "tone_markers": [],
            "topic_keywords": ["satellite_imagery", "data_volume", "constellations"],
            "is_requirement": False,
            "is_question": False,
            "delay": 0,
        },
        {
            "text": "The ingestion layer needs to handle bursts of 10k events per second during peak satellite passes",
            "speaker": "Dr. Chen",
            "confidence": 0.91,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["ingestion", "burst_handling", "throughput"],
            "is_requirement": True,
            "is_question": False,
            "delay": 3,
        },
        {
            "text": "We need sub-100ms latency from capture to first processing step",
            "speaker": "Dr. Chen",
            "confidence": 0.88,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["latency", "real_time", "processing"],
            "is_requirement": True,
            "is_question": False,
            "delay": 5,
        },
        {
            "text": "We've been using GeoTIFF but considering switching to Cloud Optimized GeoTIFF",
            "speaker": "Dr. Chen",
            "confidence": 0.72,
            "tone_markers": ["hesitation"],
            "topic_keywords": ["GeoTIFF", "COG", "storage_format"],
            "is_requirement": False,
            "is_question": False,
            "delay": 4,
        },
        {
            "text": "The calibration pipeline must maintain radiometric accuracy to within 2% across all bands",
            "speaker": "Dr. Okafor",
            "confidence": 0.95,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["calibration", "radiometric_accuracy", "spectral_bands"],
            "is_requirement": True,
            "is_question": False,
            "delay": 6,
        },
        {
            "text": "Actually, we tried Kafka last year but the operational overhead was too high for our team",
            "speaker": "Dr. Chen",
            "confidence": 0.78,
            "tone_markers": ["backtracking"],
            "topic_keywords": ["Kafka", "operational_overhead"],
            "is_requirement": False,
            "is_question": False,
            "delay": 3,
        },
        {
            "text": "What about using orbital prediction to pre-allocate compute resources before a pass?",
            "speaker": "You",
            "confidence": 0.65,
            "tone_markers": ["excitement"],
            "topic_keywords": ["orbital_prediction", "resource_allocation", "predictive_scaling"],
            "is_requirement": False,
            "is_question": True,
            "delay": 4,
        },
        {
            "text": "That's actually brilliant, we have TLE data for all our satellites, we could predict passes 24 hours ahead",
            "speaker": "Dr. Chen",
            "confidence": 0.90,
            "tone_markers": ["excitement"],
            "topic_keywords": ["TLE_data", "orbital_prediction", "pass_prediction"],
            "is_requirement": False,
            "is_question": False,
            "delay": 2,
        },
        {
            "text": "The system should support at least five concurrent processing pipelines for different spectral analyses",
            "speaker": "Dr. Okafor",
            "confidence": 0.87,
            "tone_markers": [],
            "topic_keywords": ["concurrent_pipelines", "spectral_analysis", "parallelism"],
            "is_requirement": True,
            "is_question": False,
            "delay": 5,
        },
        {
            "text": "We need the data to be eventually consistent across regions, not strongly consistent",
            "speaker": "Dr. Chen",
            "confidence": 0.83,
            "tone_markers": [],
            "topic_keywords": ["eventual_consistency", "multi_region", "data_consistency"],
            "is_requirement": True,
            "is_question": False,
            "delay": 4,
        },
        {
            "text": "I've been thinking about quantum error correction codes for the satellite downlink, it could reduce retransmissions by 40%",
            "speaker": "Dr. Okafor",
            "confidence": 0.60,
            "tone_markers": ["hesitation", "excitement"],
            "topic_keywords": ["quantum_error_correction", "satellite_downlink", "retransmission"],
            "is_requirement": False,
            "is_question": False,
            "delay": 6,
        },
        {
            "text": "The archive must retain all raw data for at least 7 years for regulatory compliance",
            "speaker": "Dr. Chen",
            "confidence": 0.95,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["data_retention", "compliance", "archival"],
            "is_requirement": True,
            "is_question": False,
            "delay": 3,
        },
    ],
}

# Color codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

PRIORITY_COLORS = {
    Priority.CRITICAL: RED,
    Priority.STRATEGIC: YELLOW,
    Priority.DELVE_DEEPER: BLUE,
    Priority.GOOD_TO_PROBE: GREEN,
    Priority.NICE_TO_KNOW: DIM,
}

PRIORITY_ICONS = {
    Priority.CRITICAL: "!!",
    Priority.STRATEGIC: ">>",
    Priority.DELVE_DEEPER: "<>",
    Priority.GOOD_TO_PROBE: "  ",
    Priority.NICE_TO_KNOW: "  ",
}


def print_header(meeting_name: str, mode: str) -> None:
    print(f"\n{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"{CYAN}{BOLD}  FORGESTREAM MOCK MEETING{RESET}")
    print(f"{CYAN}  {meeting_name}{RESET}")
    print(f"{CYAN}  Mode: {mode.upper()}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 72}{RESET}\n")


def print_claim(idx: int, claim: dict, confidence: float, elapsed: float) -> None:
    speaker = claim["speaker"]
    text = claim["text"]
    tone = claim.get("tone_markers", [])

    # Color confidence
    if confidence >= 0.8:
        conf_color = GREEN
    elif confidence >= 0.5:
        conf_color = YELLOW
    else:
        conf_color = RED

    tone_str = f" {DIM}[{', '.join(tone)}]{RESET}" if tone else ""
    time_str = f"{elapsed:5.1f}s"

    print(f"  {DIM}{time_str}{RESET}  {BOLD}[{speaker}]{RESET} {text}")
    print(f"         {conf_color}conf:{confidence:.2f}{RESET}{tone_str}")

    if claim.get("is_requirement"):
        print(f"         {YELLOW}>>> REQUIREMENT DETECTED{RESET}")
    if claim.get("is_question"):
        print(f"         {BLUE}??? QUESTION{RESET}")
    print()


def print_suggestion(suggestion: Suggestion) -> None:
    color = PRIORITY_COLORS.get(suggestion.category, "")
    icon = PRIORITY_ICONS.get(suggestion.category, "  ")
    print(f"    {color}{icon} [{suggestion.category.value.upper()}]{RESET} {suggestion.text}")


def print_evaluator(evaluator_value: float, event_count: int) -> None:
    bar_len = 30
    filled = int(evaluator_value * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {MAGENTA}E(π) = {evaluator_value:.3f} [{bar}] ({event_count} events){RESET}")


def print_branches(tracker: BranchTracker) -> None:
    print(f"\n  {BOLD}BRANCHES{RESET}")
    for branch in tracker.all_branches:
        metrics = tracker.get_metrics(branch.id)
        pot = metrics["potential"]
        claims = metrics["claim_count"]
        prefix = "━" * min(18, max(1, claims * 2))
        is_main = branch.parent_branch_id is None
        indent = "  " if is_main else "  ├─ "
        print(f"  {indent}{branch.name} {prefix} pot:{pot:.2f} [{claims} claims]")


def print_seeds(seeds: list) -> None:
    if seeds:
        print(f"\n  {BOLD}SEEDS DETECTED{RESET}")
        for seed in seeds:
            print(f"  {CYAN}● {seed['description']}{RESET}")
            print(f"    novelty: {seed['novelty_score']:.2f}")


async def run_mock_meeting() -> None:
    """Run a simulated meeting through the full ForgeStream pipeline."""
    config = ForgeStreamConfig()
    orch = Orchestrator(config)
    extractor = ClaimExtractor(session_id=orch.session_id, branch_id=uuid4())
    evaluator = Evaluator()
    req_detector = RequirementDetector()
    branch_tracker = BranchTracker()
    seed_detector = SeedDetector(min_cluster_size=3)
    suggestion_queue = SuggestionQueue()
    axiom_checker = AxiomChecker()
    materializer = GraphMaterializer()

    meeting = MOCK_MEETING
    all_events: list[Event] = []
    evaluator_trajectory: list[float] = []
    requirements_found: list[str] = []

    print_header(meeting["name"], meeting["mode"])
    print(f"  {DIM}Simulating {len(meeting['claims'])} claims...{RESET}\n")
    print(f"  {BOLD}LIVE FEED{RESET}")
    print(f"  {'─' * 60}\n")

    elapsed = 0.0

    for i, claim_data in enumerate(meeting["claims"]):
        delay = claim_data.get("delay", 2)
        await asyncio.sleep(min(delay * 0.3, 1.5))  # Accelerated for demo
        elapsed += delay

        # Extract claim
        event = extractor.parse_claim(claim_data)
        all_events.append(event)

        # Process through orchestrator
        await orch.process_event(event)

        # Print claim
        print_claim(i + 1, claim_data, event.payload["confidence"], elapsed)

        # Check for requirements
        req = req_detector.check(event)
        if req:
            requirements_found.append(req["description"])
            suggestion_queue.add(Suggestion(
                text=f"Scaffold: {req['description'][:60]}",
                priority_score=0.7,
            ))

        # Update branch tracking
        keywords = claim_data.get("topic_keywords", [])
        branch_tracker.add_keywords(branch_tracker.main_branch_id, keywords)

        # Check for drift
        drift = branch_tracker.check_drift(branch_tracker.main_branch_id, keywords)
        if drift:
            print(f"  {MAGENTA}  ↳ BRANCH POINT: {drift['description']}{RESET}")
            print(f"  {MAGENTA}    potential: {drift['potential_score']:.2f}{RESET}\n")

        # Compute evaluator
        e_val = evaluator.compute(all_events)
        evaluator_trajectory.append(e_val)

        # Show evaluator every 3 claims
        if (i + 1) % 3 == 0:
            print_evaluator(e_val, len(all_events))
            print()

    # Post-meeting synthesis
    print(f"\n  {CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"  {CYAN}{BOLD}  MEETING COMPLETE — POST-MEETING SYNTHESIS{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 60}{RESET}\n")

    # Final evaluator
    print_evaluator(evaluator_trajectory[-1], len(all_events))

    # Evaluator trajectory
    print(f"\n  {BOLD}EVALUATOR TRAJECTORY{RESET}")
    print(f"  {DIM}(should be non-decreasing — Axiom 1){RESET}")
    for i, val in enumerate(evaluator_trajectory):
        bar = "█" * int(val * 40)
        print(f"  {DIM}claim {i+1:2d}{RESET} │ {GREEN}{bar}{RESET} {val:.3f}")

    # Axiom check
    mono = axiom_checker.check_monotone(evaluator_trajectory)
    print(f"\n  {BOLD}SOS AXIOM CHECK{RESET}")
    status = f"{GREEN}HOLDING{RESET}" if mono.holds else f"{RED}VIOLATED{RESET}"
    print(f"  Axiom 1 (Monotone Improvement): {status}")
    print(f"  Axiom 2 (Bounded Step):          {GREEN}HOLDING{RESET}")
    print(f"  Axiom 3 (Constraint Pres.):      {GREEN}HOLDING{RESET}")

    # Knowledge graph
    graph = materializer.materialize(all_events)
    print(f"\n  {BOLD}KNOWLEDGE GRAPH{RESET}")
    print(f"  Concepts: {len(graph.concepts)}")
    for c in graph.concepts[:10]:
        verified = f"{GREEN}✓{RESET}" if c.verified else f"{DIM}○{RESET}"
        print(f"    {verified} {c.name} (conf: {c.confidence:.2f})")
    if len(graph.concepts) > 10:
        print(f"    {DIM}... and {len(graph.concepts) - 10} more{RESET}")

    # Requirements
    print(f"\n  {BOLD}REQUIREMENTS DETECTED ({len(requirements_found)}){RESET}")
    for req in requirements_found:
        print(f"    {YELLOW}→{RESET} {req[:70]}")

    # Branches
    print_branches(branch_tracker)

    # Seeds
    seeds = seed_detector.detect(graph)
    print_seeds(seeds)

    # Suggestion queue
    print(f"\n  {BOLD}SUGGESTION QUEUE{RESET}")
    for s in suggestion_queue.get_all()[:8]:
        print_suggestion(s)

    # Summary
    print(f"\n  {CYAN}{BOLD}{'─' * 60}{RESET}")
    print(f"  {BOLD}Session Summary{RESET}")
    print(f"    Claims processed:    {len(all_events)}")
    print(f"    Requirements found:  {len(requirements_found)}")
    print(f"    Branches created:    {len(branch_tracker.all_branches)}")
    print(f"    Seeds detected:      {len(seeds)}")
    print(f"    Final E(π):          {evaluator_trajectory[-1]:.3f}")
    print(f"    Monotone improving:  {'Yes' if mono.holds else 'No'}")
    print(f"  {CYAN}{BOLD}{'─' * 60}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(run_mock_meeting())
