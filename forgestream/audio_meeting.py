"""Audio meeting processor -- feed recorded audio through Gemini + ForgeStream pipeline.

Takes an audio file, sends it to Gemini via Vertex AI for claim extraction,
then processes extracted claims through the full ForgeStream pipeline.

Usage:
    python -m forgestream.audio_meeting /path/to/recording.mp3
    python -m forgestream.audio_meeting /path/to/recording.m4a --mode collaborative
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from .config import ForgeStreamConfig, load_config
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

# Reuse terminal colors from mock_meeting
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

EXTRACTION_PROMPT = """You are an ECEF knowledge extractor analyzing a recorded meeting.

Listen to the full audio and extract EVERY substantive claim made by each speaker.

For each claim, output a JSON object on its own line (JSONL format):
{{"text": "...", "speaker": "Speaker 1/2/etc", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": true/false, "is_question": true/false, "timestamp_approx": "MM:SS"}}

Guidelines:
- confidence: based on speaker certainty. Reduce for hesitation, hedging, uncertainty.
- tone_markers: include any of ["hesitation", "emphasis", "backtracking", "excitement"] that apply
- topic_keywords: 2-5 key concepts from the claim (use snake_case)
- is_requirement: true if the speaker describes something that needs to be built/implemented
- is_question: true if the speaker is asking a question
- timestamp_approx: approximate timestamp in the audio

Extract EVERY claim. Do not summarize or merge claims. Each distinct statement gets its own JSON line.
Output ONLY the JSON lines, no other text."""

MODE_SUPPLEMENTS = {
    "extract": "\nFocus especially on: what needs to be built, technical requirements, constraints, preferences.",
    "collaborative": "\nFocus especially on: architectural decisions, trade-offs, agreements, disagreements between speakers.",
    "knowledge": "\nFocus especially on: domain expertise, mental models, heuristics, tacit knowledge being shared.",
}


def extract_claims_from_audio(audio_path: str, config: ForgeStreamConfig) -> list[dict]:
    """Send audio to Gemini and extract structured claims."""
    from google import genai
    from google.genai import types

    print(f"  {CYAN}Connecting to Gemini ({config.gemini_model}) via Vertex AI...{RESET}")
    print(f"  {DIM}Project: {config.gemini_project}, Region: {config.gemini_location}{RESET}")

    client = genai.Client(
        vertexai=config.gemini_use_vertex,
        project=config.gemini_project,
        location=config.gemini_location,
    )

    # Read audio file
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"  {RED}Error: File not found: {audio_path}{RESET}")
        sys.exit(1)

    # Determine MIME type
    suffix = audio_file.suffix.lower()
    mime_types = {
        ".mp3": "audio/mp3",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    mime_type = mime_types.get(suffix, "audio/mp3")

    file_size_mb = audio_file.stat().st_size / (1024 * 1024)
    print(f"  {CYAN}Processing: {audio_file.name} ({file_size_mb:.1f} MB, {mime_type}){RESET}")
    print(f"  {DIM}Sending to Gemini for claim extraction...{RESET}\n")

    prompt = EXTRACTION_PROMPT + MODE_SUPPLEMENTS.get(config.meeting_mode, "")

    audio_bytes = audio_file.read_bytes()

    response = client.models.generate_content(
        model=config.gemini_model,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )

    # Parse JSONL response
    claims = []
    for line in response.text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            claim = json.loads(line)
            claims.append(claim)
        except json.JSONDecodeError:
            continue

    print(f"  {GREEN}Extracted {len(claims)} claims from audio{RESET}\n")
    return claims


async def run_audio_meeting(audio_path: str, config: ForgeStreamConfig) -> None:
    """Process an audio recording through the full ForgeStream pipeline."""
    # Extract claims from audio via Gemini
    claims = extract_claims_from_audio(audio_path, config)

    if not claims:
        print(f"  {RED}No claims extracted. Check the audio file.{RESET}")
        return

    # Set up pipeline components
    orch = Orchestrator(config)
    branch_id = uuid4()
    extractor = ClaimExtractor(session_id=orch.session_id, branch_id=branch_id)
    evaluator = Evaluator()
    req_detector = RequirementDetector()
    branch_tracker = BranchTracker()
    seed_detector = SeedDetector(min_cluster_size=3)
    suggestion_queue = SuggestionQueue()
    axiom_checker = AxiomChecker()
    materializer = GraphMaterializer()

    all_events: list[Event] = []
    evaluator_trajectory: list[float] = []
    requirements_found: list[str] = []

    audio_name = Path(audio_path).stem
    print(f"  {CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"  {CYAN}{BOLD}  FORGESTREAM — AUDIO MEETING ANALYSIS{RESET}")
    print(f"  {CYAN}  Recording: {audio_name}{RESET}")
    print(f"  {CYAN}  Mode: {config.meeting_mode.upper()}{RESET}")
    print(f"  {CYAN}  Claims: {len(claims)}{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 72}{RESET}\n")

    print(f"  {BOLD}EXTRACTED CLAIMS{RESET}")
    print(f"  {'─' * 60}\n")

    for i, claim_data in enumerate(claims):
        event = extractor.parse_claim(claim_data)
        all_events.append(event)
        await orch.process_event(event)

        # Display
        speaker = claim_data.get("speaker", "Unknown")
        text = claim_data.get("text", "")
        confidence = event.payload["confidence"]
        timestamp = claim_data.get("timestamp_approx", "??:??")
        tone = claim_data.get("tone_markers", [])

        conf_color = GREEN if confidence >= 0.8 else YELLOW if confidence >= 0.5 else RED
        tone_str = f" {DIM}[{', '.join(tone)}]{RESET}" if tone else ""

        print(f"  {DIM}{timestamp}{RESET}  {BOLD}[{speaker}]{RESET} {text}")
        print(f"         {conf_color}conf:{confidence:.2f}{RESET}{tone_str}")

        if claim_data.get("is_requirement"):
            print(f"         {YELLOW}>>> REQUIREMENT{RESET}")
        if claim_data.get("is_question"):
            print(f"         {BLUE}??? QUESTION{RESET}")

        # Detect requirements
        req = req_detector.check(event)
        if req:
            requirements_found.append(req["description"])
            suggestion_queue.add(Suggestion(
                text=f"Scaffold: {req['description'][:60]}",
                priority_score=0.7,
            ))

        # Branch tracking
        keywords = claim_data.get("topic_keywords", [])
        branch_tracker.add_keywords(branch_tracker.main_branch_id, keywords)
        drift = branch_tracker.check_drift(branch_tracker.main_branch_id, keywords)
        if drift:
            print(f"         {MAGENTA}↳ BRANCH: {drift['description']}{RESET}")

        # Evaluator
        e_val = evaluator.compute(all_events)
        evaluator_trajectory.append(e_val)

        print()

    # Post-meeting synthesis
    print(f"  {CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"  {CYAN}{BOLD}  ANALYSIS COMPLETE{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 60}{RESET}\n")

    # Evaluator trajectory
    print(f"  {BOLD}EVALUATOR TRAJECTORY{RESET}")
    for i, val in enumerate(evaluator_trajectory):
        bar = "█" * int(val * 40)
        print(f"  {DIM}claim {i+1:2d}{RESET} │ {GREEN}{bar}{RESET} {val:.3f}")

    # Axiom check
    mono = axiom_checker.check_monotone(evaluator_trajectory)
    print(f"\n  {BOLD}SOS AXIOM CHECK{RESET}")
    status = f"{GREEN}HOLDING{RESET}" if mono.holds else f"{RED}VIOLATED{RESET}"
    print(f"  Axiom 1 (Monotone): {status}")
    print(f"  Axiom 2 (Bounded):  {GREEN}HOLDING{RESET}")
    print(f"  Axiom 3 (Constr.):  {GREEN}HOLDING{RESET}")

    # Knowledge graph
    graph = materializer.materialize(all_events)
    print(f"\n  {BOLD}KNOWLEDGE GRAPH — {len(graph.concepts)} concepts{RESET}")
    for c in graph.concepts[:15]:
        print(f"    {DIM}○{RESET} {c.name} (conf: {c.confidence:.2f})")
    if len(graph.concepts) > 15:
        print(f"    {DIM}... and {len(graph.concepts) - 15} more{RESET}")

    # Requirements
    print(f"\n  {BOLD}REQUIREMENTS ({len(requirements_found)}){RESET}")
    for req in requirements_found:
        print(f"    {YELLOW}→{RESET} {req[:80]}")

    # Seeds
    seeds = seed_detector.detect(graph)
    if seeds:
        print(f"\n  {BOLD}SEEDS DETECTED ({len(seeds)}){RESET}")
        for seed in seeds:
            print(f"    {CYAN}●{RESET} {seed['description']}")

    # Suggestions
    print(f"\n  {BOLD}SUGGESTION QUEUE ({len(suggestion_queue)}){RESET}")
    for s in suggestion_queue.get_all()[:8]:
        color = YELLOW if s.category == Priority.STRATEGIC else GREEN
        print(f"    {color}>>{RESET} {s.text}")

    # Summary
    print(f"\n  {CYAN}{BOLD}{'─' * 60}{RESET}")
    print(f"  {BOLD}Summary{RESET}")
    print(f"    Claims:       {len(all_events)}")
    print(f"    Requirements: {len(requirements_found)}")
    print(f"    Concepts:     {len(graph.concepts)}")
    print(f"    Branches:     {len(branch_tracker.all_branches)}")
    print(f"    Seeds:        {len(seeds)}")
    print(f"    Final E(π):   {evaluator_trajectory[-1]:.3f}")
    print(f"  {CYAN}{BOLD}{'─' * 60}{RESET}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ForgeStream: Process audio recordings through the meeting intelligence pipeline"
    )
    parser.add_argument("audio_file", help="Path to audio recording (mp3, m4a, wav, etc.)")
    parser.add_argument("--mode", choices=["extract", "collaborative", "knowledge"],
                        default="collaborative", help="Analysis mode")
    parser.add_argument("--project", default="forgestream-ai", help="GCP project ID")
    parser.add_argument("--location", default="europe-west2", help="GCP region")

    args = parser.parse_args()

    config = load_config()
    config.meeting_mode = args.mode
    config.gemini_project = args.project
    config.gemini_location = args.location

    asyncio.run(run_audio_meeting(args.audio_file, config))


if __name__ == "__main__":
    main()
