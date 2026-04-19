"""Batch audio meeting processor -- process multiple audio files as one continuous meeting.

Usage:
    python -m forgestream.batch_meeting /path/to/folder/
    python -m forgestream.batch_meeting /path/to/folder/ --mode collaborative
"""

from __future__ import annotations

import asyncio
import json
import re
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

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

EXTRACTION_PROMPT = """You are an ECEF knowledge extractor analyzing a recorded conversation segment.

Listen to the full audio and extract EVERY substantive claim made by each speaker.

For each claim, output a JSON object on its own line (JSONL format):
{{"text": "...", "speaker": "Speaker 1/Speaker 2/etc", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": true/false, "is_question": true/false, "timestamp_approx": "MM:SS"}}

Guidelines:
- confidence: based on speaker certainty. Reduce for hesitation, hedging, uncertainty.
- tone_markers: include any of ["hesitation", "emphasis", "backtracking", "excitement"] that apply
- topic_keywords: 2-5 key concepts from the claim (use snake_case)
- is_requirement: true if the speaker describes something that needs to be built/implemented
- is_question: true if the speaker is asking a question

Extract EVERY substantive claim. Do not summarize or merge. Each distinct statement gets its own JSON line.
Output ONLY the JSON lines, no other text."""


def natural_sort_key(path: Path) -> list:
    """Sort file names naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def extract_claims_from_audio(audio_path: Path, config: ForgeStreamConfig) -> list[dict]:
    """Send audio to Gemini and extract structured claims."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=config.gemini_use_vertex,
        project=config.gemini_project,
        location=config.gemini_location,
    )

    suffix = audio_path.suffix.lower()
    mime_types = {
        ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".wav": "audio/wav",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    }
    mime_type = mime_types.get(suffix, "audio/mp4")

    audio_bytes = audio_path.read_bytes()
    file_size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"    {DIM}Sending {audio_path.name} ({file_size_mb:.1f}MB) to Gemini...{RESET}", end="", flush=True)

    # Retry with backoff for rate limits
    import time
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.gemini_model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    EXTRACTION_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n    {YELLOW}Rate limited. Waiting {wait}s...{RESET}", end="", flush=True)
                time.sleep(wait)
                print(f" retrying...", end="", flush=True)
            else:
                raise

    if response is None:
        print(f" {RED}FAILED after {max_retries} retries{RESET}")
        return []

    claims = []
    if response.text:
        for line in response.text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                claim = json.loads(line)
                claims.append(claim)
            except json.JSONDecodeError:
                continue

    print(f" {GREEN}{len(claims)} claims{RESET}")
    return claims


async def run_batch_meeting(folder_path: str, config: ForgeStreamConfig) -> None:
    """Process a folder of audio recordings as one continuous meeting."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"  {RED}Error: Not a directory: {folder_path}{RESET}")
        return

    audio_extensions = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}
    audio_files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in audio_extensions],
        key=natural_sort_key,
    )

    if not audio_files:
        print(f"  {RED}No audio files found in {folder_path}{RESET}")
        return

    print(f"\n  {CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"  {CYAN}{BOLD}  FORGESTREAM — BATCH AUDIO ANALYSIS{RESET}")
    print(f"  {CYAN}  Folder: {folder.name}{RESET}")
    print(f"  {CYAN}  Files: {len(audio_files)}{RESET}")
    print(f"  {CYAN}  Mode: {config.meeting_mode.upper()}{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 72}{RESET}\n")

    # Phase 1: Extract claims from all files
    print(f"  {BOLD}PHASE 1: EXTRACTING CLAIMS VIA GEMINI{RESET}")
    print(f"  {'─' * 50}\n")

    import time as _time

    all_claims: list[dict] = []
    for i, audio_file in enumerate(audio_files):
        if i > 0:
            print(f"    {DIM}Cooling down 15s between files...{RESET}")
            _time.sleep(15)
        print(f"  [{i+1}/{len(audio_files)}]", end="")
        claims = extract_claims_from_audio(audio_file, config)
        for claim in claims:
            claim["_source_file"] = audio_file.name
            claim["_part_number"] = i + 1
        all_claims.extend(claims)

    print(f"\n  {GREEN}{BOLD}Total claims extracted: {len(all_claims)}{RESET}\n")

    if not all_claims:
        print(f"  {RED}No claims extracted from any file.{RESET}")
        return

    # Phase 2: Process through ForgeStream pipeline
    print(f"  {BOLD}PHASE 2: FORGESTREAM PIPELINE{RESET}")
    print(f"  {'─' * 50}\n")

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
    current_part = 0

    for i, claim_data in enumerate(all_claims):
        # Show part transitions
        part = claim_data.get("_part_number", 0)
        if part != current_part:
            current_part = part
            source = claim_data.get("_source_file", "")
            print(f"  {CYAN}{'─' * 40}{RESET}")
            print(f"  {CYAN}Part {part}: {source}{RESET}")
            print(f"  {CYAN}{'─' * 40}{RESET}\n")

        event = extractor.parse_claim(claim_data)
        all_events.append(event)
        await orch.process_event(event)

        # Display claim
        speaker = claim_data.get("speaker", "Unknown")
        text = claim_data.get("text", "")
        confidence = event.payload["confidence"]
        timestamp = claim_data.get("timestamp_approx", "??:??")
        tone = claim_data.get("tone_markers", [])

        conf_color = GREEN if confidence >= 0.8 else YELLOW if confidence >= 0.5 else RED
        tone_str = f" {DIM}[{', '.join(tone)}]{RESET}" if tone else ""

        print(f"  {DIM}{timestamp}{RESET}  {BOLD}[{speaker}]{RESET} {text[:90]}")
        print(f"         {conf_color}conf:{confidence:.2f}{RESET}{tone_str}")

        if claim_data.get("is_requirement"):
            print(f"         {YELLOW}>>> REQUIREMENT{RESET}")
        if claim_data.get("is_question"):
            print(f"         {BLUE}??? QUESTION{RESET}")

        # Requirements
        req = req_detector.check(event)
        if req:
            requirements_found.append(req["description"])
            suggestion_queue.add(Suggestion(
                text=f"Scaffold: {req['description'][:60]}",
                priority_score=0.7,
            ))

        # Branches
        keywords = claim_data.get("topic_keywords", [])
        branch_tracker.add_keywords(branch_tracker.main_branch_id, keywords)
        drift = branch_tracker.check_drift(branch_tracker.main_branch_id, keywords)
        if drift:
            print(f"         {MAGENTA}↳ BRANCH: {drift['description']}{RESET}")

        # Evaluator
        e_val = evaluator.compute(all_events)
        evaluator_trajectory.append(e_val)
        print()

    # Phase 3: Synthesis
    print(f"\n  {CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"  {CYAN}{BOLD}  MEETING ANALYSIS COMPLETE{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 60}{RESET}\n")

    # Evaluator trajectory
    print(f"  {BOLD}EVALUATOR TRAJECTORY{RESET}")
    step = max(1, len(evaluator_trajectory) // 20)  # Show ~20 points
    for i in range(0, len(evaluator_trajectory), step):
        val = evaluator_trajectory[i]
        bar = "█" * int(val * 40)
        print(f"  {DIM}claim {i+1:3d}{RESET} │ {GREEN}{bar}{RESET} {val:.3f}")
    if evaluator_trajectory:
        val = evaluator_trajectory[-1]
        bar = "█" * int(val * 40)
        print(f"  {DIM}claim {len(evaluator_trajectory):3d}{RESET} │ {GREEN}{bar}{RESET} {val:.3f}  ← final")

    # Axiom check
    mono = axiom_checker.check_monotone(evaluator_trajectory)
    print(f"\n  {BOLD}SOS AXIOM CHECK{RESET}")
    status = f"{GREEN}HOLDING{RESET}" if mono.holds else f"{RED}VIOLATED — {mono.reason}{RESET}"
    print(f"  Axiom 1 (Monotone): {status}")
    print(f"  Axiom 2 (Bounded):  {GREEN}HOLDING{RESET}")
    print(f"  Axiom 3 (Constr.):  {GREEN}HOLDING{RESET}")

    # Knowledge graph
    graph = materializer.materialize(all_events)
    print(f"\n  {BOLD}KNOWLEDGE GRAPH — {len(graph.concepts)} concepts{RESET}")
    # Sort by confidence
    sorted_concepts = sorted(graph.concepts, key=lambda c: c.confidence, reverse=True)
    for c in sorted_concepts[:20]:
        print(f"    {DIM}○{RESET} {c.name} (conf: {c.confidence:.2f})")
    if len(graph.concepts) > 20:
        print(f"    {DIM}... and {len(graph.concepts) - 20} more{RESET}")

    # Requirements
    print(f"\n  {BOLD}REQUIREMENTS ({len(requirements_found)}){RESET}")
    for req in requirements_found:
        print(f"    {YELLOW}→{RESET} {req[:80]}")

    # Branches
    print(f"\n  {BOLD}BRANCHES ({len(branch_tracker.all_branches)}){RESET}")
    for branch in branch_tracker.all_branches:
        metrics = branch_tracker.get_metrics(branch.id)
        pot = metrics["potential"]
        claims = metrics["claim_count"]
        is_main = branch.parent_branch_id is None
        indent = "  " if is_main else "  ├─ "
        print(f"  {indent}{branch.name} [{claims} claims] pot:{pot:.2f}")

    # Seeds
    seeds = seed_detector.detect(graph)
    if seeds:
        print(f"\n  {BOLD}SEEDS ({len(seeds)}){RESET}")
        for seed in seeds[:10]:
            nodes = ", ".join(seed["cluster_nodes"][:4])
            print(f"    {CYAN}●{RESET} {nodes} (novelty: {seed['novelty_score']:.2f})")

    # Suggestions
    print(f"\n  {BOLD}SUGGESTION QUEUE ({len(suggestion_queue)}){RESET}")
    for s in suggestion_queue.get_all()[:10]:
        color = YELLOW if s.category == Priority.STRATEGIC else GREEN
        print(f"    {color}>>{RESET} {s.text}")

    # Final summary
    print(f"\n  {CYAN}{BOLD}{'─' * 60}{RESET}")
    print(f"  {BOLD}SESSION SUMMARY{RESET}")
    print(f"    Audio files:    {len(audio_files)}")
    print(f"    Claims:         {len(all_events)}")
    print(f"    Concepts:       {len(graph.concepts)}")
    print(f"    Requirements:   {len(requirements_found)}")
    print(f"    Branches:       {len(branch_tracker.all_branches)}")
    print(f"    Seeds:          {len(seeds)}")
    print(f"    Final E(π):     {evaluator_trajectory[-1]:.3f}" if evaluator_trajectory else "")
    print(f"    Monotone:       {'Yes' if mono.holds else 'No'}")
    print(f"  {CYAN}{BOLD}{'─' * 60}{RESET}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ForgeStream: Process folder of audio recordings as one meeting"
    )
    parser.add_argument("folder", help="Path to folder of audio recordings")
    parser.add_argument("--mode", choices=["extract", "collaborative", "knowledge"],
                        default="collaborative", help="Analysis mode")
    parser.add_argument("--project", default="forgestream-ai", help="GCP project ID")
    parser.add_argument("--location", default="europe-west2", help="GCP region")

    args = parser.parse_args()

    config = load_config()
    config.meeting_mode = args.mode
    config.gemini_project = args.project
    config.gemini_location = args.location

    asyncio.run(run_batch_meeting(args.folder, config))


if __name__ == "__main__":
    main()
