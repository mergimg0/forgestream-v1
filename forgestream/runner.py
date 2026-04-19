"""ForgeStream runner -- process audio through TUI with live updates.

Usage:
    python -m forgestream.runner /path/to/audio.m4a
    python -m forgestream.runner /path/to/folder/ --mode collaborative
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

from .config import ForgeStreamConfig, load_config
from .events.schema import Event, EventType
from .gemini.extraction import ClaimExtractor
from .orchestrator import Orchestrator
from .synthesis.requirements import RequirementDetector


EXTRACTION_PROMPT = """You are an ECEF knowledge extractor analyzing a recorded conversation.

Listen to the full audio and extract EVERY substantive claim made by each speaker.

For each claim, output a JSON object on its own line (JSONL format):
{{"text": "...", "speaker": "Speaker 1/Speaker 2/etc", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": true/false, "is_question": true/false, "timestamp_approx": "MM:SS"}}

Extract EVERY substantive claim. Each distinct statement gets its own JSON line.
Output ONLY the JSON lines, no other text."""


def natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def extract_claims(audio_path: Path, config: ForgeStreamConfig) -> list[dict]:
    """Extract claims from audio via Gemini Vertex AI."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=config.gemini_use_vertex,
        project=config.gemini_project,
        location=config.gemini_location,
    )

    mime_types = {
        ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".wav": "audio/wav",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    }
    mime_type = mime_types.get(audio_path.suffix.lower(), "audio/mp4")
    audio_bytes = audio_path.read_bytes()

    print(f"Extracting claims from {audio_path.name}...")

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
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    else:
        return []

    claims = []
    if response and response.text:
        for line in response.text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                claims.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Extracted {len(claims)} claims. Launching TUI...")
    return claims


async def feed_claims_to_orchestrator(
    claims: list[dict],
    orchestrator: Orchestrator,
    branch_id,
    delay: float = 0.5,
) -> None:
    """Feed claims into the orchestrator with delays for visual effect."""
    await asyncio.sleep(1.5)  # Wait for TUI to mount

    extractor = ClaimExtractor(
        session_id=orchestrator.session_id,
        branch_id=branch_id,
    )
    req_detector = RequirementDetector()

    for claim_data in claims:
        event = extractor.parse_claim(claim_data)
        await orchestrator.process_event(event)

        # Check for requirements and create suggestion events
        req = req_detector.check(event)
        if req:
            suggestion_event = Event(
                event_type=EventType.SUGGESTION,
                session_id=orchestrator.session_id,
                branch_id=branch_id,
                author="synthesis",
                evaluator=0.0,
                payload={
                    "text": f"Scaffold: {req['description'][:60]}",
                    "priority": 0.7,
                },
            )
            await orchestrator.process_event(suggestion_event)

        await asyncio.sleep(delay)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ForgeStream: Audio meeting with live TUI")
    parser.add_argument("audio", help="Audio file or folder of audio files")
    parser.add_argument("--mode", choices=["extract", "collaborative", "knowledge"],
                        default="collaborative")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Delay between claims in seconds (default: 0.5)")
    args = parser.parse_args()

    config = load_config()
    config.meeting_mode = args.mode

    # Resolve audio files
    input_path = Path(args.audio)
    audio_extensions = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}

    if input_path.is_dir():
        audio_files = sorted(
            [f for f in input_path.iterdir() if f.suffix.lower() in audio_extensions],
            key=natural_sort_key,
        )
    elif input_path.is_file():
        audio_files = [input_path]
    else:
        print(f"Error: {args.audio} not found")
        sys.exit(1)

    # Extract claims from all audio files (batch phase)
    all_claims: list[dict] = []
    for i, audio_file in enumerate(audio_files):
        if i > 0:
            time.sleep(15)  # Rate limit cooldown
        claims = extract_claims(audio_file, config)
        all_claims.extend(claims)

    if not all_claims:
        print("No claims extracted.")
        sys.exit(1)

    print(f"\n{len(all_claims)} total claims. Starting TUI...\n")

    # Launch TUI with orchestrator
    from .tui.app import ForgeStreamApp

    orchestrator = Orchestrator(config)
    branch_id = uuid4()

    class LiveApp(ForgeStreamApp):
        """ForgeStreamApp that feeds claims after mounting."""

        async def on_mount(self) -> None:
            await super().on_mount()
            self.run_worker(self._feed_claims)

        async def _feed_claims(self) -> None:
            await feed_claims_to_orchestrator(
                all_claims, orchestrator, branch_id, delay=args.speed
            )
            feed = self.query_one("#feed")
            feed.write("[bold cyan]>>> ALL CLAIMS PROCESSED[/bold cyan]")

    app = LiveApp(orchestrator=orchestrator)
    app.run()


if __name__ == "__main__":
    main()
