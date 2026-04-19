"""Live polling transcriber — captures audio, sends chunks to Gemini batch API.

Workaround for when the Gemini Live API key is unavailable.
Records from mic, sends 30-second audio chunks to Vertex AI for
transcription + claim extraction, writes events to live_events.jsonl.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import wave
from pathlib import Path
from uuid import uuid4

from .config import ForgeStreamConfig, load_config
from .events.schema import Event, EventType
from .gemini.extraction import ClaimExtractor
from .live_copilot import AudioRecorder, LiveEventSink
from .orchestrator import Orchestrator
from .synthesis.requirements import RequirementDetector

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a meeting transcription and analysis system.

Listen to this audio segment and:
1. Transcribe what each speaker says
2. Extract every substantive claim as a JSON object

Output ONLY JSON lines (JSONL), one per claim:
{{"text": "...", "speaker": "Speaker 1/Speaker 2", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": false, "is_question": false}}

Extract EVERY distinct statement. Be thorough."""


def _resolve_credentials_path() -> str:
    """Resolve GOOGLE_APPLICATION_CREDENTIALS to an absolute path."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if raw and not os.path.isabs(raw):
        # Already set but relative — leave it (caller's responsibility)
        return raw
    if not raw:
        # Default: resolve relative to this package's project root
        project_root = Path(__file__).resolve().parent.parent
        cred_path = project_root / ".secrets" / "service-account.json"
        return str(cred_path)
    return raw


# Buffer limits
MAX_BUFFER_BYTES = 30 * 16000 * 2  # 30 seconds of 16kHz 16-bit mono (~960 KB)

# Backoff constants
MAX_CONSECUTIVE_FAILURES = 5
BACKOFF_BASE_SECONDS = 5.0


def extract_claims_from_bytes(
    audio_bytes: bytes,
    config: ForgeStreamConfig,
) -> list[dict]:
    """Send audio bytes to Gemini Vertex AI and extract claims."""
    cred_path = _resolve_credentials_path()
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred_path)

    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=config.gemini_project,
        location=config.gemini_location,
    )

    try:
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                EXTRACTION_PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception as e:
        logger.warning("Gemini batch call failed: %s", e)
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
    return claims


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM int16 mono to WAV bytes in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def run_polling_session(
    config: ForgeStreamConfig,
    device: int | None = None,
    chunk_interval: float = 30.0,
) -> None:
    """Main loop: capture audio, poll Gemini every chunk_interval seconds."""
    from .audio.microphone import MicrophoneSource

    session_id = uuid4()
    branch_id = uuid4()
    orch = Orchestrator(config)
    orch.attach_synthesis_engine()

    # Connect PostgreSQL
    # ot-ctx-pg-leak-005: track conn for cleanup on exit
    _pg_conn = None
    try:
        import psycopg
        from .events.store import EventStore

        _pg_conn = await psycopg.AsyncConnection.connect(config.postgres_dsn)
        orch.store = EventStore(_pg_conn)
        print("  PostgreSQL: connected")
    except Exception as e:
        print(f"  PostgreSQL: unavailable ({e})")

    # Recording — scoped to config.data_dir
    from datetime import datetime
    from pathlib import Path

    data_dir = Path(config.data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = config.meeting_name.replace(" ", "_") if config.meeting_name else "meeting"
    recordings_dir = data_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(recordings_dir / f"{tag}_{ts}.wav")
    recorder = AudioRecorder(path=wav_path)

    # Live event sink
    sink = LiveEventSink(path=str(data_dir / "live_events.jsonl"))
    sink.subscribe(orch.event_bus)

    # Emotion pipeline
    emotion_extractor = None
    audio_buffer = None
    if config.emotion_enabled:
        from .emotion.buffer import AudioRingBuffer
        from .emotion.extractor import EmotionExtractor

        audio_buffer = AudioRingBuffer(capacity_seconds=config.emotion_buffer_seconds)
        emotion_extractor = EmotionExtractor(
            orchestrator=orch,
            audio_buffer=audio_buffer,
            branch_id=branch_id,
            window_seconds=config.emotion_window_seconds,
            stride_seconds=config.emotion_stride_seconds,
        )
        orch.attach_emotion_correlator()
        orch.attach_dynamics_engine()
        if config.rapport_enabled:
            from .post_meeting import PostMeetingSynthesis

            pms = PostMeetingSynthesis(config=config, data_dir=config.data_dir)
            meeting_count = pms.load_meeting_count()
            rapport_weights = pms.load_rapport_weights(meeting_count)
            orch.attach_rapport_engine(
                meeting_count=meeting_count,
                rapport_weights=rapport_weights,
            )

    # Mic capture
    mic = MicrophoneSource(device=device)
    await mic.start()
    print(f"  Audio: device {device or 'default'}")
    print(f"  Recording: {wav_path}")
    print(f"  Live events: {data_dir / 'live_events.jsonl'}")
    print(f"  Polling interval: {chunk_interval}s")
    print(f"  Emotion: {'enabled' if config.emotion_enabled else 'disabled'}")
    print("\n  Listening... Press Ctrl+C to stop.\n")

    extractor = ClaimExtractor(session_id=session_id, branch_id=branch_id)
    req_detector = RequirementDetector()
    pcm_buffer = bytearray()
    buffer_lock = asyncio.Lock()
    chunk_count = 0
    consecutive_failures = 0
    stop_event = asyncio.Event()

    import signal

    def on_signal():
        print("\n  Stopping...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    async def capture_loop():
        """Read mic chunks into buffer + recorder."""
        nonlocal chunk_count
        async for chunk in mic.chunks():
            if stop_event.is_set():
                break
            async with buffer_lock:
                pcm_buffer.extend(chunk)
                # Cap buffer to prevent OOM if Gemini calls are slow
                if len(pcm_buffer) > MAX_BUFFER_BYTES:
                    overflow = len(pcm_buffer) - MAX_BUFFER_BYTES
                    del pcm_buffer[:overflow]
                    logger.warning("Buffer overflow: dropped %d bytes of oldest audio", overflow)
            recorder.write_chunk(chunk)
            chunk_count += 1
            # Feed emotion pipeline
            if audio_buffer is not None:
                idx = audio_buffer.write_chunk(chunk)
                if emotion_extractor is not None:
                    await emotion_extractor.process_chunk(chunk, idx)

    async def poll_loop():
        """Send accumulated audio to Gemini every chunk_interval seconds."""
        nonlocal consecutive_failures
        await asyncio.sleep(chunk_interval)  # Wait for first chunk
        while not stop_event.is_set():
            # Atomic buffer swap under lock — no chunks lost
            async with buffer_lock:
                if len(pcm_buffer) == 0:
                    await asyncio.sleep(chunk_interval)
                    continue
                audio_data = bytes(pcm_buffer)
                pcm_buffer.clear()

            wav_data = pcm_to_wav_bytes(audio_data)
            duration = len(audio_data) / (16000 * 2)
            print(f"  >> Sending {duration:.0f}s audio to Gemini...")

            # Run extraction in thread to not block
            claims = await asyncio.to_thread(
                extract_claims_from_bytes, wav_data, config
            )

            if claims:
                consecutive_failures = 0
                print(f"  << {len(claims)} claims extracted")
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    backoff = min(BACKOFF_BASE_SECONDS * (2 ** (consecutive_failures - MAX_CONSECUTIVE_FAILURES)), 120.0)
                    logger.warning(
                        "Gemini: %d consecutive failures, backing off %.0fs",
                        consecutive_failures, backoff,
                    )
                    print(f"  !! {consecutive_failures} consecutive Gemini failures — backing off {backoff:.0f}s")
                    await asyncio.sleep(backoff)

            for claim_data in claims:
                event = extractor.parse_claim(claim_data)
                await orch.process_event(event)
                req = req_detector.check(event)
                if req:
                    suggestion = Event(
                        event_type=EventType.SUGGESTION,
                        session_id=session_id,
                        branch_id=branch_id,
                        author="synthesis",
                        evaluator=0.0,
                        payload={
                            "text": f"Scaffold: {req['description'][:60]}",
                            "priority": 0.7,
                        },
                    )
                    await orch.process_event(suggestion)

            await asyncio.sleep(chunk_interval)

    # Run both loops
    capture_task = asyncio.create_task(capture_loop())
    poll_task = asyncio.create_task(poll_loop())

    await stop_event.wait()

    capture_task.cancel()
    poll_task.cancel()
    await mic.stop()
    sink.close()

    # ot-ctx-pg-leak-005: close PostgreSQL connection on exit
    if _pg_conn is not None:
        try:
            await _pg_conn.close()
        except Exception:
            pass

    duration = recorder.duration_seconds
    print(f"\n  Recording: {wav_path} ({duration:.0f}s)")
    print(f"  Events: {sink.event_count}")
    print(f"  Chunks sent to Gemini: {chunk_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ForgeStream: Live polling transcriber")
    parser.add_argument("--name", default="", help="Meeting name")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds between Gemini calls (default: 30)",
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", _resolve_credentials_path()
    )

    config = load_config()
    config.meeting_name = args.name or "meeting"
    config.meeting_mode = "knowledge"

    asyncio.run(
        run_polling_session(
            config=config,
            device=args.device,
            chunk_interval=args.interval,
        )
    )


if __name__ == "__main__":
    main()
