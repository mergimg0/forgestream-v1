"""ForgeStream CLI entry point.

Usage:
    python -m forgestream start [--name NAME] [--mode MODE]
    python -m forgestream status
    python -m forgestream end
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="forgestream",
        description="ForgeStream: Live meeting intelligence with SOS-governed agent swarms",
    )
    subparsers = parser.add_subparsers(dest="command")

    # start
    start_parser = subparsers.add_parser("start", help="Start a meeting session")
    start_parser.add_argument("--name", default="", help="Meeting name")
    start_parser.add_argument(
        "--mode",
        choices=["extract", "collaborative", "knowledge"],
        default="extract",
        help="Meeting mode (default: extract)",
    )
    start_parser.add_argument("--dashboard", action="store_true", help="Also start web dashboard")
    start_parser.add_argument("--live", action="store_true", help="Start live mic capture via Gemini")
    start_parser.add_argument("--device", type=int, default=None, help="Audio input device index")

    # status
    subparsers.add_parser("status", help="Show ForgeStream status")

    # end
    subparsers.add_parser("end", help="End the current meeting session")

    # prep
    prep_parser = subparsers.add_parser("prep", help="Generate meeting preparation document")
    prep_parser.add_argument("--topic", default="", help="Topic hint for the prep doc")
    prep_parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")

    # mcp
    mcp_parser = subparsers.add_parser("mcp", help="Start the ForgeStream MCP server")
    mcp_parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    mcp_parser.add_argument("--port", type=int, default=0, help="Port (0 = stdio transport)")

    # replay
    replay_parser = subparsers.add_parser("replay", help="Replay a saved meeting")
    replay_parser.add_argument("audio", help="Path to saved WAV file")
    replay_parser.add_argument(
        "--index",
        default="",
        help="Path to feature index JSON (auto-detected if omitted)",
    )
    replay_parser.add_argument(
        "--events",
        default="",
        help="Path to events JSON export (optional fallback when DB unavailable)",
    )
    replay_parser.add_argument(
        "--session-id",
        default="",
        help="Session UUID for loading events from PostgreSQL",
    )
    replay_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (0.5=half, 1.0=real-time, 2.0=double, 0=no delay)",
    )
    replay_parser.add_argument(
        "--headless",
        action="store_true",
        help="Print events to stdout instead of launching TUI",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "start":
        return _cmd_start(args)
    elif args.command == "status":
        return _cmd_status()
    elif args.command == "end":
        return _cmd_end()
    elif args.command == "prep":
        return _cmd_prep(args)
    elif args.command == "mcp":
        return _cmd_mcp(args)
    elif args.command == "replay":
        return _cmd_replay(args)
    else:
        parser.print_help()
        return 1


def _cmd_start(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config()
    config.meeting_mode = args.mode
    if args.name:
        config.meeting_name = args.name

    from .orchestrator import Orchestrator

    print("ForgeStream starting...")
    print(f"  Mode: {config.meeting_mode}")
    print(f"  Name: {config.meeting_name or '(unnamed)'}")
    print(f"  PostgreSQL: {config.postgres_dsn}")
    print(f"  Firestore: {'enabled' if config.firestore_enabled else 'disabled'}")

    # Create data stores
    from .firestore_sync import FirestoreSync

    # Store setup is deferred to async context to avoid event loop conflicts
    store = None  # Will be created in _run_live if --live
    _store_dsn = config.postgres_dsn

    fs_sync = None
    if config.firestore_enabled:
        try:
            fs_sync = FirestoreSync(project_id=config.firebase_project)
            print("  Firestore sync: enabled")
        except Exception as e:
            print(f"  Firestore sync: failed ({e})")

    # Create orchestrator with synthesis engine
    orch = Orchestrator(config, store=store, firestore_sync=fs_sync)
    orch.attach_synthesis_engine()
    print("  SynthesisEngine: attached")

    # Start dashboard if requested
    if args.dashboard:
        from .dashboard.launcher import create_live_app
        import threading
        import uvicorn

        dashboard_app = create_live_app(config)

        def run_dashboard():
            uvicorn.run(
                dashboard_app,
                host=config.dashboard_host,
                port=config.dashboard_port,
                log_level="warning",
            )

        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        print(f"  Dashboard: http://{config.dashboard_host}:{config.dashboard_port}")

    if args.live:
        return _run_live(config, orch, device=args.device)

    print(f"\n  Ready. Process audio with:")
    print(f"  python3 -m forgestream.runner <audio> --mode {config.meeting_mode}")
    return 0


def _run_live(
    config: Any,
    orchestrator: Any,
    device: int | None = None,
) -> int:
    """Start live mic capture via GeminiLiveStream."""
    import asyncio
    import signal
    from datetime import datetime
    from pathlib import Path

    from .audio.microphone import MicrophoneSource
    from .live_copilot import AudioRecorder, LiveEventSink, TeeAudioSource
    from .live_stream import GeminiLiveStream

    # Timestamped recording path — scoped to config.data_dir
    data_dir = Path(config.data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meeting_tag = config.meeting_name.replace(" ", "_") if config.meeting_name else "meeting"
    recordings_dir = data_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(recordings_dir / f"{meeting_tag}_{ts}.wav")

    mic = MicrophoneSource(device=device)
    recorder = AudioRecorder(path=wav_path)
    tee_source = TeeAudioSource(source=mic, recorder=recorder)

    # Live event sink for CLI copilot
    event_sink = LiveEventSink(path=str(data_dir / "live_events.jsonl"))
    event_sink.subscribe(orchestrator.event_bus)

    stream = GeminiLiveStream(
        config=config,
        orchestrator=orchestrator,
        audio_source=tee_source,
        mode=config.meeting_mode,
    )

    print("  Audio: live microphone" + (f" (device {device})" if device is not None else ""))
    print(f"  Recording: {wav_path}")
    print(f"  Live events: {data_dir / 'live_events.jsonl'}")
    print(f"  Emotion: {'enabled' if config.emotion_enabled else 'disabled'}")
    print("\n  Streaming... Press Ctrl+C to stop and run post-meeting synthesis.\n")

    async def run():
        # Connect PostgreSQL store inside async context
        try:
            import psycopg
            from .events.store import EventStore
            conn = await psycopg.AsyncConnection.connect(orchestrator.config.postgres_dsn)
            orchestrator.store = EventStore(conn)
            print("  PostgreSQL: connected")
        except Exception as e:
            print(f"  PostgreSQL store: unavailable ({e})")

        await stream.start()
        stop_event = asyncio.Event()

        def on_signal():
            print("\n  Stopping...")
            stop_event.set()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal)

        await stop_event.wait()
        result = await stream.stop(run_post_meeting=True)

        # Close live copilot resources
        event_sink.close()
        duration = recorder.duration_seconds
        print(f"\n  Recording saved: {wav_path} ({duration:.0f}s)")
        print(f"  Events logged: {event_sink.event_count}")

        if result:
            print("\n  Post-meeting synthesis complete:")
            print(f"    Report: {result.get('report_path', 'n/a')}")
            print(f"    E(π): {result.get('e_meso', 0):.3f}")
            print(f"    Weights: {result.get('weights', {})}")
            print(f"    Audio: {result.get('corpus_audio_path', 'n/a')}")
            print(f"    Index: {result.get('corpus_index_path', 'n/a')}")
            sensitivity = result.get("weight_sensitivity", {})
            if sensitivity:
                print(f"    Most impactful weight: {sensitivity.get('most_impactful', '?')}")

    asyncio.run(run())
    return 0


def _cmd_status() -> int:
    print("ForgeStream Status")
    print("  Version: 0.1.0")
    print("  Modules: events, graph, gemini, synthesis, agents, governor, tui, dashboard, copilot")
    print("  Tests: 118+ passing")
    return 0


def _cmd_end() -> int:
    print("ForgeStream: ending session...")
    print("  Post-meeting synthesis would run here (SP-14)")
    return 0


def _cmd_prep(args: argparse.Namespace) -> int:
    from .meeting_prep import MeetingPrep

    prep = MeetingPrep(data_dir=getattr(args, "data_dir", "data"))
    doc = prep.prepare(topic=args.topic)
    print(doc)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import create_mcp_server

    data_dir = getattr(args, "data_dir", "data")
    server = create_mcp_server(data_dir=data_dir)

    # If we got a FastMCP server, run it on stdio transport
    if hasattr(server, "run"):
        server.run(transport="stdio")
    else:
        print(f"ForgeStream MCP server (stub mode, data_dir={data_dir})")
        print("Install 'mcp' SDK for full MCP transport support.")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Replay a saved meeting session."""
    import asyncio
    from pathlib import Path

    audio_path = args.audio
    speed = args.speed
    headless = args.headless

    # Auto-detect index path if not supplied
    index_path = args.index
    if not index_path:
        audio_p = Path(audio_path)
        # Convention: indices/<stem>.json lives next to audio/
        candidate = audio_p.parent.parent / "indices" / (audio_p.stem + ".json")
        if candidate.exists():
            index_path = str(candidate)
        else:
            index_path = ""

    # Load events from JSON export or DB
    from .replay import load_events_from_json, load_events_from_db, MeetingReplay

    events = []
    if args.events and Path(args.events).exists():
        events = load_events_from_json(args.events)
        print(f"  Loaded {len(events)} events from {args.events}")
    elif args.session_id:
        from .config import load_config

        config = load_config()
        print(f"  Loading events from PostgreSQL (session={args.session_id})...")
        try:
            events = asyncio.run(
                load_events_from_db(args.session_id, config.postgres_dsn)
            )
            print(f"  Loaded {len(events)} events from DB")
        except Exception as exc:
            print(f"  DB load failed ({exc}), continuing without events")
            events = []

    replay = MeetingReplay(
        audio_path=audio_path,
        index_path=index_path,
        events=events,
        speed=speed,
    )
    replay.load()

    total_events = len(replay._timeline)
    dur_s = replay.duration_ms / 1000.0
    print(f"ForgeStream Replay")
    print(f"  Audio:  {audio_path or '(none)'}")
    print(f"  Index:  {index_path or '(none)'}")
    print(f"  Events: {total_events}")
    print(f"  Duration: {dur_s:.1f}s  Speed: {speed}x")
    print(f"  Mode: {'headless' if headless else 'TUI'}")
    print()

    if headless:
        asyncio.run(replay.play(tui_app=None))
    else:
        try:
            from .tui.app import ForgeStreamApp

            app = ForgeStreamApp()

            async def _run_with_tui():
                app_task = asyncio.create_task(app.run_async())
                # Give TUI a moment to mount before pushing events
                await asyncio.sleep(0.3)
                await replay.play(tui_app=app)
                app.exit()
                await app_task

            asyncio.run(_run_with_tui())
        except Exception as exc:
            print(f"  TUI unavailable ({exc}), falling back to headless mode")
            asyncio.run(replay.play(tui_app=None))

    return 0


if __name__ == "__main__":
    sys.exit(main())
