"""Meeting DVR -- replay saved meetings with synchronized events.

Usage (CLI):
    python3 -m forgestream replay data/emotion_corpus/audio/20260329-session.wav
    python3 -m forgestream replay data/emotion_corpus/audio/20260329-session.wav --speed 2.0
    python3 -m forgestream replay data/emotion_corpus/audio/20260329-session.wav --headless

Architecture:
    MeetingReplay loads a saved WAV + feature index, builds a sorted event
    timeline, and plays them back synchronized to wall-clock time at the
    requested speed multiplier.  Audio is played via sounddevice (optional).
    TUI output uses a Textual app when available; headless mode prints events
    to stdout.
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from typing import Any

try:
    import sounddevice as sd
    import numpy as np

    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    HAS_SOUNDDEVICE = False

from forgestream.events.schema import Event, EventType


# ---------------------------------------------------------------------------
# Timeline entry
# ---------------------------------------------------------------------------

class _TimelineEntry:
    """A single event placed on the replay timeline."""

    __slots__ = ("timestamp_ms", "event")

    def __init__(self, timestamp_ms: float, event: Event) -> None:
        self.timestamp_ms = timestamp_ms
        self.event = event

    def __lt__(self, other: "_TimelineEntry") -> bool:
        return self.timestamp_ms < other.timestamp_ms


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> bytes:
    """Read a 16kHz mono int16 WAV file and return raw PCM bytes."""
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def load_json(path: str) -> dict[str, Any]:
    """Read a JSON feature index file."""
    return json.loads(Path(path).read_text())


def _wav_duration_ms(path: str) -> float:
    """Return the duration of a WAV file in milliseconds."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate() * 1000.0


# ---------------------------------------------------------------------------
# MeetingReplay core
# ---------------------------------------------------------------------------

class MeetingReplay:
    """Replay a saved meeting with synchronized audio and events.

    Parameters:
        audio_path: Path to the 16kHz mono int16 WAV file.
        index_path: Path to the JSON feature index produced by EmotionCorpus.
        events: List of Event objects for the session.  May be empty.
        speed: Playback speed multiplier (0.5x, 1.0x, 2.0x …).
            speed=0 means as-fast-as-possible (no timing delay).

    The replay can operate in three modes:
        - TUI mode: pass tui_app to play(); events are pushed to panels.
        - Headless mode: events are printed to stdout.
        - Events-only mode: audio_path='' disables audio; events still fire.
    """

    SAMPLE_RATE: int = 16000
    CHUNK_DURATION_S: float = 0.5  # seconds per audio chunk
    CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_DURATION_S)
    CHUNK_BYTES: int = CHUNK_SAMPLES * 2  # 16-bit = 2 bytes/sample

    def __init__(
        self,
        audio_path: str,
        index_path: str,
        events: list[Event],
        speed: float = 1.0,
    ) -> None:
        self._audio_path = audio_path
        self._index_path = index_path
        self._events = list(events)
        self._speed = max(0.0, speed)

        # Lazily loaded
        self._audio_bytes: bytes | None = None
        self._index: dict[str, Any] | None = None
        self._timeline: list[_TimelineEntry] = []
        self._duration_ms: float = 0.0

        self._paused = False
        self._stopped = False
        self._current_ms: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = max(0.0, value)

    @property
    def current_ms(self) -> float:
        """Current playback position in milliseconds."""
        return self._current_ms

    @property
    def duration_ms(self) -> float:
        return self._duration_ms

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load audio and build event timeline.

        Safe to call multiple times (no-op on second call).
        """
        if self._index is not None:
            return  # already loaded

        if self._audio_path and Path(self._audio_path).exists():
            self._audio_bytes = load_wav(self._audio_path)
            self._duration_ms = _wav_duration_ms(self._audio_path)

        if self._index_path and Path(self._index_path).exists():
            self._index = load_json(self._index_path)
        else:
            self._index = {}

        self._build_timeline()

    def _build_timeline(self) -> None:
        """Build sorted timeline from events and index entries."""
        entries: list[_TimelineEntry] = []
        idx = self._index or {}

        # Events passed directly (from DB or JSON export)
        for ev in self._events:
            ts_ms = _extract_timestamp_ms(ev)
            entries.append(_TimelineEntry(ts_ms, ev))

        # Prosodic features from the index (if no matching Event objects)
        seen_ids = {str(ev.id) for ev in self._events}
        for pf in idx.get("prosodic_features", []):
            if pf.get("event_id") in seen_ids:
                continue
            ts_ms = float(pf.get("timestamp_ms", 0))
            ev = _make_synthetic_event(
                event_type=EventType.PROSODIC_FEATURE,
                session_id=idx.get("session_id", ""),
                payload={
                    "timestamp_ms": ts_ms,
                    "arousal": pf.get("arousal", 0.5),
                    "valence": pf.get("valence", 0.5),
                    "dominance": pf.get("dominance", 0.5),
                    "speaker_id": pf.get("speaker_id", "unknown"),
                    "emotion_tag": pf.get("emotion_tag"),
                    "_source": "index",
                },
            )
            entries.append(_TimelineEntry(ts_ms, ev))

        # Claims from the index
        for cl in idx.get("claims", []):
            if cl.get("event_id") in seen_ids:
                continue
            audio_ts = cl.get("audio_timestamp")
            ts_ms = float(audio_ts) if audio_ts is not None else 0.0
            ev = _make_synthetic_event(
                event_type=EventType.CLAIM,
                session_id=idx.get("session_id", ""),
                payload={
                    "text": cl.get("text", ""),
                    "audio_timestamp": ts_ms,
                    "confidence": cl.get("confidence", 0.5),
                    "speaker": cl.get("speaker", "unknown"),
                    "_source": "index",
                },
            )
            entries.append(_TimelineEntry(ts_ms, ev))

        entries.sort()
        self._timeline = entries

        # Extend duration to cover all events if audio is absent
        if self._duration_ms == 0.0 and entries:
            self._duration_ms = entries[-1].timestamp_ms + 500.0

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stopped = True

    async def play(self, tui_app: Any = None) -> None:
        """Play back the meeting with synchronized events.

        Parameters:
            tui_app: Optional Textual app.  If provided, events are dispatched
                to the app's panels; otherwise events are printed to stdout.
        """
        self.load()

        if self._audio_bytes and HAS_SOUNDDEVICE:
            audio_task = asyncio.create_task(self._play_audio())
        else:
            audio_task = None

        try:
            await self._replay_events(tui_app)
        finally:
            if audio_task is not None:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass

    async def _play_audio(self) -> None:
        """Stream audio in chunks via sounddevice (background task)."""
        if not HAS_SOUNDDEVICE or not self._audio_bytes:
            return
        assert np is not None and sd is not None  # guarded by HAS_SOUNDDEVICE

        audio_data = np.frombuffer(self._audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        chunk_samples = self.CHUNK_SAMPLES
        total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
        loop = asyncio.get_event_loop()

        for i in range(total_chunks):
            if self._stopped:
                break

            while self._paused:
                await asyncio.sleep(0.05)

            chunk = audio_data[i * chunk_samples : (i + 1) * chunk_samples]
            if len(chunk) < chunk_samples:
                padding = np.zeros(chunk_samples - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, padding])

            chunk_2d = chunk.reshape(-1, 1)
            done_event = asyncio.Event()

            def _callback(outdata, frames, time_info, status, _chunk=chunk_2d, _ev=done_event):
                outdata[:] = _chunk
                loop.call_soon_threadsafe(_ev.set)

            stream = sd.OutputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                callback=_callback,
            )
            with stream:
                await done_event.wait()

            if self._speed > 0:
                await asyncio.sleep(self.CHUNK_DURATION_S / self._speed)

    async def _replay_events(self, tui_app: Any) -> None:
        """Walk the event timeline, waiting for each event's timestamp."""
        if not self._timeline:
            return

        start_wall = asyncio.get_event_loop().time()
        event_idx = 0
        total = len(self._timeline)

        while event_idx < total and not self._stopped:
            while self._paused:
                await asyncio.sleep(0.05)
                start_wall += 0.05  # compensate pause time

            entry = self._timeline[event_idx]
            event_ms = entry.timestamp_ms

            # How much wall time should have elapsed to reach this event?
            if self._speed > 0:
                target_wall_s = (event_ms / 1000.0) / self._speed
                elapsed_wall_s = asyncio.get_event_loop().time() - start_wall
                wait_s = target_wall_s - elapsed_wall_s
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

            self._current_ms = event_ms
            _dispatch_event(entry.event, tui_app, timeline_ms=event_ms)
            event_idx += 1

        if not self._stopped:
            self._current_ms = self._duration_ms


# ---------------------------------------------------------------------------
# Event dispatch / rendering
# ---------------------------------------------------------------------------

def _dispatch_event(event: Event, tui_app: Any, timeline_ms: float = 0.0) -> None:
    """Send event to TUI app or print to stdout."""
    if tui_app is not None:
        _dispatch_to_tui(event, tui_app)
    else:
        _print_event(event, timeline_ms=timeline_ms)


def _dispatch_to_tui(event: Event, tui_app: Any) -> None:
    """Push event into the TUI app's panels (best-effort)."""
    try:
        from forgestream.tui.panels.feed import FeedPanel

        feed = tui_app.query_one("#feed", FeedPanel)
        feed.on_event_received(event)
    except Exception:
        pass

    try:
        from forgestream.tui.panels.suggestions import SuggestionsPanel

        suggestions = tui_app.query_one("#suggestions", SuggestionsPanel)
        suggestions.on_event_received(event)
    except Exception:
        pass

    # Update evaluator bar if present
    try:
        from forgestream.tui.app import EvaluatorBar

        bar = tui_app.query_one("#evaluator-bar", EvaluatorBar)
        bar.update_evaluator(event.evaluator, 0, "replay")
    except Exception:
        pass


def _print_event(event: Event, timeline_ms: float = 0.0) -> None:
    """Print a human-readable event line to stdout.

    Uses timeline_ms (the sorted position) for the display timestamp so the
    output always reflects playback order, regardless of which payload key
    stores the original timestamp.
    """
    ts_ms = timeline_ms if timeline_ms > 0.0 else _extract_timestamp_ms(event)
    ts_s = ts_ms / 1000.0
    minutes = int(ts_s // 60)
    seconds = ts_s % 60
    ts_str = f"{minutes:02d}:{seconds:05.2f}"

    et = event.event_type

    if et == EventType.CLAIM:
        speaker = event.payload.get("speaker", "?")
        text = event.payload.get("text", "")
        conf = event.payload.get("confidence", 0.5)
        print(f"[{ts_str}] CLAIM [{speaker}] (conf={conf:.2f}) {text}")

    elif et == EventType.PROSODIC_FEATURE:
        arousal = event.payload.get("arousal", 0.5)
        valence = event.payload.get("valence", 0.5)
        speaker = event.payload.get("speaker_id", "?")
        emotion = event.payload.get("emotion_tag") or ""
        tag = f" [{emotion}]" if emotion else ""
        print(f"[{ts_str}] PROSODIC [{speaker}] A={arousal:.2f} V={valence:.2f}{tag}")

    elif et == EventType.EMOTION_STATE:
        label = event.payload.get("label", "?")
        speaker = event.payload.get("speaker_id", "?")
        print(f"[{ts_str}] EMOTION_SHIFT [{speaker}] -> {label}")

    elif et == EventType.RAPPORT_SCORE:
        score = event.payload.get("score", 0.0)
        print(f"[{ts_str}] RAPPORT  score={score:.3f}")

    elif et == EventType.ENTRAINMENT_SNAPSHOT:
        rho = event.payload.get("rr", 0.0)
        print(f"[{ts_str}] ENTRAINMENT  rr={rho:.3f}")

    else:
        print(f"[{ts_str}] {et.value.upper()} {event.payload}")


# ---------------------------------------------------------------------------
# Textual replay app
# ---------------------------------------------------------------------------

class ReplayControlBar:
    """Placeholder — a real Textual widget would go here in a full TUI sprint.

    Currently prints control state to stdout.  This is intentionally minimal
    to keep this MVP focused on the core replay logic.
    """

    def __init__(self, replay: MeetingReplay) -> None:
        self._replay = replay

    def show_status(self) -> None:
        pos = self._replay.current_ms / 1000.0
        dur = self._replay.duration_ms / 1000.0
        speed = self._replay.speed
        print(f"[REPLAY] {pos:.1f}s / {dur:.1f}s  speed={speed}x  "
              f"{'PAUSED' if self._replay._paused else 'PLAYING'}")


# ---------------------------------------------------------------------------
# Event loading helpers
# ---------------------------------------------------------------------------

async def load_events_from_db(
    session_id: str,
    postgres_dsn: str,
) -> list[Event]:
    """Load events for a session from PostgreSQL.

    Returns empty list if psycopg is unavailable or connection fails.
    """
    try:
        import psycopg
        from uuid import UUID

        from forgestream.events.store import EventStore

        conn = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        store = EventStore(conn)
        events = await store.get_events(UUID(session_id))
        await conn.close()
        return events
    except Exception:
        return []


def load_events_from_json(json_path: str) -> list[Event]:
    """Load events from a JSON export file (list of event dicts).

    Returns empty list if file is missing or malformed.
    """
    try:
        data = json.loads(Path(json_path).read_text())
        if isinstance(data, list):
            return [Event.from_dict(d) for d in data]
        if isinstance(data, dict) and "events" in data:
            return [Event.from_dict(d) for d in data["events"]]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extract_timestamp_ms(event: Event) -> float:
    """Extract the best audio timestamp (ms) from an event payload."""
    p = event.payload
    for key in ("timestamp_ms", "audio_timestamp", "audio_timestamp_ms"):
        if key in p and p[key] is not None:
            return float(p[key])
    # Fall back to event wall-clock time relative to epoch (not ideal but
    # ensures every event lands on the timeline)
    return 0.0


def _make_synthetic_event(
    event_type: EventType,
    session_id: str,
    payload: dict[str, Any],
) -> Event:
    """Build a minimal Event from index data (no real UUID/branch needed)."""
    from uuid import uuid4

    # Use a zero UUID for synthetic events from index files
    from uuid import UUID

    session_uuid = UUID(session_id) if _is_valid_uuid(session_id) else UUID(int=0)
    return Event(
        event_type=event_type,
        session_id=session_uuid,
        branch_id=UUID(int=0),
        author="replay",
        evaluator=0.5,
        payload=payload,
    )


def _is_valid_uuid(value: str) -> bool:
    from uuid import UUID

    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False
