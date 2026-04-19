"""ForgeStream configuration -- env vars and defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ForgeStreamConfig:
    """Configuration for a ForgeStream session."""

    # PostgreSQL
    postgres_dsn: str = "postgresql://claude:claude_dev@localhost:5432/forgestream"

    # Gemini (Vertex AI)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_project: str = "forgestream-ai"
    gemini_location: str = "europe-west2"
    gemini_use_vertex: bool = True

    # Meeting
    meeting_mode: str = "extract"  # extract, collaborative, knowledge
    meeting_name: str = ""

    # Trust region defaults
    trust_region_epsilon_base: float = 0.3
    max_concurrent_research: int = 3
    max_concurrent_scaffold: int = 4
    spawn_cooldown_seconds: int = 60
    scaffold_timeout_minutes: int = 10

    # Firebase / Firestore
    firebase_project: str = "forgestream-ai"
    firestore_enabled: bool = True

    # Dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8501

    # Paths
    meetings_dir: str = "docs/meetings"
    data_dir: str = "data"

    # Emotion pipeline
    emotion_enabled: bool = True
    emotion_window_seconds: float = 3.0   # analysis window duration
    emotion_stride_seconds: float = 1.0   # emit interval
    emotion_ml_enabled: bool = False       # SenseVoice/emotion2vec (requires torch)
    emotion_buffer_seconds: float = 30.0   # ring buffer capacity
    sensevoice_model: str = "iic/SenseVoiceSmall"  # FunASR model identifier

    # Speaker diarization
    diarization_enabled: bool = False  # opt-in (requires HF token + pyannote)
    huggingface_token: str = ""
    diarization_update_interval: int = 10  # re-run diarization every N chunks

    # Rapport tracking
    rapport_enabled: bool = True
    rapport_damping_factor: float = 0.3    # disengagement damping multiplier
    runpod_crqa_endpoint: str = ""         # RunPod URL (empty = local fallback only)
    runpod_timeout_seconds: float = 4.0    # per-request timeout

    # User profile (Theme 5)
    user_profile_path: str = "data/user_profile.json"


def load_config_overrides(
    data_dir: str = "data",
    config: ForgeStreamConfig | None = None,
) -> ForgeStreamConfig:
    """Load saved config overrides from data/config_overrides.json and merge with defaults.

    Reads the JSON written by PostMeetingSynthesis.save_config_overrides() and
    returns a new ForgeStreamConfig with those values applied.  Unknown keys
    in the file are silently ignored (forward-compat).

    Args:
        data_dir: Directory containing config_overrides.json (default: "data").
        config: Base config to merge into; uses load_config() defaults if None.
    """
    base = config if config is not None else load_config()
    overrides_path = Path(data_dir) / "config_overrides.json"
    if not overrides_path.exists():
        return base

    try:
        overrides = json.loads(overrides_path.read_text())
    except (json.JSONDecodeError, OSError):
        return base

    _TUNABLE_FIELDS = {
        "spawn_cooldown_seconds",
        "scaffold_timeout_minutes",
        "max_concurrent_research",
        "max_concurrent_scaffold",
        "emotion_stride_seconds",
        "emotion_window_seconds",
    }

    updated: dict = {}
    for field_name in _TUNABLE_FIELDS:
        if field_name in overrides:
            raw = overrides[field_name]
            current_val = getattr(base, field_name)
            if isinstance(current_val, int):
                updated[field_name] = int(round(raw))
            else:
                updated[field_name] = float(raw)

    if not updated:
        return base

    import dataclasses
    return dataclasses.replace(base, **updated)


def load_config() -> ForgeStreamConfig:
    """Load config from environment variables with FORGESTREAM_ prefix."""
    data_dir = os.environ.get("FORGESTREAM_DATA_DIR", ForgeStreamConfig.data_dir)
    meetings_dir = os.environ.get(
        "FORGESTREAM_MEETINGS_DIR", ForgeStreamConfig.meetings_dir
    )
    return ForgeStreamConfig(
        postgres_dsn=os.environ.get(
            "FORGESTREAM_POSTGRES_DSN",
            ForgeStreamConfig.postgres_dsn,
        ),
        gemini_api_key=os.environ.get("FORGESTREAM_GEMINI_API_KEY", ""),
        gemini_model=os.environ.get(
            "FORGESTREAM_GEMINI_MODEL",
            ForgeStreamConfig.gemini_model,
        ),
        meeting_mode=os.environ.get(
            "FORGESTREAM_MEETING_MODE",
            ForgeStreamConfig.meeting_mode,
        ),
        meeting_name=os.environ.get("FORGESTREAM_MEETING_NAME", ""),
        dashboard_host=os.environ.get(
            "FORGESTREAM_DASHBOARD_HOST",
            ForgeStreamConfig.dashboard_host,
        ),
        dashboard_port=int(
            os.environ.get(
                "FORGESTREAM_DASHBOARD_PORT",
                str(ForgeStreamConfig.dashboard_port),
            )
        ),
        firestore_enabled=os.environ.get(
            "FORGESTREAM_FIRESTORE_ENABLED", "true"
        ).lower() == "true",
        data_dir=data_dir,
        meetings_dir=meetings_dir,
        user_profile_path=os.environ.get(
            "FORGESTREAM_USER_PROFILE_PATH",
            str(Path(data_dir) / "user_profile.json"),
        ),
    )
