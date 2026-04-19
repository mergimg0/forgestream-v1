"""Configuration tests."""

import os
from pathlib import Path

from forgestream.config import ForgeStreamConfig, load_config


class TestConfig:
    def test_default_config(self):
        config = ForgeStreamConfig()
        assert config.postgres_dsn == "postgresql://claude:claude_dev@localhost:5432/forgestream"
        assert config.gemini_model == "gemini-2.5-flash"
        assert config.meeting_mode == "extract"

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("FORGESTREAM_POSTGRES_DSN", "postgresql://other:pw@host/db")
        monkeypatch.setenv("FORGESTREAM_GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("FORGESTREAM_MEETING_MODE", "collaborative")

        config = load_config()
        assert config.postgres_dsn == "postgresql://other:pw@host/db"
        assert config.gemini_api_key == "test-key-123"
        assert config.meeting_mode == "collaborative"

    def test_config_trust_region_defaults(self):
        config = ForgeStreamConfig()
        assert config.trust_region_epsilon_base == 0.3
        assert config.max_concurrent_research == 3
        assert config.max_concurrent_scaffold == 4

    def test_config_validates_mode(self):
        config = ForgeStreamConfig(meeting_mode="extract")
        assert config.meeting_mode == "extract"
        config = ForgeStreamConfig(meeting_mode="collaborative")
        assert config.meeting_mode == "collaborative"
        config = ForgeStreamConfig(meeting_mode="knowledge")
        assert config.meeting_mode == "knowledge"
