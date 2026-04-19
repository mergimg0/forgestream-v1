"""Tests for GRPO end-to-end loop — weight loading at startup, post-meeting synthesis on stop."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator
from forgestream.post_meeting import PostMeetingSynthesis


class TestGRPOLoop:
    def test_weights_loaded_at_startup(self):
        """GeminiLiveStream loads saved weights into the evaluator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save custom weights
            weights = {
                "knowledge": 0.20, "verification": 0.20,
                "scaffold": 0.20, "uptake": 0.20, "engagement": 0.20,
                "meeting_count": 5,
            }
            Path(tmpdir, "weights.json").write_text(json.dumps(weights))

            config = ForgeStreamConfig(
                emotion_enabled=True, data_dir=tmpdir,
            )
            orch = Orchestrator(config=config)
            source = MagicMock()
            stream = GeminiLiveStream(config, orch, source)

            # Evaluator should have loaded the custom weights
            assert orch.evaluator.weights["knowledge"] == pytest.approx(0.20)
            assert orch.evaluator.weights["engagement"] == pytest.approx(0.20)
            assert stream._meeting_count == 5

    def test_default_weights_when_no_file(self):
        """Without saved weights, defaults are used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(
                emotion_enabled=True, data_dir=tmpdir,
            )
            orch = Orchestrator(config=config)
            source = MagicMock()
            stream = GeminiLiveStream(config, orch, source)

            assert orch.evaluator.weights["engagement"] == pytest.approx(0.15)
            assert stream._meeting_count == 1

    @pytest.mark.asyncio
    async def test_stop_runs_post_meeting(self):
        """stop() with run_post_meeting=True invokes PostMeetingSynthesis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(
                emotion_enabled=True, data_dir=tmpdir,
                meetings_dir=str(Path(tmpdir) / "meetings"),
            )
            orch = Orchestrator(config=config)
            source = MagicMock()
            source.stop = MagicMock(return_value=None)
            # Make stop() awaitable
            import asyncio
            async def noop(): pass
            source.stop = noop

            stream = GeminiLiveStream(config, orch, source)
            stream._active = True
            stream._tasks = []

            result = await stream.stop(run_post_meeting=True)
            assert result is not None
            assert "weights" in result
            assert "meeting_count" in result
            assert "rapport_weights" in result

    @pytest.mark.asyncio
    async def test_stop_skips_post_meeting_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig(
                emotion_enabled=True, data_dir=tmpdir,
            )
            orch = Orchestrator(config=config)
            source = MagicMock()
            import asyncio
            async def noop(): pass
            source.stop = noop

            stream = GeminiLiveStream(config, orch, source)
            stream._active = True
            stream._tasks = []

            result = await stream.stop(run_post_meeting=False)
            assert result is None
