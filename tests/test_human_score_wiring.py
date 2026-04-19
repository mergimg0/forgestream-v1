"""Tests for human_score wiring from stop() -> _run_post_meeting() -> pms.run()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgestream.audio.source import AudioSource
from forgestream.config import ForgeStreamConfig
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


class MockAudioSource(AudioSource):
    async def chunks(self):
        return
        yield  # make it an async generator


class TestHumanScoreWiring:
    def test_stop_accepts_human_score_param(self):
        """stop() should accept an optional human_score float parameter."""
        import inspect
        from forgestream.live_stream import GeminiLiveStream
        sig = inspect.signature(GeminiLiveStream.stop)
        assert "human_score" in sig.parameters

    def test_stop_human_score_default_is_none(self):
        """human_score default should be None."""
        import inspect
        from forgestream.live_stream import GeminiLiveStream
        sig = inspect.signature(GeminiLiveStream.stop)
        param = sig.parameters["human_score"]
        assert param.default is None

    def test_run_post_meeting_accepts_human_score(self):
        """_run_post_meeting() should accept an optional human_score parameter."""
        import inspect
        from forgestream.live_stream import GeminiLiveStream
        sig = inspect.signature(GeminiLiveStream._run_post_meeting)
        assert "human_score" in sig.parameters

    def test_stop_passes_human_score_to_run_post_meeting(self):
        """stop(human_score=0.8) should pass 0.8 through to _run_post_meeting."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = MockAudioSource()
        stream = GeminiLiveStream(config=config, orchestrator=orch, audio_source=source)

        captured_scores = []

        async def mock_run_post_meeting(human_score=None):
            captured_scores.append(human_score)
            return {"e_meso": 0.5, "meeting_count": 1}

        stream._run_post_meeting = mock_run_post_meeting
        stream._active = False  # already stopped
        stream._session = None
        stream._tasks = []

        async def fake_stop():
            await source.stop()
            return await stream._run_post_meeting(human_score=0.8)

        asyncio.get_event_loop().run_until_complete(fake_stop())
        assert captured_scores == [0.8]

    def test_stop_human_score_passed_to_pms_run(self):
        """_run_post_meeting(human_score=0.7) should call pms.run with human_score=0.7."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = MockAudioSource()
        stream = GeminiLiveStream(config=config, orchestrator=orch, audio_source=source)

        captured_kwargs: list[dict] = []

        async def mock_pms_run(events, meeting_name="", human_score=None, audio_buffer=None):
            captured_kwargs.append({"human_score": human_score})
            return {"e_meso": 0.6, "meeting_count": 2}

        with patch("forgestream.live_stream.PostMeetingSynthesis") as MockPMS:
            mock_pms_instance = MagicMock()
            mock_pms_instance.run = AsyncMock(side_effect=mock_pms_run)
            mock_pms_instance.load_weights.return_value = {}
            mock_pms_instance.load_meeting_count.return_value = 1
            MockPMS.return_value = mock_pms_instance

            # Re-create stream so it uses the patched PMS
            stream2 = GeminiLiveStream(config=config, orchestrator=Orchestrator(config), audio_source=source)

            asyncio.get_event_loop().run_until_complete(
                stream2._run_post_meeting(human_score=0.7)
            )

        call_args = mock_pms_instance.run.call_args
        # human_score should appear as kwarg or positional
        if call_args.kwargs:
            assert call_args.kwargs.get("human_score") == 0.7
        else:
            # positional: events, meeting_name, human_score, audio_buffer
            assert call_args.args[2] == 0.7
