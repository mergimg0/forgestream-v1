"""Test human feedback flow."""

from forgestream.tui.app import ForgeStreamApp
from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator


class TestFeedbackPrompt:
    def test_app_has_end_meeting_action(self):
        app = ForgeStreamApp()
        assert hasattr(app, "action_end_meeting")

    def test_app_tracks_meeting_ended_state(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app._meeting_ended is False

    def test_parse_feedback_score_valid(self):
        assert ForgeStreamApp._parse_feedback("8") == 0.8
        assert ForgeStreamApp._parse_feedback("10") == 1.0
        assert ForgeStreamApp._parse_feedback("1") == 0.1

    def test_parse_feedback_score_empty(self):
        assert ForgeStreamApp._parse_feedback("") is None
        assert ForgeStreamApp._parse_feedback("skip") is None

    def test_parse_feedback_score_invalid(self):
        assert ForgeStreamApp._parse_feedback("abc") is None
        assert ForgeStreamApp._parse_feedback("0") is None
        assert ForgeStreamApp._parse_feedback("11") is None
