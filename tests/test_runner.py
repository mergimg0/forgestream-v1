"""Runner tests -- verify the TUI integration path."""

from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator
from forgestream.tui.app import ForgeStreamApp


class TestRunner:
    def test_app_accepts_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch

    def test_app_works_without_orchestrator(self):
        app = ForgeStreamApp()
        assert app.orchestrator is None
