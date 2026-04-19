"""Test unified session launcher."""

from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator


class TestSessionSetup:
    def test_attach_synthesis_engine(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = orch.attach_synthesis_engine()
        assert engine is not None
        assert engine.orchestrator is orch
        assert len(orch.event_bus._subscribers) >= 1

    def test_full_session_components(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = orch.attach_synthesis_engine()
        assert engine.orchestrator is orch

        from forgestream.tui.app import ForgeStreamApp
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch
