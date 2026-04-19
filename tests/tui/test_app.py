"""TUI app structure tests — verify widgets and layout exist."""

from forgestream.tui.app import ForgeStreamApp


class TestForgeStreamApp:
    def test_app_instantiates(self):
        app = ForgeStreamApp()
        assert app.title == "ForgeStream"

    def test_app_has_css(self):
        app = ForgeStreamApp()
        assert len(app.CSS) > 0
