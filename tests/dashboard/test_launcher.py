"""Test dashboard launcher with Firestore."""

from unittest.mock import MagicMock, patch

from forgestream.config import ForgeStreamConfig
from forgestream.dashboard.launcher import create_live_app


class TestDashboardLauncher:
    def test_creates_app_without_firestore(self):
        config = ForgeStreamConfig(firestore_enabled=False)
        app = create_live_app(config)
        assert app is not None

    @patch("forgestream.dashboard.launcher.firebase_admin")
    @patch("forgestream.dashboard.launcher.firestore")
    def test_creates_app_with_firestore(self, mock_firestore, mock_admin):
        mock_db = MagicMock()
        mock_firestore.client.return_value = mock_db
        mock_admin.get_app.side_effect = ValueError("no app")

        config = ForgeStreamConfig(firestore_enabled=True)
        app = create_live_app(config)
        assert app is not None

    def test_app_serves_health(self):
        from fastapi.testclient import TestClient

        config = ForgeStreamConfig(firestore_enabled=False)
        app = create_live_app(config)
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
