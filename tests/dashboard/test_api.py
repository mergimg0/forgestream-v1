"""Dashboard API tests."""

from fastapi.testclient import TestClient

from forgestream.dashboard.server import create_app


class TestDashboardAPI:
    def test_health_endpoint(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_graph_endpoint_empty(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert "concepts" in data
        assert "edges" in data

    def test_evaluator_endpoint(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/evaluator")
        assert response.status_code == 200
        data = response.json()
        assert "trajectory" in data

    def test_static_index(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "ForgeStream" in response.text
