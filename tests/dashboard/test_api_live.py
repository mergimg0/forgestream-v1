"""Dashboard API tests with Firestore data."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from forgestream.dashboard.server import create_app


class TestDashboardLiveData:
    def test_graph_returns_firestore_concepts(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {
            "event_type": "claim",
            "payload": {"topic_keywords": ["Kafka", "ingestion"], "confidence": 0.9},
            "evaluator": 0.5,
        }
        mock_doc2 = MagicMock()
        mock_doc2.to_dict.return_value = {
            "event_type": "claim",
            "payload": {"topic_keywords": ["latency"], "confidence": 0.8},
            "evaluator": 0.6,
        }
        mock_collection.order_by.return_value.stream.return_value = [mock_doc1, mock_doc2]

        app = create_app(firestore_db=mock_db)
        client = TestClient(app)
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert len(data["concepts"]) > 0

    def test_evaluator_returns_trajectory(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "event_type": "claim",
            "evaluator": 0.45,
            "payload": {},
        }
        mock_collection.order_by.return_value.stream.return_value = [mock_doc]

        app = create_app(firestore_db=mock_db)
        client = TestClient(app)
        response = client.get("/api/evaluator")
        assert response.status_code == 200
        data = response.json()
        assert len(data["trajectory"]) > 0

    def test_health_always_works(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_graph_empty_without_firestore(self):
        app = create_app(firestore_db=None)
        client = TestClient(app)
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["concepts"] == []
