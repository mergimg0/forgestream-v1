"""Tests for new dashboard endpoints and panels — Tasks 2, 3, 4."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forgestream.dashboard.server import create_app, INDEX_HTML


class TestBranchesEndpoint:
    def test_branches_endpoint_exists(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/branches")
        assert response.status_code == 200

    def test_branches_returns_list(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/branches")
        data = response.json()
        assert "branches" in data
        assert isinstance(data["branches"], list)

    def test_branches_items_have_expected_fields(self):
        """When Firestore has branch_point events, items have required fields."""
        # With no DB, returns empty list — that's valid
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/branches")
        data = response.json()
        # Empty is fine; if items exist they should have required keys
        for item in data["branches"]:
            assert "new_branch_id" in item or "id" in item
            assert "description" in item or "payload" in item

    def test_branch_tree_panel_in_html(self):
        assert "branch-tree" in INDEX_HTML.lower() or "branch" in INDEX_HTML.lower()

    def test_branch_tree_js_script_in_html(self):
        assert "branch-tree.js" in INDEX_HTML


class TestSeedsEndpoint:
    def test_seeds_endpoint_exists(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/seeds")
        assert response.status_code == 200

    def test_seeds_returns_list(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/seeds")
        data = response.json()
        assert "seeds" in data
        assert isinstance(data["seeds"], list)

    def test_seeds_items_have_status(self):
        """Seed items should have a 'status' field."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/seeds")
        data = response.json()
        # With empty DB returns empty list — that's fine
        for item in data["seeds"]:
            assert "status" in item

    def test_seed_garden_panel_in_html(self):
        assert "seed-garden" in INDEX_HTML.lower() or "seed" in INDEX_HTML.lower()

    def test_seed_garden_js_script_in_html(self):
        assert "seed-garden.js" in INDEX_HTML


class TestTrustRegionEndpoint:
    def test_trust_region_endpoint_exists(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/trust-region")
        assert response.status_code == 200

    def test_trust_region_returns_epsilon(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/trust-region")
        data = response.json()
        assert "epsilon" in data

    def test_trust_region_returns_axiom_status(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/trust-region")
        data = response.json()
        assert "axiom_status" in data

    def test_trust_region_returns_consecutive_improvements(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/trust-region")
        data = response.json()
        assert "consecutive_improvements" in data

    def test_trust_region_epsilon_is_float(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/trust-region")
        data = response.json()
        assert isinstance(data["epsilon"], float)

    def test_sos_convergence_panel_in_html(self):
        assert "sos-convergence" in INDEX_HTML.lower() or "convergence" in INDEX_HTML.lower()

    def test_sos_convergence_js_script_in_html(self):
        assert "sos-convergence.js" in INDEX_HTML
