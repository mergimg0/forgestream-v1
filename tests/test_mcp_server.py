"""MCP server tests — 6 tools that read from data/ directory."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType


def _write_knowledge_graph(data_dir: str) -> None:
    """Write a minimal knowledge_graph.json to data_dir."""
    data = {
        "concepts": [
            {"name": "quantum", "domain": "physics", "confidence": 0.9},
            {"name": "entanglement", "domain": "physics", "confidence": 0.7},
            {"name": "ml", "domain": "computing", "confidence": 0.85},
        ],
        "requirements": [
            {"id": "req-1", "description": "Low latency pipeline", "domain": "infra"},
        ],
        "seeds": [
            {
                "hypothesis": "Entanglement could accelerate ML",
                "status": "active",
                "confidence": 0.4,
            },
        ],
        "contradictions": [
            {
                "concept_a": "quantum",
                "concept_b": "classical",
                "description": "Quantum vs classical interpretation",
                "resolved": False,
            },
        ],
    }
    Path(data_dir).joinpath("knowledge_graph.json").write_text(json.dumps(data))


def _write_claims(data_dir: str) -> None:
    """Write a minimal claims_index.json."""
    claims = [
        {"text": "Kafka achieves sub-10ms latency", "speaker_id": "alice", "confidence": 0.9},
        {"text": "Quantum computers need error correction", "speaker_id": "bob", "confidence": 0.8},
    ]
    Path(data_dir).joinpath("claims_index.json").write_text(json.dumps(claims))


def _write_expert_profile(data_dir: str, speaker: str = "alice") -> None:
    """Write a minimal expert profile for a speaker."""
    profiles_dir = Path(data_dir) / "expert_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "speaker_id": speaker,
        "expertise_topics": {"quantum": 0.9, "ml": 0.7},
        "communication_style": {"arousal": 0.6},
        "rapport_with_user": 0.75,
        "meetings_count": 3,
        "total_claims": 12,
    }
    (profiles_dir / f"{speaker}.json").write_text(json.dumps(profile))


class TestMCPServerTools:
    """Test each MCP tool returns valid data."""

    def test_query_knowledge_returns_list(self):
        """forgestream_query_knowledge returns a list with matching concepts."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_knowledge_graph(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.query_knowledge("quantum")

        assert isinstance(result, list)
        # At least one result should mention quantum
        assert any("quantum" in str(r).lower() for r in result)

    def test_query_knowledge_empty_when_no_match(self):
        """query_knowledge returns empty or generic response for unknown topic."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_knowledge_graph(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.query_knowledge("xyzzy_nonexistent")

        assert isinstance(result, list)

    def test_get_requirements_returns_list(self):
        """forgestream_get_requirements returns list of requirement descriptions."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_knowledge_graph(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.get_requirements()

        assert isinstance(result, list)
        assert len(result) >= 1
        assert any("latency" in str(r).lower() or "req" in str(r).lower() for r in result)

    def test_get_seeds_returns_list(self):
        """forgestream_get_seeds returns list of seeds with status."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_knowledge_graph(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.get_seeds()

        assert isinstance(result, list)
        assert len(result) >= 1
        assert any("active" in str(r).lower() for r in result)

    def test_get_contradictions_returns_list(self):
        """forgestream_get_contradictions returns unresolved contradictions."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_knowledge_graph(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.get_contradictions()

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_search_claims_returns_relevant(self):
        """forgestream_search_claims returns claims matching query."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_claims(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.search_claims("latency")

        assert isinstance(result, list)
        assert any("latency" in str(r).lower() for r in result)

    def test_search_claims_empty_when_no_match(self):
        """search_claims returns empty list when nothing matches."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_claims(tmpdir)
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.search_claims("xyzzy_totally_unique_nonexistent")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_expert_returns_profile(self):
        """forgestream_get_expert returns profile data for a known speaker."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_expert_profile(tmpdir, "alice")
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.get_expert("alice")

        assert isinstance(result, list)
        assert len(result) >= 1
        # Should contain speaker name
        assert any("alice" in str(r).lower() for r in result)

    def test_get_expert_unknown_returns_not_found(self):
        """get_expert for unknown speaker returns a not-found message."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            result = server.get_expert("nobody")

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_missing_data_files_handled_gracefully(self):
        """All tools handle missing data files without crashing."""
        from forgestream.mcp_server import ForgeStreamMCPServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = ForgeStreamMCPServer(data_dir=tmpdir)
            # None of these should raise
            server.query_knowledge("anything")
            server.get_requirements()
            server.get_seeds()
            server.get_contradictions()
            server.search_claims("anything")
            server.get_expert("someone")

    def test_create_mcp_server_function_exists(self):
        """create_mcp_server() function should exist and be callable."""
        from forgestream.mcp_server import create_mcp_server

        # Should not raise even if mcp SDK not available
        server_or_stub = create_mcp_server(data_dir="/tmp")
        assert server_or_stub is not None
