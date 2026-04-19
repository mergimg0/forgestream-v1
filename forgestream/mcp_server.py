"""MCP server exposing ForgeStream knowledge to Claude Code.

Uses the ``mcp`` Python SDK when available. If the SDK is not installed,
falls back to a lightweight stub that defines the same interface without
the dependency.

Tools exposed:
    forgestream_query_knowledge(topic)   — search the knowledge graph
    forgestream_get_requirements()       — list all detected requirements
    forgestream_get_seeds()              — list all seeds with status
    forgestream_get_contradictions()     — list unresolved contradictions
    forgestream_search_claims(query)     — full-text search across claims
    forgestream_get_expert(speaker)      — get expert profile for a speaker
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data reader helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load JSON from path; returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_knowledge_graph(data_dir: Path) -> dict[str, Any]:
    """Load knowledge_graph.json; returns empty structure if not found."""
    data = _load_json(data_dir / "knowledge_graph.json")
    if data is None:
        return {"concepts": [], "requirements": [], "seeds": [], "contradictions": []}
    return data


def _load_claims(data_dir: Path) -> list[dict[str, Any]]:
    """Load claims_index.json; returns [] if not found."""
    data = _load_json(data_dir / "claims_index.json")
    if not isinstance(data, list):
        return []
    return data


def _load_expert_profile(data_dir: Path, speaker: str) -> dict[str, Any] | None:
    """Load a speaker's expert profile JSON."""
    path = data_dir / "expert_profiles" / f"{speaker}.json"
    return _load_json(path)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Core server class (SDK-agnostic)
# ---------------------------------------------------------------------------

class ForgeStreamMCPServer:
    """Pure-Python implementation of the 6 ForgeStream MCP tools.

    Each method returns a ``list[str]`` — one element per result — so they
    are easy to test and to wrap with either the real MCP SDK or the stub.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def query_knowledge(self, topic: str) -> list[str]:
        """Search the knowledge graph for concepts matching *topic*."""
        graph = _load_knowledge_graph(self.data_dir)
        results: list[str] = []
        for concept in graph.get("concepts", []):
            name = concept.get("name", "")
            if topic.lower() in name.lower():
                conf = concept.get("confidence", 0.0)
                domain = concept.get("domain", "")
                results.append(
                    f"Concept: {name} | domain: {domain} | confidence: {conf:.2f}"
                )
        return results

    def get_requirements(self) -> list[str]:
        """List all detected requirements."""
        graph = _load_knowledge_graph(self.data_dir)
        results: list[str] = []
        for req in graph.get("requirements", []):
            req_id = req.get("id", "?")
            desc = req.get("description", "")
            domain = req.get("domain", "")
            results.append(f"[{req_id}] {desc} (domain: {domain})")
        if not results:
            results.append("No requirements found.")
        return results

    def get_seeds(self) -> list[str]:
        """List all seeds with their status."""
        graph = _load_knowledge_graph(self.data_dir)
        results: list[str] = []
        for seed in graph.get("seeds", []):
            hyp = seed.get("hypothesis", "")
            status = seed.get("status", "unknown")
            conf = seed.get("confidence", 0.0)
            results.append(f"[{status}] {hyp} (confidence: {conf:.2f})")
        if not results:
            results.append("No seeds found.")
        return results

    def get_contradictions(self) -> list[str]:
        """List unresolved contradictions."""
        graph = _load_knowledge_graph(self.data_dir)
        results: list[str] = []
        for c in graph.get("contradictions", []):
            if c.get("resolved", False):
                continue
            a = c.get("concept_a", "")
            b = c.get("concept_b", "")
            desc = c.get("description", f"{a} vs {b}")
            results.append(f"Contradiction: {desc} ({a} ↔ {b})")
        if not results:
            results.append("No unresolved contradictions.")
        return results

    def search_claims(self, query: str) -> list[str]:
        """Full-text search across claims index."""
        claims = _load_claims(self.data_dir)
        q = query.lower()
        results: list[str] = []
        for claim in claims:
            text = claim.get("text", "")
            if q in text.lower():
                speaker = claim.get("speaker_id", "unknown")
                conf = claim.get("confidence", 0.0)
                results.append(f"[{speaker}] {text} (confidence: {conf:.2f})")
        return results

    def get_expert(self, speaker: str) -> list[str]:
        """Get expert profile for a speaker."""
        profile = _load_expert_profile(self.data_dir, speaker)
        if profile is None:
            return [f"No profile found for speaker: {speaker}"]

        topics = profile.get("expertise_topics", {})
        style = profile.get("communication_style", {})
        rapport = profile.get("rapport_with_user", 0.5)
        meetings = profile.get("meetings_count", 0)
        claims = profile.get("total_claims", 0)

        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_str = ", ".join(f"{t}:{s:.2f}" for t, s in top_topics) or "none"
        style_str = ", ".join(f"{k}:{v:.2f}" for k, v in style.items()) or "none"

        return [
            f"Expert: {speaker}",
            f"  Top topics: {topic_str}",
            f"  Communication style: {style_str}",
            f"  Rapport with user: {rapport:.2f}",
            f"  Meetings: {meetings} | Total claims: {claims}",
        ]


# ---------------------------------------------------------------------------
# MCP SDK integration (optional)
# ---------------------------------------------------------------------------

def create_mcp_server(data_dir: str = "data") -> Any:
    """Create and return a configured MCP server.

    Tries to use FastMCP (mcp >= 1.0) first, then falls back to the
    low-level Server API, then to the pure-Python stub if the SDK is
    not installed at all.
    """
    fs = ForgeStreamMCPServer(data_dir=data_dir)

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

        server = FastMCP("forgestream")

        @server.tool()
        async def forgestream_query_knowledge(topic: str) -> str:
            """Search the knowledge graph for concepts matching a topic."""
            return "\n".join(fs.query_knowledge(topic))

        @server.tool()
        async def forgestream_get_requirements() -> str:
            """List all detected requirements."""
            return "\n".join(fs.get_requirements())

        @server.tool()
        async def forgestream_get_seeds() -> str:
            """List all seeds with status."""
            return "\n".join(fs.get_seeds())

        @server.tool()
        async def forgestream_get_contradictions() -> str:
            """List unresolved contradictions."""
            return "\n".join(fs.get_contradictions())

        @server.tool()
        async def forgestream_search_claims(query: str) -> str:
            """Full-text search across claims."""
            return "\n".join(fs.search_claims(query))

        @server.tool()
        async def forgestream_get_expert(speaker: str) -> str:
            """Get expert profile for a speaker."""
            return "\n".join(fs.get_expert(speaker))

        return server

    except (ImportError, AttributeError):
        # mcp SDK not installed or incompatible — return the pure-Python stub
        return fs
