"""Research agent prompt template and output parsing."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from forgestream.events.schema import Event, EventType


class ResearchTemplate:
    """Builds prompts for research agents and parses their output."""

    def build_prompt(
        self,
        query: str,
        context_claims: list[str],
    ) -> str:
        """Build a research prompt from query and meeting context."""
        claims_text = "\n".join(f"- {c}" for c in context_claims) if context_claims else "None"

        return (
            "You are a PhD-level research agent in the ForgeStream system.\n\n"
            f"TASK: {query}\n\n"
            f"CONTEXT FROM MEETING:\n{claims_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Research this topic thoroughly using web search\n"
            "2. Find primary sources — papers, docs, official references\n"
            "3. Verify claims against sources\n"
            "4. Output your findings as JSON:\n"
            "   {\n"
            '     "query": "...",\n'
            '     "finding": "...",\n'
            '     "sources": [{"url": "...", "title": "..."}],\n'
            '     "verification_chain": "...",\n'
            '     "confidence": 0.0-1.0,\n'
            '     "connections": ["..."],\n'
            '     "growth_vectors": ["..."]\n'
            "   }\n\n"
            "Be rigorous. Cite everything. If you can't verify, say so."
        )

    def parse_output(
        self,
        output: str,
        session_id: UUID,
        branch_id: UUID,
    ) -> Event | None:
        """Parse research agent output into a verified_finding event."""
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None

        return Event(
            event_type=EventType.VERIFIED_FINDING,
            session_id=session_id,
            branch_id=branch_id,
            author="research_agent",
            evaluator=0.0,
            payload={
                "query": data.get("query", ""),
                "finding": data.get("finding", ""),
                "sources": data.get("sources", []),
                "verification_chain": data.get("verification_chain", ""),
                "confidence": data.get("confidence", 0.5),
                "connections": data.get("connections", []),
                "growth_vectors": data.get("growth_vectors", []),
            },
        )
