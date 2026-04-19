"""Build context injection summaries for Gemini Live API."""

from __future__ import annotations

from forgestream.graph.model import KnowledgeGraph


class ContextBuilder:
    """Generates concise summaries of the knowledge graph for Gemini context injection."""

    def build_injection(
        self,
        graph: KnowledgeGraph,
        active_branches: list[str],
    ) -> str:
        """Build a context injection string for Gemini.

        Injected every ~10 minutes to fight context decay.
        """
        parts = ["Current knowledge state:"]

        concepts = graph.concepts
        verified = [c for c in concepts if c.verified]
        parts.append(
            f"- {len(verified)} verified concepts, "
            f"{len(concepts) - len(verified)} unverified"
        )

        if verified:
            names = ", ".join(c.name for c in verified[:10])
            parts.append(f"- Verified: {names}")

        reqs = graph.requirements
        if reqs:
            parts.append(f"- {len(reqs)} requirements detected:")
            for r in reqs[:5]:
                parts.append(f"  - {r.description} (status: {r.status.value})")

        arts = graph.artifacts
        if arts:
            compiling = sum(1 for a in arts if a.compiles)
            parts.append(f"- {len(arts)} scaffolds ({compiling} compiling)")

        if active_branches:
            parts.append(f"- Active branches: {', '.join(active_branches)}")

        parts.append(
            "\nContinue extracting claims. "
            "Flag anything that relates to or contradicts the above."
        )

        return "\n".join(parts)
