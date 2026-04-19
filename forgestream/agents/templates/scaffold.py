"""Scaffold agent prompt template and output parsing."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from forgestream.events.schema import Event, EventType


class ScaffoldTemplate:
    """Builds prompts for scaffold agents and parses their output."""

    def build_prompt(
        self,
        requirement: str,
        domain: str,
        verified_findings: list[str],
    ) -> str:
        """Build a scaffold prompt from requirement and research context."""
        findings_text = (
            "\n".join(f"- {f}" for f in verified_findings)
            if verified_findings
            else "None"
        )

        return (
            "You are a scaffold agent in the ForgeStream system.\n\n"
            f"REQUIREMENT:\n{requirement}\n\n"
            f"DOMAIN: {domain}\n\n"
            f"LINKED KNOWLEDGE:\n{findings_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create the project structure appropriate for this domain\n"
            "2. Implement a working V1 — not stubs, not TODOs, working code\n"
            "3. Include tests that validate the core requirement\n"
            "4. Write a DESIGN.md explaining architectural decisions\n"
            "5. When done, output a JSON summary:\n"
            "   {\n"
            '     "files_created": ["..."],\n'
            '     "compiles": true/false,\n'
            '     "tests_pass": true/false,\n'
            '     "design_decisions": ["..."],\n'
            '     "open_questions": ["..."],\n'
            '     "estimated_completeness": 0.0-1.0\n'
            "   }\n\n"
            "Build for the requirement. Nothing more, nothing less."
        )

    def parse_output(
        self,
        output: str,
        session_id: UUID,
        branch_id: UUID,
    ) -> tuple[Event, list[Event]] | None:
        """Parse scaffold agent output into artifact + suggestion events."""
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None

        artifact = Event(
            event_type=EventType.ARTIFACT,
            session_id=session_id,
            branch_id=branch_id,
            author="scaffold_agent",
            evaluator=0.0,
            payload={
                "worktree_path": "",
                "branch_name": "",
                "files_created": data.get("files_created", []),
                "compiles": data.get("compiles", False),
                "tests_pass": data.get("tests_pass", False),
                "design_decisions": data.get("design_decisions", []),
                "estimated_completeness": data.get("estimated_completeness", 0.0),
            },
        )

        suggestions = []
        for question in data.get("open_questions", []):
            suggestions.append(
                Event(
                    event_type=EventType.SUGGESTION,
                    session_id=session_id,
                    branch_id=branch_id,
                    author="scaffold_agent",
                    evaluator=0.0,
                    payload={
                        "text": question,
                        "priority": 0.4,
                        "category": "good_to_probe",
                        "linked_events": [str(artifact.id)],
                        "decay_rate": 0.02,
                    },
                )
            )

        return artifact, suggestions
