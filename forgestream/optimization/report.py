"""ArchitecturalReport -- markdown report from PerformanceAnalyzer results."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .analyzer import AnalysisResult


class ArchitecturalReport:
    """Generates a markdown architectural analysis report from AnalysisResult."""

    def generate(
        self,
        analysis: AnalysisResult,
        meeting_name: str = "",
        config_overrides: dict | None = None,
    ) -> str:
        """Generate a markdown string summarising the architectural analysis.

        Args:
            analysis: Output of PerformanceAnalyzer.analyze()
            meeting_name: Display name for the report header
            config_overrides: Tuned config dict to embed in the report (optional)
        """
        lines: list[str] = [
            f"# Architectural Analysis: {meeting_name or 'Untitled'}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "",
            "## Bottlenecks",
        ]

        if analysis.bottleneck_count == 0:
            lines.append("- No claim processing gaps > 10s detected.")
        else:
            lines.append(
                f"- {analysis.bottleneck_count} claim processing gap(s) > 10s detected."
            )
            for ts in analysis.bottleneck_timestamps[:10]:
                lines.append(f"  - {ts.strftime('%H:%M:%S')}")

        lines.extend(["", "## Emotion-Quality Correlation"])

        if analysis.emotion_claim_correlation is None:
            lines.append("- No emotion data available for correlation analysis.")
        else:
            r = analysis.emotion_claim_correlation
            strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
            direction = "positive" if r > 0 else "negative"
            lines.append(f"- Engagement-claim density: r={r:.2f} ({strength} {direction})")

            if analysis.high_engagement_confidence_avg is not None:
                lines.append(
                    f"- High-engagement claims: "
                    f"{analysis.high_engagement_confidence_avg:.0%} confidence avg"
                )
            if analysis.low_engagement_confidence_avg is not None:
                lines.append(
                    f"- Low-engagement claims: "
                    f"{analysis.low_engagement_confidence_avg:.0%} confidence avg"
                )
            if analysis.rapport_verification_ratio is not None:
                lines.append(
                    f"- Rapport > 0.6 windows: "
                    f"{analysis.rapport_verification_ratio:.1%} of verified findings"
                )

        lines.extend(["", "## Agent Performance"])

        if analysis.agent_timeout_count == 0:
            lines.append("- No agent timeouts recorded.")
        else:
            lines.append(f"- {analysis.agent_timeout_count} agent timeout(s) detected.")

        lines.append(
            f"- Claims per minute: {analysis.claims_per_minute:.1f}"
        )

        lines.extend(["", "## Config Suggestions"])

        suggestions = self._generate_suggestions(analysis)
        if suggestions:
            for s in suggestions:
                lines.append(f"- {s}")
        else:
            lines.append("- No configuration changes suggested.")

        if config_overrides:
            lines.extend([
                "",
                "## Tuned Config (saved to data/config_overrides.json)",
                "```json",
                json.dumps(config_overrides, indent=2),
                "```",
            ])

        return "\n".join(lines)

    def _generate_suggestions(self, analysis: AnalysisResult) -> list[str]:
        """Derive human-readable config suggestions from analysis results."""
        suggestions: list[str] = []

        if analysis.agent_timeout_count > 0:
            suggestions.append(
                f"scaffold_timeout: increase (detected {analysis.agent_timeout_count} timeout(s))"
            )

        if analysis.bottleneck_count > 2:
            suggestions.append(
                f"spawn_cooldown: consider reducing (detected {analysis.bottleneck_count} bottlenecks)"
            )

        if (
            analysis.emotion_claim_correlation is not None
            and abs(analysis.emotion_claim_correlation) > 0.6
        ):
            suggestions.append(
                "emotion_window_seconds: current settings well-correlated with claim density"
            )

        return suggestions
