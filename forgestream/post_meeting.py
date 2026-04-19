"""Post-meeting synthesis -- GRPO weight tuning, reports, knowledge persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ForgeStreamConfig
from .emotion.buffer import AudioRingBuffer
from .emotion.emotion2vec_client import Emotion2VecClient
from .emotion.persistence import EmotionCorpus
from .events.schema import Event, EventType
from .emotion.rapport import EARLY_WEIGHTS, ESTABLISHED_WEIGHTS, interpolate_weights
from .gemini.prompt_tuner import PromptParams, PromptTuner
from .governor.evaluator import Evaluator
from .governor.improvement import MeetingSynthesizer, PromptEvolution, WeightTuner
from .governor.sensitivity import WeightSensitivityAnalyzer
from .governor.tone_tuner import ToneAdjustmentTuner
from .optimization import ArchitecturalReport, ConfigTuner, PerformanceAnalyzer
from .profile.expert import ExpertProfileManager
from .profile.extractor import UserProfileExtractor
from .profile.model import UserProfile
from .synthesis.meta_gleanings import MetaGleaningEngine
from .synthesis.proof_obligations import ProofObligationDetector


class PostMeetingSynthesis:
    """Runs after each meeting to improve the system.

    Phase 1: Generate meeting report
    Phase 2: Tune evaluator weights (GRPO)
    Phase 3: Score and evolve prompts
    Phase 4: Update trust region
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        data_dir: str | None = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir or "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.synthesizer = MeetingSynthesizer()
        self.weight_tuner = WeightTuner()
        self.prompt_evolution = PromptEvolution()
        self.prompt_tuner = PromptTuner()
        self.evaluator = Evaluator()
        self.tone_tuner = ToneAdjustmentTuner()
        self.corpus = EmotionCorpus(
            corpus_dir=str(self.data_dir / "emotion_corpus")
        )
        self.performance_analyzer = PerformanceAnalyzer()
        self.config_tuner = ConfigTuner()
        self.profile_extractor = UserProfileExtractor()
        self.expert_profile_manager = ExpertProfileManager(
            profiles_dir=str(self.data_dir / "expert_profiles")
        )
        self.meta_gleaning_engine = MetaGleaningEngine()
        self.sensitivity_analyzer = WeightSensitivityAnalyzer()
        self.emotion2vec_client = Emotion2VecClient(
            runpod_endpoint=config.runpod_crqa_endpoint,
        )

    def generate_report(self, events: list[Event], meeting_name: str = "") -> str:
        """Generate a markdown meeting report."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]

        e_final = events[-1].evaluator if events else 0.0

        lines = [
            f"# Meeting: {meeting_name or 'Untitled'}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"**Claims:** {len(claims)}",
            f"**E(pi) final:** {e_final:.3f}",
            "",
            "## Knowledge Extracted",
            f"- {len(claims)} claims captured",
            f"- {len(findings)} verified findings",
            f"- {len(requirements)} requirements detected",
            f"- {len(artifacts)} artifacts produced",
            "",
        ]

        if requirements:
            lines.append("## Requirements")
            for r in requirements[:15]:
                lines.append(f"- {r.payload.get('description', 'N/A')[:80]}")
            lines.append("")

        if artifacts:
            lines.append("## Artifacts")
            for a in artifacts[:10]:
                compiles = a.payload.get("compiles", False)
                tests = a.payload.get("tests_pass", False)
                files = a.payload.get("files_created", [])
                status = "pass" if compiles and tests else "partial" if compiles else "fail"
                lines.append(f"- [{status}] {len(files)} files")
            lines.append("")

        lines.extend([
            "## SOS Status",
            "- Evaluator weights: see data/weights.json",
            f"- Final E(pi): {e_final:.3f}",
            "",
            "## Human Review Queue",
            "- [ ] Review detected requirements",
            "- [ ] Check scaffold artifacts",
            "- [ ] Promote or archive seeds",
        ])

        return "\n".join(lines)

    def save_report(self, events: list[Event], meeting_name: str = "") -> str:
        """Save meeting report to docs/meetings/."""
        report = self.generate_report(events, meeting_name)
        meetings_dir = Path(self.config.meetings_dir)
        meetings_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = meeting_name.lower().replace(" ", "-")[:30] if meeting_name else "meeting"
        filename = f"{date_str}-{slug}.md"
        path = meetings_dir / filename
        path.write_text(report)
        return str(path)

    def load_meeting_count(self) -> int:
        """Load meeting count from disk."""
        weights_file = self.data_dir / "weights.json"
        if weights_file.exists():
            data = json.loads(weights_file.read_text())
            return data.get("meeting_count", 1)
        return 1

    def load_weights(self) -> dict[str, float]:
        """Load evaluator weights from disk."""
        weights_file = self.data_dir / "weights.json"
        if weights_file.exists():
            data = json.loads(weights_file.read_text())
            valid_keys = {"knowledge", "verification", "scaffold", "uptake", "engagement"}
            loaded = {k: v for k, v in data.items() if k in valid_keys}
            # Backward compat: merge with defaults for any missing keys
            for k, v in Evaluator.DEFAULT_WEIGHTS.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
        return Evaluator.DEFAULT_WEIGHTS.copy()

    def load_rapport_weights(self, meeting_count: int = 1) -> dict[str, float]:
        """Load rapport component weights from disk, merged with interpolated defaults."""
        rapport_file = self.data_dir / "rapport_weights.json"
        base = interpolate_weights(meeting_count)
        if rapport_file.exists():
            data = json.loads(rapport_file.read_text())
            valid_keys = {"attentiveness", "positivity", "coordination", "symmetry"}
            loaded = {k: v for k, v in data.items() if k in valid_keys}
            for k in base:
                if k not in loaded:
                    loaded[k] = base[k]
            return loaded
        return base

    @staticmethod
    def _update_index_with_emotion2vec(
        index_path: str, labels: list[dict]
    ) -> None:
        """Merge emotion2vec labels into the feature index JSON."""
        import json as _json
        path = Path(index_path)
        if not path.exists():
            return
        index = _json.loads(path.read_text())
        entries = index.get("features", index if isinstance(index, list) else [])

        # Build offset→label lookup
        label_map = {l["offset_ms"]: l for l in labels if "offset_ms" in l}

        for entry in entries:
            ts = entry.get("timestamp_ms", entry.get("offset_ms", -1))
            # Find closest label within 1.5s window
            closest = None
            min_dist = 1500
            for offset, label in label_map.items():
                dist = abs(ts - offset)
                if dist < min_dist:
                    min_dist = dist
                    closest = label
            if closest:
                entry["emotion2vec_tag"] = closest["tag"]
                entry["emotion2vec_confidence"] = closest["confidence"]
                entry["emotion2vec_scores"] = closest.get("scores", {})

        path.write_text(_json.dumps(index, indent=2))

    def save_rapport_weights(
        self, weights: dict[str, float], meeting_count: int = 0
    ) -> None:
        """Save rapport component weights to disk."""
        data = {
            **weights,
            "meeting_count": meeting_count,
            "last_tuned": datetime.now(timezone.utc).isoformat(),
        }
        path = self.data_dir / "rapport_weights.json"
        path.write_text(json.dumps(data, indent=2))

    def tune_rapport_weights(
        self,
        events: list[Event],
        meeting_count: int = 1,
    ) -> dict[str, float]:
        """GRPO-tune rapport component weights toward maturity-interpolated targets.

        Note: human_score was removed from this signature. Rapport weights optimize
        toward the theoretically-motivated sigmoid targets via tune_multi_objective,
        not toward a claim-based score.
        """
        current = self.load_rapport_weights(meeting_count)
        targets = interpolate_weights(meeting_count)
        return self.weight_tuner.tune_multi_objective(
            current, events, component_targets=targets,
        )

    def save_weights(
        self, weights: dict[str, float], meeting_count: int = 0
    ) -> None:
        """Save evaluator weights to disk."""
        data = {
            **weights,
            "meeting_count": meeting_count,
            "last_tuned": datetime.now(timezone.utc).isoformat(),
        }
        weights_file = self.data_dir / "weights.json"
        weights_file.write_text(json.dumps(data, indent=2))

        history_file = self.data_dir / "weights_history.json"
        history: list = []
        if history_file.exists():
            history = json.loads(history_file.read_text())
        history.append(data)
        # Cap history to last 200 entries to prevent unbounded growth
        if len(history) > 200:
            history = history[-200:]
        history_file.write_text(json.dumps(history, indent=2))

    def tune_weights(
        self,
        events: list[Event],
        human_score: float | None = None,
    ) -> dict[str, float]:
        """Run GRPO weight tuning against meeting events."""
        current = self.load_weights()
        target = human_score if human_score is not None else self.compute_auto_score(events)
        return self.weight_tuner.tune(current, events, human_score=target)

    def compute_auto_score(self, events: list[Event]) -> float:
        """Compute automatic meeting quality score."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]
        prosodic = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]

        req_scaffold = (
            len(artifacts) / max(len(requirements), 1)
            if requirements else 0.0
        )
        findings_per_claim = (
            len(findings) / max(len(claims), 1)
            if claims else 0.0
        )

        # Engagement signal from prosodic features
        if prosodic:
            mean_arousal = sum(
                e.payload.get("arousal", 0.5) for e in prosodic
            ) / len(prosodic)
            engagement_bonus = 0.1 * mean_arousal
        else:
            engagement_bonus = 0.0

        base_score = (
            0.35 * min(1.0, req_scaffold)
            + 0.35 * min(1.0, findings_per_claim)
            + 0.30 * engagement_bonus * 10.0  # scale arousal (0-0.1) to (0-0.3)
        )
        return min(1.0, max(0.0, base_score))

    def save_config_overrides(self, overrides: dict) -> str:
        """Persist tuned config parameters to data/config_overrides.json."""
        path = self.data_dir / "config_overrides.json"
        path.write_text(json.dumps(overrides, indent=2))
        return str(path)

    def _current_config_params(self) -> dict:
        """Extract tunable config parameters from the active ForgeStreamConfig."""
        return {
            "spawn_cooldown_seconds": float(self.config.spawn_cooldown_seconds),
            "scaffold_timeout_minutes": float(self.config.scaffold_timeout_minutes),
            "max_concurrent_research": float(self.config.max_concurrent_research),
            "max_concurrent_scaffold": float(self.config.max_concurrent_scaffold),
            "emotion_stride_seconds": float(self.config.emotion_stride_seconds),
            "emotion_window_seconds": float(self.config.emotion_window_seconds),
        }

    def save_arch_report(
        self, arch_report: str, meeting_name: str = ""
    ) -> str:
        """Save architectural analysis report alongside the meeting report."""
        meetings_dir = Path(self.config.meetings_dir)
        meetings_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = meeting_name.lower().replace(" ", "-")[:30] if meeting_name else "meeting"
        filename = f"{date_str}-{slug}-arch-report.md"
        path = meetings_dir / filename
        path.write_text(arch_report)
        return str(path)

    async def run(
        self,
        events: list[Event],
        meeting_name: str = "",
        human_score: float | None = None,
        audio_buffer: AudioRingBuffer | None = None,
    ) -> dict[str, Any]:
        """Run the full post-meeting synthesis pipeline."""
        meeting_count = self.load_meeting_count() + 1
        report_path = self.save_report(events, meeting_name)
        new_weights = self.tune_weights(events, human_score)
        self.save_weights(new_weights, meeting_count=meeting_count)
        e_meso = self.evaluator.compute(events)

        # Consensus claim deduplication (reduces GRPO noise from extraction variance)
        from .claims.consensus import build_consensus
        claim_events = [e for e in events if e.event_type == EventType.CLAIM]
        if claim_events:
            run1 = [
                {
                    "text": e.payload.get("text", ""),
                    "topic_keywords": e.payload.get("topic_keywords", []),
                    "confidence": e.payload.get("confidence", 0.5),
                }
                for e in claim_events
            ]
            consensus_claims = build_consensus([run1], jaccard_threshold=0.5, min_runs=1)
        else:
            consensus_claims = []

        # Weight sensitivity analysis — which weight most impacts E(pi)?
        sensitivity_result = self.sensitivity_analyzer.analyze(events, new_weights)

        result: dict[str, Any] = {
            "report_path": report_path,
            "weights": new_weights,
            "e_meso": e_meso,
            "meeting_count": meeting_count,
            "weight_sensitivity": sensitivity_result,
            "consensus_claim_count": len(consensus_claims),
            "raw_claim_count": len(claim_events),
        }

        # Emotion corpus persistence
        if audio_buffer is not None:
            session_id = meeting_name or "unnamed"
            prosodic = [e for e in events if e.event_type == EventType.PROSODIC_FEATURE]
            claims = [e for e in events if e.event_type == EventType.CLAIM]
            result["corpus_audio_path"] = self.corpus.save_meeting_audio(
                session_id, audio_buffer
            )
            result["corpus_index_path"] = self.corpus.save_feature_index(
                session_id, prosodic, claims
            )

            # Offline emotion2vec re-labeling (real probabilities, not hardcoded)
            try:
                raw_audio = audio_buffer.read_window(duration_seconds=600.0)
                if raw_audio and len(raw_audio) > 32000:  # >1s of audio
                    labels = await self.emotion2vec_client.classify_segments(raw_audio)
                    result["emotion2vec_labels"] = labels
                    if labels and result["corpus_index_path"]:
                        self._update_index_with_emotion2vec(
                            result["corpus_index_path"], labels
                        )
                else:
                    result["emotion2vec_labels"] = []
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "emotion2vec offline labeling skipped: %s", exc
                )
                result["emotion2vec_labels"] = []
        else:
            result["corpus_audio_path"] = None
            result["corpus_index_path"] = None

        # Tone adjustment tuning
        target = human_score if human_score is not None else self.compute_auto_score(events)
        result["tone_adjustments"] = self.tone_tuner.tune(
            self.tone_tuner.DEFAULT_ADJUSTMENTS, events, human_score=target
        )

        # Rapport weight tuning
        rapport_weights = self.tune_rapport_weights(events, meeting_count)
        self.save_rapport_weights(rapport_weights, meeting_count)
        result["rapport_weights"] = rapport_weights

        # User profile extraction (Theme 5)
        profile_path = self.config.user_profile_path
        current_profile = UserProfile.load(profile_path)
        updated_profile = self.profile_extractor.update(events, current_profile)
        updated_profile.save(profile_path)
        result["user_profile_path"] = profile_path

        # Expert profile update (Task 5)
        expert_profiles = self.expert_profile_manager.update_from_events(events)
        result["expert_profiles_updated"] = len(expert_profiles)

        # Architecture self-optimization (Theme 4)
        analysis = self.performance_analyzer.analyze(events)
        current_params = self._current_config_params()
        config_score = human_score if human_score is not None else self.compute_auto_score(events)
        new_config = self.config_tuner.tune(current_params, analysis, human_score=config_score)
        config_overrides_path = self.save_config_overrides(new_config)
        arch_report_text = ArchitecturalReport().generate(analysis, meeting_name, new_config)

        # Append weight sensitivity section to the architectural report
        sensitivity_lines = ["\n## Evaluator Weight Sensitivity\n"]
        most_impactful = sensitivity_result.get("most_impactful")
        sensitivity_lines.append(
            f"Most impactful weight: **{most_impactful}**\n"
        )
        sensitivity_lines.append("| Weight | Variance | Mean | Range |")
        sensitivity_lines.append("|--------|----------|------|-------|")
        for wkey, wstats in sensitivity_result.get("sensitivities", {}).items():
            sensitivity_lines.append(
                f"| {wkey} | {wstats['variance']:.6f} | {wstats['mean']:.4f} | {wstats['range']:.4f} |"
            )
        arch_report_text = arch_report_text + "\n" + "\n".join(sensitivity_lines)
        arch_report_path = self.save_arch_report(arch_report_text, meeting_name)

        result["arch_report_path"] = arch_report_path
        result["config_overrides"] = new_config
        result["config_overrides_path"] = config_overrides_path
        result["bottleneck_count"] = analysis.bottleneck_count
        result["emotion_claim_correlation"] = analysis.emotion_claim_correlation

        # Proof obligation export (ProofForge integration)
        from .orchestrator import EventBus as _EventBus
        _bus = _EventBus()
        proof_detector = ProofObligationDetector(event_bus=_bus)
        for ev in events:
            if ev.event_type == EventType.CLAIM:
                await proof_detector.on_event(ev)
        proof_path = proof_detector.save_obligations(str(self.data_dir))
        result["proof_obligations_path"] = proof_path

        # Prompt template tuning (Task 2)
        prompt_params_path = str(self.data_dir / "prompt_params.json")
        try:
            current_prompt_params = PromptParams.load(prompt_params_path)
        except (FileNotFoundError, OSError):
            current_prompt_params = PromptParams()
        tuned_prompt_params = self.prompt_tuner.tune(
            current_prompt_params, events,
            human_score=human_score if human_score is not None else self.compute_auto_score(events),
        )
        tuned_prompt_params.save(prompt_params_path)
        result["prompt_params"] = tuned_prompt_params.to_dict()
        result["prompt_params_path"] = prompt_params_path

        # Meta-gleanings (Task 3)
        gleanings = self.meta_gleaning_engine.analyze(events)
        gleanings_path = str(self.data_dir / "meta_gleanings.json")
        Path(gleanings_path).write_text(
            json.dumps([g.to_dict() for g in gleanings], indent=2)
        )
        result["meta_gleanings"] = [g.to_dict() for g in gleanings]
        result["meta_gleanings_path"] = gleanings_path

        # Add meta-gleanings section to the meeting report
        if gleanings:
            report_content = Path(result["report_path"]).read_text()
            gleaning_lines = ["\n## Meta-Gleanings\n"]
            for g in gleanings:
                gleaning_lines.append(
                    f"- **{g.gleaning_type}** (conf={g.confidence:.2f}): {g.description}"
                )
            Path(result["report_path"]).write_text(
                report_content + "\n".join(gleaning_lines)
            )

        return result
