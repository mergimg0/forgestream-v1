# ForgeStream Architecture Self-Optimization — Design Spec (Theme 4)

**Date:** 2026-03-29
**Status:** Approved
**Depends on:** Emotion Pipeline (complete), Rapport Tracking (complete), PostMeetingSynthesis (complete)

---

## Overview

Post-meeting performance analysis that correlates emotional engagement with extraction quality, identifies pipeline bottlenecks, and GRPO-tunes system configuration parameters (spawn cooldowns, agent timeouts, branch thresholds) across meetings.

## Architecture

```
PostMeetingSynthesis.run()
    │
    ├── (existing) Generate report, tune weights, save corpus
    │
    └── (NEW) PerformanceAnalyzer.analyze(events)
            ├── Bottleneck detection (claim processing latency)
            ├── Emotion-extraction correlation
            ├── Agent timeout analysis
            └── ArchitecturalReport.generate()
                    └── docs/meetings/YYYY-MM-DD-arch-report.md

    └── (NEW) ConfigTuner.tune(current_config, analysis)
            └── data/config_overrides.json
```

## PerformanceAnalyzer

Processes the full event log and computes:

### Bottleneck Detection
- Claim processing latency: time between consecutive CLAIM events (gaps > 10s = bottleneck)
- Synthesis lag: time between CLAIM and derived events (REQUIREMENT, BRANCH_POINT)
- Event throughput: claims per minute, trending over meeting duration

### Emotion-Extraction Correlation
- Pearson r between `emotional_engagement` window averages and claim density in the same window
- Identifies: "Claims extracted during high-engagement windows (arousal > 0.7) had {X}% higher confidence"
- Rapport-quality correlation: does `group_rapport_composite > 0.6` predict higher verification rate?

### Agent Performance (when agents are spawned)
- Timeout rate: what fraction of spawned agents timed out
- Domain correlation: which topic keywords are associated with timeouts (e.g., physics queries take longer)

## ConfigTuner

GRPO-tunes system configuration parameters. Same perturbation-selection-blend algorithm as WeightTuner, applied to a different parameter space:

```python
TUNABLE_PARAMS = {
    "spawn_cooldown_seconds": (10, 120),     # range
    "scaffold_timeout_minutes": (5, 30),
    "max_concurrent_research": (2, 8),
    "max_concurrent_scaffold": (2, 8),
    "emotion_stride_seconds": (0.5, 3.0),
    "emotion_window_seconds": (1.0, 5.0),
}
```

Each parameter is perturbed within its range. The "score" for each perturbation is retrospective: what WOULD the meeting quality have been with these parameters? For timeout-related params, shorter timeouts that still succeed = better (faster). For emotion params, the stride/window that maximizes engagement-claim correlation = better.

Persistence: `data/config_overrides.json`. Loaded at meeting start and merged with ForgeStreamConfig defaults.

## ArchitecturalReport

Markdown report saved alongside the meeting report:

```markdown
# Architectural Analysis: {meeting_name}

## Bottlenecks
- 3 claim processing gaps > 10s (at 5:23, 12:45, 18:02)
- Synthesis lag average: 2.1s (within budget)

## Emotion-Quality Correlation
- Engagement-claim density: r=0.72 (strong positive)
- High-engagement claims: 87% confidence avg vs 62% in low-engagement
- Rapport > 0.6 windows: 2.1x verification rate

## Config Suggestions
- scaffold_timeout: 10 → 15 min (3 timeouts on physics queries)
- spawn_cooldown: 60 → 45s (no resource contention observed)

## Tuned Config (saved to data/config_overrides.json)
{json block}
```

## File Structure

| File | Responsibility | Est. Lines |
|------|---------------|-----------|
| `forgestream/optimization/__init__.py` | Package exports | ~5 |
| `forgestream/optimization/analyzer.py` | `PerformanceAnalyzer` — bottleneck + correlation + agent analysis | ~150 |
| `forgestream/optimization/config_tuner.py` | `ConfigTuner` — GRPO on system config | ~100 |
| `forgestream/optimization/report.py` | `ArchitecturalReport` — markdown generation | ~80 |
| `tests/optimization/test_analyzer.py` | Analyzer tests | ~100 |
| `tests/optimization/test_config_tuner.py` | Config tuner tests | ~60 |

## Modified Files

| File | Changes |
|------|---------|
| `forgestream/post_meeting.py` | Call `PerformanceAnalyzer.analyze()` + `ConfigTuner.tune()` in `run()` |
| `forgestream/config.py` | Add `load_config_overrides()` to merge saved overrides |

## Integration with PostMeetingSynthesis

Added as the final step in `run()`:

```python
# Architecture self-optimization
analyzer = PerformanceAnalyzer()
analysis = analyzer.analyze(events)
report = ArchitecturalReport().generate(analysis, meeting_name)
# Save report
config_tuner = ConfigTuner()
new_config = config_tuner.tune(current_config_params, analysis, human_score)
save_config_overrides(new_config)
result["arch_report_path"] = report_path
result["config_overrides"] = new_config
```
