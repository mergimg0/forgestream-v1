# Architectural Analysis: e2e-test-2
**Date:** 2026-03-29

## Bottlenecks
- No claim processing gaps > 10s detected.

## Emotion-Quality Correlation
- No emotion data available for correlation analysis.

## Agent Performance
- No agent timeouts recorded.
- Claims per minute: 0.0

## Config Suggestions
- No configuration changes suggested.

## Tuned Config (saved to data/config_overrides.json)
```json
{
  "spawn_cooldown_seconds": 61.88998919937421,
  "scaffold_timeout_minutes": 10.181731203288297,
  "max_concurrent_research": 3.049273843610084,
  "max_concurrent_scaffold": 4.0752390010557,
  "emotion_stride_seconds": 1.1220137798792196,
  "emotion_window_seconds": 3.063616160171729
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **uptake**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| uptake | 0.000888 | 0.1882 | 0.1121 |
| engagement | 0.000797 | 0.1888 | 0.0888 |
| verification | 0.000386 | 0.1775 | 0.0715 |
| scaffold | 0.000302 | 0.1889 | 0.0689 |
| knowledge | 0.000297 | 0.1753 | 0.0724 |