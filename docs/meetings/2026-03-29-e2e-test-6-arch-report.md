# Architectural Analysis: e2e-test-6
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
  "spawn_cooldown_seconds": 61.1708218290162,
  "scaffold_timeout_minutes": 10.282199015931415,
  "max_concurrent_research": 2.9272011175507693,
  "max_concurrent_scaffold": 4.157404791700098,
  "emotion_stride_seconds": 1.0394092411048592,
  "emotion_window_seconds": 3.0343868940956664
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **engagement**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| engagement | 0.000771 | 0.1861 | 0.1005 |
| uptake | 0.000570 | 0.1997 | 0.0914 |
| verification | 0.000303 | 0.1974 | 0.0546 |
| scaffold | 0.000280 | 0.1963 | 0.0696 |
| knowledge | 0.000264 | 0.2051 | 0.0585 |