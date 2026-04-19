# Architectural Analysis: e2e-test-5
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
  "spawn_cooldown_seconds": 63.36852193007341,
  "scaffold_timeout_minutes": 9.844525178166641,
  "max_concurrent_research": 2.990517703937414,
  "max_concurrent_scaffold": 3.9026502247449857,
  "emotion_stride_seconds": 0.9327149587827893,
  "emotion_window_seconds": 2.947672870642533
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **engagement**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| engagement | 0.000664 | 0.1880 | 0.0805 |
| uptake | 0.000491 | 0.1924 | 0.0779 |
| scaffold | 0.000301 | 0.1955 | 0.0781 |
| verification | 0.000220 | 0.1908 | 0.0552 |
| knowledge | 0.000213 | 0.1892 | 0.0572 |