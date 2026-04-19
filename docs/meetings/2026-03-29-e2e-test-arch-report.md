# Architectural Analysis: e2e-test
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
  "spawn_cooldown_seconds": 62.40898564803625,
  "scaffold_timeout_minutes": 10.131530705031725,
  "max_concurrent_research": 2.8994332925244493,
  "max_concurrent_scaffold": 4.033624560496049,
  "emotion_stride_seconds": 1.0279921881821883,
  "emotion_window_seconds": 2.946430965820912
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **engagement**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| engagement | 0.001300 | 0.1624 | 0.1321 |
| uptake | 0.000645 | 0.1690 | 0.0935 |
| scaffold | 0.000284 | 0.1618 | 0.0669 |
| knowledge | 0.000254 | 0.1629 | 0.0705 |
| verification | 0.000246 | 0.1743 | 0.0591 |