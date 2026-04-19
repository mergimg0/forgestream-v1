# Architectural Analysis: e2e-test-3
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
  "spawn_cooldown_seconds": 60.61228518863253,
  "scaffold_timeout_minutes": 10.094221519456603,
  "max_concurrent_research": 2.8859165062522427,
  "max_concurrent_scaffold": 3.9917735076472827,
  "emotion_stride_seconds": 0.9633593651875901,
  "emotion_window_seconds": 2.9834528604574317
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **uptake**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| uptake | 0.000978 | 0.1785 | 0.0987 |
| engagement | 0.000512 | 0.1930 | 0.0852 |
| verification | 0.000482 | 0.1949 | 0.0673 |
| scaffold | 0.000350 | 0.1862 | 0.0857 |
| knowledge | 0.000223 | 0.1883 | 0.0525 |