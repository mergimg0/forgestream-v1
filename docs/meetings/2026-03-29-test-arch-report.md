# Architectural Analysis: test
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
  "spawn_cooldown_seconds": 58.38698404555744,
  "scaffold_timeout_minutes": 9.925663259280144,
  "max_concurrent_research": 3.0507454921684545,
  "max_concurrent_scaffold": 4.04764371450856,
  "emotion_stride_seconds": 1.0598678975621585,
  "emotion_window_seconds": 2.9408710671847262
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **engagement**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| engagement | 0.000781 | 0.1991 | 0.1043 |
| knowledge | 0.000718 | 0.1735 | 0.1143 |
| scaffold | 0.000452 | 0.1972 | 0.0772 |
| uptake | 0.000278 | 0.1913 | 0.0550 |
| verification | 0.000229 | 0.1923 | 0.0616 |