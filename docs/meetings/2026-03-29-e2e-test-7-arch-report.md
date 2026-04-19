# Architectural Analysis: e2e-test-7
**Date:** 2026-03-29

## Bottlenecks
- 1 claim processing gap(s) > 10s detected.
  - 14:55:22

## Emotion-Quality Correlation
- Engagement-claim density: r=0.00 (weak negative)
- Low-engagement claims: 70% confidence avg

## Agent Performance
- No agent timeouts recorded.
- Claims per minute: 0.7

## Config Suggestions
- No configuration changes suggested.

## Tuned Config (saved to data/config_overrides.json)
```json
{
  "spawn_cooldown_seconds": 62.35541744252988,
  "scaffold_timeout_minutes": 10.224781727827441,
  "max_concurrent_research": 3.1111293863227214,
  "max_concurrent_scaffold": 4.055205062801103,
  "emotion_stride_seconds": 0.9658221902401409,
  "emotion_window_seconds": 2.9696524556553965
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **uptake**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| uptake | 0.001560 | 0.1436 | 0.1316 |
| knowledge | 0.000315 | 0.1490 | 0.0671 |
| scaffold | 0.000218 | 0.1510 | 0.0604 |
| verification | 0.000207 | 0.1480 | 0.0521 |
| engagement | 0.000082 | 0.1459 | 0.0419 |