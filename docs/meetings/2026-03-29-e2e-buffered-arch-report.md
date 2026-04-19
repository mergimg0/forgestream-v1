# Architectural Analysis: e2e-buffered
**Date:** 2026-03-29

## Bottlenecks
- No claim processing gaps > 10s detected.

## Emotion-Quality Correlation
- Engagement-claim density: r=0.00 (weak negative)
- Low-engagement claims: 82% confidence avg

## Agent Performance
- No agent timeouts recorded.
- Claims per minute: 0.5

## Config Suggestions
- No configuration changes suggested.

## Tuned Config (saved to data/config_overrides.json)
```json
{
  "spawn_cooldown_seconds": 61.07498198871066,
  "scaffold_timeout_minutes": 10.405740060988759,
  "max_concurrent_research": 2.9745784399147714,
  "max_concurrent_scaffold": 4.028998867390591,
  "emotion_stride_seconds": 1.0373499161553121,
  "emotion_window_seconds": 2.9776563076104874
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **knowledge**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| knowledge | 0.004237 | 0.3362 | 0.2347 |
| scaffold | 0.001462 | 0.3709 | 0.1369 |
| verification | 0.001139 | 0.3573 | 0.1105 |
| uptake | 0.000165 | 0.3636 | 0.0533 |
| engagement | 0.000064 | 0.3611 | 0.0315 |