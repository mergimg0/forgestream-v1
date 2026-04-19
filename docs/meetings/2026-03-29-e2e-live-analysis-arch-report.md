# Architectural Analysis: e2e-live-analysis
**Date:** 2026-03-29

## Bottlenecks
- No claim processing gaps > 10s detected.

## Emotion-Quality Correlation
- Engagement-claim density: r=0.00 (weak negative)
- Low-engagement claims: 72% confidence avg

## Agent Performance
- No agent timeouts recorded.
- Claims per minute: 7.8

## Config Suggestions
- No configuration changes suggested.

## Tuned Config (saved to data/config_overrides.json)
```json
{
  "spawn_cooldown_seconds": 61.73804632181849,
  "scaffold_timeout_minutes": 10.001556679654716,
  "max_concurrent_research": 3.122580975216671,
  "max_concurrent_scaffold": 4.028593355751249,
  "emotion_stride_seconds": 0.9872217951863091,
  "emotion_window_seconds": 3.0154397266190767
}
```

## Evaluator Weight Sensitivity

Most impactful weight: **verification**

| Weight | Variance | Mean | Range |
|--------|----------|------|-------|
| verification | 0.000666 | 0.2512 | 0.0935 |
| scaffold | 0.000553 | 0.2358 | 0.0866 |
| uptake | 0.000403 | 0.2456 | 0.0697 |
| engagement | 0.000123 | 0.2510 | 0.0447 |
| knowledge | 0.000005 | 0.2472 | 0.0084 |