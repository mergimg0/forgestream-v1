# ForgeStream User Profile Extraction — Design Spec (Theme 5)

**Date:** 2026-03-29
**Status:** Approved
**Depends on:** Emotion Pipeline (complete), Rapport Tracking (complete)

---

## Overview

Builds a persistent user profile across meetings from prosodic features, rapport dynamics, and claim patterns. Captures communication style, topic preferences, engagement signatures, and suggestion responsiveness. The profile enables ForgeStream to adapt its output format, suggestion priorities, and interaction style to the specific user.

## Architecture

```
PostMeetingSynthesis.run()
    │
    └── (NEW) UserProfileExtractor.update(events, profile)
            ├── Communication style analysis
            ├── Engagement signature extraction
            ├── Topic preference tracking
            ├── Suggestion responsiveness scoring
            └── data/user_profile.json (accumulated across meetings)

ForgeStream startup
    │
    └── (NEW) StyleAdapter.adapt(config, profile)
            └── Adjusts report format, suggestion text style, TUI display
```

## UserProfile Dataclass

```python
@dataclass
class UserProfile:
    # Communication style
    avg_arousal: float = 0.5           # baseline emotional expressiveness
    avg_f0_variability: float = 0.0    # pitch dynamism
    preferred_energy: float = 0.1      # typical speaking energy
    expressiveness_score: float = 0.5  # composite: high = animated, low = measured

    # Engagement signature
    peak_engagement_topics: list[str] = field(default_factory=list)  # topics with arousal > 0.7
    disengagement_triggers: list[str] = field(default_factory=list)  # topics during disengagement
    avg_meeting_engagement: float = 0.5
    engagement_trend: float = 0.0      # across meetings: improving or declining

    # Topic preferences
    topic_frequency: dict[str, int] = field(default_factory=dict)    # keyword → count across meetings
    topic_depth: dict[str, float] = field(default_factory=dict)      # keyword → avg claims per topic

    # Rapport affinity
    best_rapport_speakers: list[str] = field(default_factory=list)   # speakers with highest rapport
    rapport_component_weights: dict[str, float] = field(default_factory=dict)  # learned from GRPO

    # Suggestion responsiveness
    suggestion_uptake_rate: float = 0.5
    preferred_priority_level: str = "strategic"  # which priority level gets acted on
    ignored_categories: list[str] = field(default_factory=list)

    # Habits
    avg_meeting_duration_minutes: float = 30.0
    preferred_mode: str = "collaborative"        # most-used meeting mode
    meetings_count: int = 0

    # Meta
    last_updated: str = ""
```

## UserProfileExtractor

Updates the profile after each meeting:

### Communication Style
- Average arousal, F0 std, and energy from PROSODIC_FEATURE events where speaker is the user
- Expressiveness score: `0.4 * norm_arousal + 0.3 * norm_f0_var + 0.3 * norm_energy`
- Updated with EMA (alpha=0.2) across meetings — recent meetings weight more

### Engagement Signature
- Extract topic keywords from CLAIM events that co-occur with high-arousal windows (arousal > 0.7)
- Extract topic keywords from claims during disengagement windows
- Track `avg_meeting_engagement` from the evaluator's `emotional_engagement` metric

### Topic Preferences
- Count keyword frequency across all meetings
- Compute depth: claims per unique topic (high depth = the user digs deep into that topic)

### Rapport Affinity
- From RAPPORT_SCORE pair_scores: which speakers consistently have highest composite with the user
- From GRPO-tuned rapport weights: which components (attentiveness, coordination, etc.) the user's meetings weight most

### Suggestion Responsiveness
- Track which suggestions the user acts on vs ignores (requires TUI interaction tracking — starts with a placeholder 0.5 until Theme 1 wires the feedback loop)

## StyleAdapter

Reads the profile at meeting start and adjusts ForgeStream's behavior:

### Report Format Adaptation
- High expressiveness user → detailed reports with context
- Low expressiveness user → terse bullet-point reports
- Implemented as a `report_style` field: "detailed" | "concise" | "balanced"

### Suggestion Calibration
- Boost priority of topics in `peak_engagement_topics`
- Deprioritize topics in `ignored_categories`
- Adjust priority threshold based on `preferred_priority_level`

### Mode Suggestion
- If the user consistently uses "collaborative" mode, suggest it at startup
- If meeting duration is trending shorter, suggest "extract" mode (faster)

## File Structure

| File | Responsibility | Est. Lines |
|------|---------------|-----------|
| `forgestream/profile/__init__.py` | Package exports | ~5 |
| `forgestream/profile/model.py` | `UserProfile` dataclass | ~60 |
| `forgestream/profile/extractor.py` | `UserProfileExtractor` — builds/updates profile from events | ~150 |
| `forgestream/profile/adaptation.py` | `StyleAdapter` — adjusts config/format based on profile | ~80 |
| `tests/profile/test_model.py` | Profile model tests | ~30 |
| `tests/profile/test_extractor.py` | Extractor tests | ~80 |
| `tests/profile/test_adaptation.py` | Adaptation tests | ~50 |

## Modified Files

| File | Changes |
|------|---------|
| `forgestream/post_meeting.py` | Call `UserProfileExtractor.update()` in `run()` |
| `forgestream/config.py` | Add `user_profile_path` config field |

## Persistence

`data/user_profile.json` — loaded at startup, updated after each meeting. Never reset. Each update uses EMA smoothing (alpha=0.2) for numeric fields and set union for list fields.

```python
def update(self, events: list[Event], current_profile: UserProfile) -> UserProfile:
    """Update profile with data from this meeting. EMA for numeric, union for lists."""
```

## Privacy Note

The user profile contains behavioral patterns (engagement triggers, communication style, topic preferences) that are sensitive. The profile is stored locally only (`data/user_profile.json`), never sent to external services, and never included in Gemini API prompts.
