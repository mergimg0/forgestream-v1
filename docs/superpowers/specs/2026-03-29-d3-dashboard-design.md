# ForgeStream D3 Dashboard — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Depends on:** Emotion Pipeline (complete), Rapport Tracking (complete), Dashboard API (complete)

---

## Overview

Replace the placeholder HTML in `dashboard/server.py` with a fully interactive D3.js dashboard featuring 6 visualization panels: knowledge graph, evaluator trajectory, meeting timeline, emotion timeline, entrainment heatmap, and rapport trajectory. All panels auto-refresh via polling (5s interval), with WebSocket push for real-time updates.

## Architecture

Single-page app rendered inline in `INDEX_HTML` (no build step, no npm). D3 v7 loaded from CDN. Each panel is an ES module class with `fetch()` → `render()` → `update()` pattern. Panels arranged in CSS Grid: 2-column layout, knowledge graph spans full width on first row.

## API Endpoints (All Existing)

| Endpoint | Data | Panel |
|----------|------|-------|
| `/api/graph` | Concepts, edges, requirements, artifacts | Knowledge Graph |
| `/api/evaluator` | E(π) trajectory, axiom status | Evaluator Trajectory |
| `/api/emotion/timeline` | Per-speaker arousal/valence over time | Emotion Timeline |
| `/api/emotion/entrainment` | ENTRAINMENT_SNAPSHOT history | Entrainment Heatmap |
| `/api/emotion/rapport` | RAPPORT_SCORE history | Rapport Trajectory |
| `/api/emotion/speakers` | Per-speaker summary | Speaker Summary (sidebar) |

## Panel Designs

### 1. Knowledge Graph (Force-Directed)

```
┌──────────────────────────────────────────────────────────┐
│ Knowledge Graph                               [🔍 zoom] │
│                                                          │
│      ○ quantum ─── ○ computing                           │
│     ╱    ╲              │                                │
│    ○ error ─── ○ correction ─── ○ surface codes          │
│         ╲                                                │
│          ○ threshold ─── ○ noise                         │
│                                                          │
│ Nodes: sized by confidence, colored by domain            │
│ Edges: weighted by co-occurrence, gray lines             │
│ Click node: show source claims in tooltip                │
│ Drag: reposition nodes                                   │
│ Scroll: zoom                                             │
└──────────────────────────────────────────────────────────┘
```

- **D3 force simulation**: `d3.forceSimulation` with `forceManyBody(-100)`, `forceLink(distance=80)`, `forceCenter`
- **Node size**: `r = 5 + 15 * confidence`
- **Node color**: `d3.scaleOrdinal(d3.schemeTableau10)` by domain
- **Edge width**: `1 + 3 * weight`
- **Interaction**: drag to reposition, click for tooltip, scroll to zoom

### 2. Evaluator Trajectory (Line Chart)

```
┌────────────────────────────┐
│ Evaluator Trajectory       │
│ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─    │
│     ╱╲                     │
│ 0.5 ╱──╲───╱╲─── ─ ─      │
│ 0.0 ─ ─ ─ ─ ─ ─ ─ ─ ─    │
│     0    50   100  events  │
│                            │
│ Axioms: ● ● ● (all green) │
└────────────────────────────┘
```

- **Line**: E(π) over events, `d3.curveMonotoneX`
- **Area fill**: light blue under the line (area chart)
- **Axiom indicators**: 3 colored circles (green=holds, red=violated)
- **Current value**: large text overlay

### 3. Meeting Timeline (Horizontal)

```
┌──────────────────────────────────────────────────────────┐
│ Timeline  ▲claim  ◆req  ★finding  ⚡branch  ●seed       │
│ ─────────▲──▲──▲──◆──▲──▲──★──▲──⚡──▲──▲──●──▲─────── │
│ 0:00          5:00         10:00        15:00            │
└──────────────────────────────────────────────────────────┘
```

- **Event markers**: different shapes per event type, colored
- **X-axis**: time (minutes)
- **Hover**: tooltip with event payload summary
- **Click**: scroll to related knowledge graph node

### 4. Emotion Timeline (Per-Speaker Lines)

```
┌──────────────────────────────────────────────────────────┐
│ Emotion Timeline                                         │
│ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                      │
│      ╱╲  Speaker 0 arousal (orange)                      │
│ 0.5 ╱──╲───╱╲──────── ─ ─ ─ ─                           │
│    ╱     ╲╱   ╲  Speaker 1 arousal (teal)                │
│ 0.0 ─ ─ ─ ─ ─ ─╲─ ─ ─ ─ ─ ─ ─                         │
│    ▲   ▲    ▲      ▲  ← claim markers                   │
│    0s  30s  60s  90s                                     │
└──────────────────────────────────────────────────────────┘
```

- **Per-speaker arousal lines**: colored by speaker, `d3.curveMonotoneX`
- **Claim markers**: small triangles on x-axis at claim timestamps
- **Y-axis**: [0, 1] for arousal
- **Toggle**: button to switch between arousal/valence/dominance

### 5. Entrainment Heatmap

```
┌────────────────────────────┐
│ Entrainment                │
│        sp0   sp1   sp2     │
│ sp0   1.00  0.65  0.42    │
│ sp1   0.65  1.00  0.38    │
│ sp2   0.42  0.38  1.00    │
│                            │
│ Color: ░▒▓█ (0→1)         │
│ Latest snapshot            │
└────────────────────────────┘
```

- **Heatmap cells**: `d3.scaleSequential(d3.interpolateYlOrRd)` for correlation
- **Labels**: speaker IDs on both axes
- **Values**: text inside cells
- **Border**: rounded corners on cells

### 6. Rapport Trajectory (Dual-Axis)

```
┌──────────────────────────────────────────────────────────┐
│ Rapport + Engagement                                     │
│ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─                                  │
│      ╱╲  Rapport composite (gold)                        │
│ 0.5 ╱──╲────╱╲─── ─ ─                                   │
│    ╱     ╲──╱  ╲  E(π) (blue)                            │
│ 0.0 ─ ─ ─ ─ ─ ─ ─                                       │
│    0    1    2    3  (snapshots)                          │
│                                                          │
│ Trend: ↑ +0.03  Components: att=0.72 pos=0.65 ...       │
│ ⚠ Disengaged: (none)                                     │
└──────────────────────────────────────────────────────────┘
```

- **Dual lines**: rapport composite (gold) + E(π) (blue)
- **Trend indicator**: arrow + value
- **Component breakdown**: small text showing latest attentiveness/positivity/coordination/symmetry
- **Disengagement alert**: red text if any speakers flagged

## Layout

```
┌─────────────────────────────────────────────────┐
│                 Knowledge Graph                  │  row 1 (full width)
│              (force-directed D3)                 │
├──────────────────────┬──────────────────────────┤
│  Evaluator           │  Entrainment Heatmap     │  row 2 (2 cols)
│  Trajectory          │                          │
├──────────────────────┴──────────────────────────┤
│              Meeting Timeline                    │  row 3 (full width)
├──────────────────────┬──────────────────────────┤
│  Emotion Timeline    │  Rapport Trajectory      │  row 4 (2 cols)
└──────────────────────┴──────────────────────────┘
```

CSS Grid: `grid-template-columns: 1fr 1fr`, with full-width panels using `grid-column: 1 / -1`.

## Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep navy | `#1a1a2e` |
| Panel background | Dark blue | `#16213e` |
| Panel border | Medium blue | `#0f3460` |
| Accent (headings) | Cyan | `#0ff` |
| E(π) line | Blue | `#3b82f6` |
| Rapport line | Gold | `#f59e0b` |
| Arousal line | Orange | `#f97316` |
| Valence line | Green | `#22c55e` |
| Disengagement alert | Red | `#ef4444` |
| Text | Light gray | `#e5e7eb` |

## Auto-Refresh Strategy

All panels poll their respective API endpoints every 5 seconds:

```javascript
setInterval(() => {
    knowledgeGraph.update();
    evaluator.update();
    timeline.update();
    emotionTimeline.update();
    heatmap.update();
    rapportTrajectory.update();
}, 5000);
```

Future enhancement: WebSocket push via `dashboard/ws.py` for instant updates on new events.

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `forgestream/dashboard/static/js/knowledge-graph.js` | Force-directed D3 knowledge graph |
| `forgestream/dashboard/static/js/evaluator-trajectory.js` | E(π) line chart + axiom indicators |
| `forgestream/dashboard/static/js/meeting-timeline.js` | Horizontal event timeline |
| `forgestream/dashboard/static/js/emotion-timeline.js` | Per-speaker arousal/valence lines |
| `forgestream/dashboard/static/js/entrainment-heatmap.js` | Speaker-pair correlation matrix |
| `forgestream/dashboard/static/js/rapport-trajectory.js` | Rapport composite + E(π) dual-axis |
| `forgestream/dashboard/static/css/dashboard.css` | Full dashboard styles |

### Modified files

| File | Changes |
|------|---------|
| `forgestream/dashboard/server.py` | Replace `INDEX_HTML` with full HTML loading D3 + JS modules |

## Dependencies

- D3 v7: loaded from CDN (`https://d3js.org/d3.v7.min.js`)
- No npm, no build step, no bundler
- Static files served by FastAPI's `StaticFiles` mount
