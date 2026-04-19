# D3 Emotion Dashboard Visualizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three D3.js visualizations to the ForgeStream web dashboard that render emotion data in real-time: emotion timeline (per-speaker arousal/valence tracks), entrainment heatmap (speaker-pair correlation matrix), and engagement trajectory (collective engagement overlaid on E(π)).

**Architecture:** The dashboard already has a FastAPI server (`dashboard/server.py`) serving static files from `dashboard/static/`. The three new API endpoints (`/emotion/timeline`, `/emotion/entrainment`, `/emotion/speakers`) are already implemented. This plan adds JavaScript modules that fetch from those endpoints and render D3 visualizations. WebSocket updates push new data as events arrive.

**Tech Stack:** D3.js v7 (CDN), vanilla JavaScript (no build step), CSS Grid for layout

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `forgestream/dashboard/static/js/emotion-timeline.js` | Per-speaker arousal/valence line chart with claim markers |
| `forgestream/dashboard/static/js/entrainment-heatmap.js` | Speaker-pair correlation matrix, animated over time |
| `forgestream/dashboard/static/js/engagement-trajectory.js` | Collective engagement + E(π) dual-axis line chart |
| `forgestream/dashboard/static/css/emotion.css` | Styles for emotion panels |

### Modified files

| File | Changes |
|------|---------|
| `forgestream/dashboard/static/index.html` | Add emotion panel sections + D3 script imports |
| `forgestream/dashboard/ws.py` | Push PROSODIC_FEATURE and ENTRAINMENT_SNAPSHOT events over WebSocket |

---

## Task 1: Emotion Timeline (Per-Speaker Arousal/Valence)

**Files:**
- Create: `forgestream/dashboard/static/js/emotion-timeline.js`
- Modify: `forgestream/dashboard/static/index.html`

### Design

```
┌──────────────────────────────────────────┐
│ Emotion Timeline                    [▼]  │
│                                          │
│ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│      ╱╲    Speaker 0 (arousal)           │
│ 0.5 ╱  ╲───╱╲──────────────── ─ ─ ─ ─   │
│    ╱     ╲╱   ╲    Speaker 1 (arousal)   │
│ 0.0 ─ ─ ─ ─ ─ ─╲─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│    ▲   ▲    ▲      ▲   ← claim markers  │
│    0s  30s  60s  90s                     │
└──────────────────────────────────────────┘
```

- [ ] **Step 1: Create emotion-timeline.js**

```javascript
// forgestream/dashboard/static/js/emotion-timeline.js
// Fetches /api/emotion/timeline and renders per-speaker arousal/valence
// as D3 line charts with claim event markers on the x-axis.

class EmotionTimeline {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 20, right: 80, bottom: 30, left: 50 };
        this.width = 600 - this.margin.left - this.margin.right;
        this.height = 200 - this.margin.top - this.margin.bottom;
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    }

    async fetch() {
        const resp = await fetch('/api/emotion/timeline');
        return (await resp.json()).timeline;
    }

    render(data) {
        this.container.selectAll('*').remove();
        if (!data.length) return;

        const svg = this.container.append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d.timestamp_ms))
            .range([0, this.width]);

        const y = d3.scaleLinear().domain([0, 1]).range([this.height, 0]);

        svg.append('g').attr('transform', `translate(0,${this.height})`).call(d3.axisBottom(x).tickFormat(d => `${d/1000}s`));
        svg.append('g').call(d3.axisLeft(y));

        // Group by speaker
        const speakers = d3.group(data, d => d.speaker_id);
        for (const [speaker, points] of speakers) {
            const line = d3.line()
                .x(d => x(d.timestamp_ms))
                .y(d => y(d.arousal))
                .curve(d3.curveMonotoneX);

            svg.append('path')
                .datum(points)
                .attr('fill', 'none')
                .attr('stroke', this.colorScale(speaker))
                .attr('stroke-width', 2)
                .attr('d', line);
        }
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
```

- [ ] **Step 2: Commit**

---

## Task 2: Entrainment Heatmap

**Files:**
- Create: `forgestream/dashboard/static/js/entrainment-heatmap.js`

### Design

```
┌──────────────────────────────┐
│ Entrainment Heatmap          │
│        sp0   sp1   sp2       │
│ sp0   1.00  0.65  0.42       │
│ sp1   0.65  1.00  0.38       │
│ sp2   0.42  0.38  1.00       │
│                              │
│ Color: ░░▒▒▓▓██ (0→1)       │
└──────────────────────────────┘
```

- [ ] **Step 1: Create entrainment-heatmap.js**

```javascript
// Fetches /api/emotion/entrainment, renders latest snapshot as a
// speaker-pair correlation heatmap. Updates on WebSocket push.

class EntrainmentHeatmap {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
    }

    async fetch() {
        const resp = await fetch('/api/emotion/entrainment');
        const data = await resp.json();
        return data.snapshots.length ? data.snapshots[data.snapshots.length - 1] : null;
    }

    render(snapshot) {
        this.container.selectAll('*').remove();
        if (!snapshot || !snapshot.speaker_pairs.length) {
            this.container.append('p').text('Waiting for entrainment data...');
            return;
        }

        // Build matrix from speaker pairs
        const speakers = new Set();
        snapshot.speaker_pairs.forEach(p => {
            speakers.add(p.speaker_a);
            speakers.add(p.speaker_b);
        });
        const speakerList = [...speakers].sort();
        const n = speakerList.length;
        const cellSize = 60;

        const svg = this.container.append('svg')
            .attr('width', (n + 1) * cellSize + 40)
            .attr('height', (n + 1) * cellSize + 40);

        const g = svg.append('g').attr('transform', 'translate(60, 40)');

        // Labels
        speakerList.forEach((s, i) => {
            g.append('text').attr('x', i * cellSize + cellSize / 2).attr('y', -8)
                .attr('text-anchor', 'middle').attr('font-size', '11px').text(s);
            g.append('text').attr('x', -8).attr('y', i * cellSize + cellSize / 2 + 4)
                .attr('text-anchor', 'end').attr('font-size', '11px').text(s);
        });

        // Build lookup
        const lookup = {};
        snapshot.speaker_pairs.forEach(p => {
            lookup[`${p.speaker_a}-${p.speaker_b}`] = p.f0_correlation;
            lookup[`${p.speaker_b}-${p.speaker_a}`] = p.f0_correlation;
        });

        // Draw cells
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = i === j ? 1.0 : (lookup[`${speakerList[i]}-${speakerList[j]}`] || 0);
                g.append('rect')
                    .attr('x', j * cellSize).attr('y', i * cellSize)
                    .attr('width', cellSize - 2).attr('height', cellSize - 2)
                    .attr('fill', this.colorScale(Math.abs(val)))
                    .attr('rx', 4);
                g.append('text')
                    .attr('x', j * cellSize + cellSize / 2 - 1)
                    .attr('y', i * cellSize + cellSize / 2 + 4)
                    .attr('text-anchor', 'middle').attr('font-size', '12px')
                    .attr('fill', Math.abs(val) > 0.5 ? '#fff' : '#333')
                    .text(val.toFixed(2));
            }
        }
    }

    async update() {
        const snapshot = await this.fetch();
        this.render(snapshot);
    }
}
```

- [ ] **Step 2: Commit**

---

## Task 3: Engagement Trajectory

**Files:**
- Create: `forgestream/dashboard/static/js/engagement-trajectory.js`

### Design

```
┌──────────────────────────────────────────┐
│ Engagement + E(π) Trajectory             │
│                                          │
│ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │
│      ╱╲  Engagement (orange)             │
│ 0.5 ╱──╲────────╱╲─── ─ ─ ─ ─ ─ ─ ─    │
│    ╱     ╲──────╱  ╲   E(π) (blue)      │
│ 0.0 ─ ─ ─ ─ ─ ─ ─ ─╲─ ─ ─ ─ ─ ─ ─    │
│    0s  30s  60s  90s  120s               │
└──────────────────────────────────────────┘
```

- [ ] **Step 1: Create engagement-trajectory.js**

```javascript
// Dual-axis line chart: collective engagement (from /emotion/entrainment)
// overlaid with E(π) trajectory (from /evaluator).

class EngagementTrajectory {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 20, right: 60, bottom: 30, left: 50 };
        this.width = 600 - this.margin.left - this.margin.right;
        this.height = 200 - this.margin.top - this.margin.bottom;
    }

    async fetch() {
        const [entrainResp, evalResp] = await Promise.all([
            fetch('/api/emotion/entrainment'),
            fetch('/api/evaluator'),
        ]);
        const entrainment = await entrainResp.json();
        const evaluator = await evalResp.json();
        return { entrainment: entrainment.snapshots, evaluator: evaluator.trajectory };
    }

    render({ entrainment, evaluator }) {
        this.container.selectAll('*').remove();

        const svg = this.container.append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        const y = d3.scaleLinear().domain([0, 1]).range([this.height, 0]);
        svg.append('g').call(d3.axisLeft(y));

        // Engagement line (orange)
        if (entrainment.length) {
            const engPoints = entrainment.map((s, i) => ({
                x: i, y: s.group_metrics?.collective_engagement || 0
            }));
            const xEng = d3.scaleLinear().domain([0, engPoints.length - 1]).range([0, this.width]);
            const line = d3.line().x(d => xEng(d.x)).y(d => y(d.y)).curve(d3.curveMonotoneX);
            svg.append('path').datum(engPoints)
                .attr('fill', 'none').attr('stroke', '#f59e0b').attr('stroke-width', 2).attr('d', line);
        }

        // E(π) line (blue)
        if (evaluator.length) {
            const xEval = d3.scaleLinear().domain([0, evaluator.length - 1]).range([0, this.width]);
            const line = d3.line().x((d, i) => xEval(i)).y(d => y(d.evaluator)).curve(d3.curveMonotoneX);
            svg.append('path').datum(evaluator)
                .attr('fill', 'none').attr('stroke', '#3b82f6').attr('stroke-width', 2).attr('d', line);
        }

        // Legend
        const legend = svg.append('g').attr('transform', `translate(${this.width - 120}, 0)`);
        legend.append('line').attr('x1', 0).attr('x2', 20).attr('y1', 0).attr('y2', 0).attr('stroke', '#f59e0b').attr('stroke-width', 2);
        legend.append('text').attr('x', 25).attr('y', 4).attr('font-size', '11px').text('Engagement');
        legend.append('line').attr('x1', 0).attr('x2', 20).attr('y1', 16).attr('y2', 16).attr('stroke', '#3b82f6').attr('stroke-width', 2);
        legend.append('text').attr('x', 25).attr('y', 20).attr('font-size', '11px').text('E(π)');
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
```

- [ ] **Step 2: Commit**

---

## Task 4: HTML Layout + CSS + Auto-Refresh

**Files:**
- Modify: `forgestream/dashboard/static/index.html`
- Create: `forgestream/dashboard/static/css/emotion.css`

- [ ] **Step 1: Add emotion panel HTML**

Add to index.html before closing `</body>`:

```html
<!-- Emotion Panels -->
<section id="emotion-panels" class="emotion-grid">
    <div class="panel">
        <h3>Emotion Timeline</h3>
        <div id="emotion-timeline"></div>
    </div>
    <div class="panel">
        <h3>Entrainment</h3>
        <div id="entrainment-heatmap"></div>
    </div>
    <div class="panel">
        <h3>Engagement + E(π)</h3>
        <div id="engagement-trajectory"></div>
    </div>
</section>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="/static/js/emotion-timeline.js"></script>
<script src="/static/js/entrainment-heatmap.js"></script>
<script src="/static/js/engagement-trajectory.js"></script>
<script>
    const timeline = new EmotionTimeline('emotion-timeline');
    const heatmap = new EntrainmentHeatmap('entrainment-heatmap');
    const trajectory = new EngagementTrajectory('engagement-trajectory');

    // Initial render
    timeline.update();
    heatmap.update();
    trajectory.update();

    // Auto-refresh every 5 seconds
    setInterval(() => {
        timeline.update();
        heatmap.update();
        trajectory.update();
    }, 5000);
</script>
```

- [ ] **Step 2: Add emotion.css**

```css
/* forgestream/dashboard/static/css/emotion.css */
.emotion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px;
}
.emotion-grid .panel {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
}
.emotion-grid .panel:last-child {
    grid-column: 1 / -1;
}
.emotion-grid h3 {
    color: #e0e0e0;
    margin: 0 0 12px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
```

- [ ] **Step 3: Commit**

---

## Task 5: WebSocket Push for Real-Time Updates

**Files:**
- Modify: `forgestream/dashboard/ws.py`

- [ ] **Step 1: Add emotion event types to WS broadcast**

```python
# In the WebSocket handler, add PROSODIC_FEATURE and ENTRAINMENT_SNAPSHOT
# to the list of event types that get pushed to connected clients.
# This enables real-time updates without polling.
```

- [ ] **Step 2: Commit**

---

## Verification

```bash
# Start dashboard
cd /Users/mghome/projects/forgestream
python3 -m forgestream start --dashboard

# Open http://127.0.0.1:8501 — should see emotion panels
# Run a mock meeting to generate data:
python3 -m forgestream.mock_meeting
# Panels should populate with emotion timeline, heatmap, and engagement chart
```
