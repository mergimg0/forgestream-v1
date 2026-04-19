/**
 * AutonomyProgression — epsilon history line chart across meetings.
 *
 * Fetches /api/autonomy-progression and renders:
 *   - D3 line chart of ε over meeting number
 *   - Horizontal threshold line at 0.6 (AUTO_SPAWN)
 *   - Horizontal threshold line at 0.7 (BRANCH_AUTO_ALLOCATE)
 *   - Projected trajectory (dashed line) to autonomy
 *   - Current ε as large text overlay
 */
class AutonomyProgression {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 28, right: 30, bottom: 40, left: 52 };
    }

    async fetch() {
        try {
            const resp = await fetch('/api/autonomy-progression');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (err) {
            console.warn('[AutonomyProgression] fetch failed:', err);
            return {
                history: [],
                current_epsilon: 0.525,
                auto_spawn_threshold: 0.6,
                branch_auto_allocate_threshold: 0.7,
                slope_per_meeting: 0.0,
                predicted_auto_spawn_meeting: null,
                predicted_branch_auto_meeting: null,
            };
        }
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 420, height: 200 };
        const rect = el.getBoundingClientRect();
        return {
            width: (rect.width > 0 ? rect.width : 420) - this.margin.left - this.margin.right,
            height: 190,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();

        const history = data.history || [];
        const currentEpsilon = data.current_epsilon || 0.525;
        const autoSpawnThreshold = data.auto_spawn_threshold || 0.6;
        const branchAutoThreshold = data.branch_auto_allocate_threshold || 0.7;
        const slope = data.slope_per_meeting || 0.0;
        const predictedAutoSpawn = data.predicted_auto_spawn_meeting;
        const predictedBranchAuto = data.predicted_branch_auto_meeting;

        if (history.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('No meeting history yet — \u03b5 will be tracked after the first save.');
            return;
        }

        const { width, height } = this._getSize();

        const svg = this.container
            .append('svg')
            .attr('width', '100%')
            .attr('height', height + this.margin.top + this.margin.bottom)
            .attr('viewBox', `0 0 ${width + this.margin.left + this.margin.right} ${height + this.margin.top + this.margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');

        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Determine x-domain: extend to predicted meeting if available
        const lastMeeting = history[history.length - 1].meeting;
        const projectedEnd = Math.max(
            lastMeeting + 3,
            predictedBranchAuto ? predictedBranchAuto + 1 : lastMeeting + 3,
            predictedAutoSpawn ? predictedAutoSpawn + 1 : lastMeeting + 3,
        );
        const firstMeeting = history[0].meeting;

        const x = d3.scaleLinear()
            .domain([firstMeeting, projectedEnd])
            .range([0, width]);

        const epsilonValues = history.map(d => d.epsilon);
        const yMin = Math.max(0, Math.min(...epsilonValues) - 0.05);
        const yMax = Math.min(1.0, Math.max(branchAutoThreshold + 0.08, Math.max(...epsilonValues) + 0.05));

        const y = d3.scaleLinear()
            .domain([yMin, yMax])
            .range([height, 0]);

        // Grid lines
        g.append('g')
            .attr('class', 'grid')
            .call(
                d3.axisLeft(y)
                    .tickSize(-width)
                    .tickFormat('')
                    .ticks(4)
            );

        // X axis
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(Math.min(lastMeeting, 8)).tickFormat(d => `M${d}`));

        // X axis label
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height + 34)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '11px')
            .text('meeting');

        // Y axis
        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(2)));

        // --- Threshold lines ---
        // AUTO_SPAWN threshold at 0.6
        g.append('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', y(autoSpawnThreshold)).attr('y2', y(autoSpawnThreshold))
            .attr('stroke', '#f59e0b')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '6,3');

        g.append('text')
            .attr('x', width - 2)
            .attr('y', y(autoSpawnThreshold) - 4)
            .attr('text-anchor', 'end')
            .attr('fill', '#f59e0b')
            .attr('font-size', '9px')
            .text('AUTO_SPAWN \u03b5=' + autoSpawnThreshold.toFixed(1));

        // BRANCH_AUTO_ALLOCATE threshold at 0.7
        g.append('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', y(branchAutoThreshold)).attr('y2', y(branchAutoThreshold))
            .attr('stroke', '#a78bfa')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '6,3');

        g.append('text')
            .attr('x', width - 2)
            .attr('y', y(branchAutoThreshold) - 4)
            .attr('text-anchor', 'end')
            .attr('fill', '#a78bfa')
            .attr('font-size', '9px')
            .text('BRANCH_AUTO \u03b5=' + branchAutoThreshold.toFixed(1));

        // --- Projected trajectory (dashed) ---
        if (slope > 0 && history.length >= 2) {
            const lastEps = history[history.length - 1].epsilon;
            const projectedPoints = [
                { meeting: lastMeeting, epsilon: lastEps },
                { meeting: projectedEnd, epsilon: Math.min(1.0, lastEps + slope * (projectedEnd - lastMeeting)) },
            ];
            const projLine = d3.line()
                .x(d => x(d.meeting))
                .y(d => y(d.epsilon));

            g.append('path')
                .datum(projectedPoints)
                .attr('fill', 'none')
                .attr('stroke', '#6b7280')
                .attr('stroke-width', 1.5)
                .attr('stroke-dasharray', '5,4')
                .attr('d', projLine);
        }

        // --- Epsilon history line ---
        const line = d3.line()
            .x(d => x(d.meeting))
            .y(d => y(d.epsilon))
            .curve(d3.curveMonotoneX);

        // Area fill
        const area = d3.area()
            .x(d => x(d.meeting))
            .y0(height)
            .y1(d => y(d.epsilon))
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(history)
            .attr('fill', '#3b82f6')
            .attr('fill-opacity', 0.12)
            .attr('d', area);

        g.append('path')
            .datum(history)
            .attr('fill', 'none')
            .attr('stroke', '#3b82f6')
            .attr('stroke-width', 2.5)
            .attr('d', line);

        // Data points
        g.selectAll('.ep-dot')
            .data(history)
            .enter()
            .append('circle')
            .attr('class', 'ep-dot')
            .attr('cx', d => x(d.meeting))
            .attr('cy', d => y(d.epsilon))
            .attr('r', 3.5)
            .attr('fill', '#3b82f6')
            .attr('stroke', '#1d4ed8')
            .attr('stroke-width', 1);

        // --- Predicted meeting markers ---
        if (predictedAutoSpawn && predictedAutoSpawn <= projectedEnd) {
            g.append('line')
                .attr('x1', x(predictedAutoSpawn)).attr('x2', x(predictedAutoSpawn))
                .attr('y1', 0).attr('y2', height)
                .attr('stroke', '#f59e0b')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '3,3')
                .attr('opacity', 0.7);

            g.append('text')
                .attr('x', x(predictedAutoSpawn) + 3)
                .attr('y', 12)
                .attr('fill', '#f59e0b')
                .attr('font-size', '9px')
                .text('M' + predictedAutoSpawn);
        }

        if (predictedBranchAuto && predictedBranchAuto <= projectedEnd) {
            g.append('line')
                .attr('x1', x(predictedBranchAuto)).attr('x2', x(predictedBranchAuto))
                .attr('y1', 0).attr('y2', height)
                .attr('stroke', '#a78bfa')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '3,3')
                .attr('opacity', 0.7);

            g.append('text')
                .attr('x', x(predictedBranchAuto) + 3)
                .attr('y', 24)
                .attr('fill', '#a78bfa')
                .attr('font-size', '9px')
                .text('M' + predictedBranchAuto);
        }

        // --- Current epsilon overlay (large text, top-right) ---
        const epsilonColor = currentEpsilon >= branchAutoThreshold
            ? '#a78bfa'
            : currentEpsilon >= autoSpawnThreshold
            ? '#f59e0b'
            : '#3b82f6';

        g.append('text')
            .attr('x', width)
            .attr('y', 18)
            .attr('text-anchor', 'end')
            .attr('fill', epsilonColor)
            .attr('font-size', '22px')
            .attr('font-weight', '700')
            .attr('font-variant-numeric', 'tabular-nums')
            .text('\u03b5 ' + currentEpsilon.toFixed(3));

        g.append('text')
            .attr('x', width)
            .attr('y', 32)
            .attr('text-anchor', 'end')
            .attr('fill', '#6b7280')
            .attr('font-size', '10px')
            .text('current trust region');
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
