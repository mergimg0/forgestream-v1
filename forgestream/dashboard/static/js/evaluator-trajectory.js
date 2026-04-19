/**
 * Evaluator Trajectory — E(π) line chart with area fill and axiom indicators
 * Fetches /api/evaluator
 */

class EvaluatorTrajectory {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 24, right: 24, bottom: 36, left: 48 };
    }

    async fetch() {
        try {
            const resp = await fetch('/api/evaluator');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (err) {
            console.warn('[EvaluatorTrajectory] fetch failed:', err);
            return { trajectory: [], current: {}, axioms: {} };
        }
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 420, height: 200 };
        const rect = el.getBoundingClientRect();
        return {
            width: (rect.width > 0 ? rect.width : 420) - this.margin.left - this.margin.right,
            height: 180,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();

        const { trajectory, axioms, current } = data;

        if (!trajectory || trajectory.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for evaluator data...');
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

        const x = d3.scaleLinear()
            .domain([0, trajectory.length - 1])
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, 1])
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
            .call(
                d3.axisBottom(x)
                    .ticks(6)
                    .tickFormat(d => `${d}`)
            );

        // X axis label
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height + 30)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '11px')
            .text('events');

        // Y axis
        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(1)));

        // Area fill under line
        const area = d3.area()
            .x((d, i) => x(i))
            .y0(height)
            .y1(d => y(d.evaluator || 0))
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(trajectory)
            .attr('class', 'area-fill')
            .attr('fill', '#3b82f6')
            .attr('d', area);

        // E(π) line
        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d.evaluator || 0))
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(trajectory)
            .attr('fill', 'none')
            .attr('stroke', '#3b82f6')
            .attr('stroke-width', 2.5)
            .attr('d', line);

        // Current value overlay
        const currentVal = current?.E_micro ?? (trajectory[trajectory.length - 1]?.evaluator ?? 0);
        g.append('text')
            .attr('x', width)
            .attr('y', 16)
            .attr('text-anchor', 'end')
            .attr('fill', '#3b82f6')
            .attr('font-size', '22px')
            .attr('font-weight', '700')
            .attr('font-variant-numeric', 'tabular-nums')
            .text(currentVal.toFixed(3));

        g.append('text')
            .attr('x', width)
            .attr('y', 30)
            .attr('text-anchor', 'end')
            .attr('fill', '#6b7280')
            .attr('font-size', '10px')
            .text('E(π)');

        // Axiom indicators (below chart)
        const axiomData = [
            { key: 'monotone', label: 'Monotone' },
            { key: 'bounded_step', label: 'Bounded' },
            { key: 'constraint', label: 'Constraint' },
        ];

        const axiomG = svg.append('g')
            .attr('transform', `translate(${this.margin.left}, ${height + this.margin.top + this.margin.bottom - 14})`);

        axiomData.forEach((a, i) => {
            const holds = axioms ? axioms[a.key] : null;
            const color = holds === true ? '#22c55e' : holds === false ? '#ef4444' : '#6b7280';
            const xOff = i * 90;
            axiomG.append('circle').attr('cx', xOff + 5).attr('cy', 0).attr('r', 5).attr('fill', color);
            axiomG.append('text')
                .attr('x', xOff + 14)
                .attr('y', 4)
                .attr('fill', '#9ca3af')
                .attr('font-size', '10px')
                .text(a.label);
        });
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
