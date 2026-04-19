/**
 * Rapport Trajectory — dual-axis line chart: rapport composite (gold) + E(π) (blue)
 * Fetches /api/emotion/rapport and /api/evaluator
 */

class RapportTrajectory {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 24, right: 24, bottom: 36, left: 50 };
        this._tooltip = null;
    }

    async fetch() {
        try {
            const [rapportResp, evalResp] = await Promise.all([
                fetch('/api/emotion/rapport'),
                fetch('/api/evaluator'),
            ]);
            if (!rapportResp.ok) throw new Error(`HTTP ${rapportResp.status}`);
            if (!evalResp.ok) throw new Error(`HTTP ${evalResp.status}`);
            const rapport = await rapportResp.json();
            const evaluator = await evalResp.json();
            return { rapport, evaluator };
        } catch (err) {
            console.warn('[RapportTrajectory] fetch failed:', err);
            return { rapport: { scores: [], latest_trend: 0 }, evaluator: { trajectory: [], axioms: {} } };
        }
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 520, height: 190 };
        const rect = el.getBoundingClientRect();
        return {
            width: (rect.width > 0 ? rect.width : 520) - this.margin.left - this.margin.right,
            height: 190,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();

        const { rapport, evaluator } = data;
        const scores = rapport?.scores || [];
        const trajectory = evaluator?.trajectory || [];

        const hasData = scores.length > 0 || trajectory.length > 0;
        if (!hasData) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for rapport data...');
            return;
        }

        const { width, height } = this._getSize();
        const totalH = height + this.margin.top + this.margin.bottom;

        const svg = this.container
            .append('svg')
            .attr('width', '100%')
            .attr('height', totalH)
            .attr('viewBox', `0 0 ${width + this.margin.left + this.margin.right} ${totalH}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');

        const g = svg.append('g').attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        // Grid lines
        g.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(y).tickSize(-width).tickFormat('').ticks(4));

        // Y axis
        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(1)));

        // Tooltip
        if (!this._tooltip) {
            this._tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);
        }
        const tooltip = this._tooltip;

        // ── Rapport composite line (gold) ──
        if (scores.length > 0) {
            const xR = d3.scaleLinear().domain([0, scores.length - 1]).range([0, width]);

            g.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(0,${height})`)
                .call(d3.axisBottom(xR).ticks(Math.min(scores.length, 6)).tickFormat(d => `${Math.round(d)}`));

            // X label
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height + 30)
                .attr('text-anchor', 'middle')
                .attr('fill', '#6b7280')
                .attr('font-size', '11px')
                .text('snapshots');

            const rapportLine = d3.line()
                .x((d, i) => xR(i))
                .y(d => y(d.composite_rapport ?? d.rapport_score ?? 0))
                .curve(d3.curveMonotoneX);

            g.append('path')
                .datum(scores)
                .attr('fill', 'none')
                .attr('stroke', '#f59e0b')
                .attr('stroke-width', 2.5)
                .attr('d', rapportLine);

            // Dots
            g.selectAll('.rapport-dot')
                .data(scores)
                .join('circle')
                .attr('cx', (d, i) => xR(i))
                .attr('cy', d => y(d.composite_rapport ?? d.rapport_score ?? 0))
                .attr('r', 3.5)
                .attr('fill', '#f59e0b')
                .on('mouseover', (event, d) => {
                    tooltip.transition().duration(150).style('opacity', 0.97);
                    const r = d.composite_rapport ?? d.rapport_score ?? 0;
                    tooltip.html(
                        `<strong>Rapport</strong>` +
                        `<div>composite: ${r.toFixed(3)}</div>` +
                        (d.attentiveness !== undefined ? `<div>attention: ${d.attentiveness?.toFixed(2)}</div>` : '') +
                        (d.positivity !== undefined ? `<div>positivity: ${d.positivity?.toFixed(2)}</div>` : '')
                    )
                        .style('left', `${event.pageX + 12}px`)
                        .style('top', `${event.pageY - 28}px`);
                })
                .on('mousemove', (event) => {
                    tooltip.style('left', `${event.pageX + 12}px`).style('top', `${event.pageY - 28}px`);
                })
                .on('mouseout', () => {
                    tooltip.transition().duration(200).style('opacity', 0);
                });
        } else if (trajectory.length > 0) {
            // No rapport data yet — just show x axis for E(π)
            const xE = d3.scaleLinear().domain([0, trajectory.length - 1]).range([0, width]);
            g.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(0,${height})`)
                .call(d3.axisBottom(xE).ticks(6).tickFormat(d => `${Math.round(d)}`));
        }

        // ── E(π) line (blue) ──
        if (trajectory.length > 0) {
            const xE = d3.scaleLinear().domain([0, trajectory.length - 1]).range([0, width]);
            const epiLine = d3.line()
                .x((d, i) => xE(i))
                .y(d => y(d.evaluator || 0))
                .curve(d3.curveMonotoneX);

            g.append('path')
                .datum(trajectory)
                .attr('fill', 'none')
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', scores.length > 0 ? '5,3' : 'none')
                .attr('d', epiLine);
        }

        // ── Legend ──
        const legendG = svg.append('g')
            .attr('transform', `translate(${this.margin.left + 8}, ${this.margin.top + 4})`);

        legendG.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0)
            .attr('stroke', '#f59e0b').attr('stroke-width', 2.5);
        legendG.append('text').attr('x', 22).attr('y', 4).attr('fill', '#9ca3af').attr('font-size', '10px').text('Rapport');

        legendG.append('line').attr('x1', 80).attr('x2', 98).attr('y1', 0).attr('y2', 0)
            .attr('stroke', '#3b82f6').attr('stroke-width', 2).attr('stroke-dasharray', '4,2');
        legendG.append('text').attr('x', 102).attr('y', 4).attr('fill', '#9ca3af').attr('font-size', '10px').text('E(π)');

        // ── Trend Indicator ──
        const trend = rapport?.latest_trend ?? 0;
        const trendContainer = this.container.append('div').attr('class', 'rapport-trend');
        const trendClass = trend > 0.01 ? 'trend-up' : trend < -0.01 ? 'trend-down' : 'trend-flat';
        const trendArrow = trend > 0.01 ? '↑' : trend < -0.01 ? '↓' : '→';
        trendContainer.append('span')
            .attr('class', trendClass)
            .text(`Trend: ${trendArrow} ${trend >= 0 ? '+' : ''}${trend.toFixed(3)}`);

        // ── Latest component breakdown ──
        const latest = scores.length > 0 ? scores[scores.length - 1] : null;
        if (latest) {
            const comps = ['attentiveness', 'positivity', 'coordination', 'symmetry'];
            const parts = comps.map(c => `${c}: ${(latest[c] ?? 0).toFixed(2)}`).filter(p => !p.endsWith('0.00'));
            if (parts.length > 0) {
                this.container.append('div')
                    .attr('class', 'component-breakdown')
                    .text(parts.join('  ·  '));
            }

            // Disengagement alert
            const disengaged = latest.disengaged_speakers || [];
            if (disengaged.length > 0) {
                this.container.append('div')
                    .attr('class', 'disengagement-alert')
                    .text(`Disengaged: ${disengaged.join(', ')}`);
            }
        }
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
