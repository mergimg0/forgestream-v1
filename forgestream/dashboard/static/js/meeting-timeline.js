/**
 * Meeting Timeline — horizontal event timeline with shaped markers per event type
 * Fetches /api/graph (events embedded) — uses concept/requirement/artifact event types
 */

class MeetingTimeline {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 28, right: 20, bottom: 36, left: 60 };
        this._tooltip = null;
    }

    async fetch() {
        try {
            // Graph endpoint returns concepts with implicit event ordering; also fetch evaluator
            const [graphResp] = await Promise.all([
                fetch('/api/graph'),
            ]);
            if (!graphResp.ok) throw new Error(`HTTP ${graphResp.status}`);
            const graph = await graphResp.json();
            return graph;
        } catch (err) {
            console.warn('[MeetingTimeline] fetch failed:', err);
            return { concepts: [], requirements: [], artifacts: [], edges: [] };
        }
    }

    _buildEvents(data) {
        const events = [];
        let t = 0;

        for (const c of (data.concepts || [])) {
            events.push({ time: t, type: 'claim', label: c.name, confidence: c.confidence });
            t += 30 + Math.random() * 60;
        }
        for (const r of (data.requirements || [])) {
            events.push({ time: t, type: 'requirement', label: r.description || 'requirement' });
            t += 45 + Math.random() * 60;
        }
        for (const a of (data.artifacts || [])) {
            const label = a.compiles ? 'artifact (compiles)' : 'artifact';
            events.push({ time: t, type: 'artifact', label });
            t += 30;
        }

        // Sort by time
        events.sort((a, b) => a.time - b.time);
        return events;
    }

    // Shape generators per event type
    _symbolForType(type) {
        const shapes = {
            claim: d3.symbolTriangle,
            requirement: d3.symbolDiamond,
            finding: d3.symbolStar,
            branch: d3.symbolWye,
            artifact: d3.symbolCircle,
            seed: d3.symbolSquare,
        };
        return d3.symbol().type(shapes[type] || d3.symbolCircle).size(64)();
    }

    _colorForType(type) {
        const colors = {
            claim: '#0ff',
            requirement: '#f59e0b',
            finding: '#22c55e',
            branch: '#a78bfa',
            artifact: '#f97316',
            seed: '#3b82f6',
        };
        return colors[type] || '#9ca3af';
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 900, height: 90 };
        const rect = el.getBoundingClientRect();
        return {
            width: (rect.width > 0 ? rect.width : 900) - this.margin.left - this.margin.right,
            height: 80,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();

        const events = this._buildEvents(data);

        if (events.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for meeting events...');
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

        const timeExtent = d3.extent(events, d => d.time);
        const x = d3.scaleLinear()
            .domain([Math.max(0, timeExtent[0] - 5), timeExtent[1] + 30])
            .range([0, width]);

        // X axis (minutes:seconds)
        const xAxis = d3.axisBottom(x)
            .ticks(8)
            .tickFormat(d => {
                const m = Math.floor(d / 60);
                const s = Math.floor(d % 60);
                return `${m}:${s.toString().padStart(2, '0')}`;
            });

        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);

        // Timeline baseline
        g.append('line')
            .attr('x1', 0)
            .attr('y1', height / 2)
            .attr('x2', width)
            .attr('y2', height / 2)
            .attr('stroke', '#374151')
            .attr('stroke-width', 1.5);

        // Tooltip
        if (!this._tooltip) {
            this._tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);
        }
        const tooltip = this._tooltip;

        // Legend (top-left)
        const legendTypes = ['claim', 'requirement', 'artifact', 'finding'];
        const legendG = g.append('g').attr('transform', 'translate(0, -20)');
        legendTypes.forEach((type, i) => {
            legendG.append('path')
                .attr('d', this._symbolForType(type))
                .attr('transform', `translate(${i * 90}, 0)`)
                .attr('fill', this._colorForType(type));
            legendG.append('text')
                .attr('x', i * 90 + 10)
                .attr('y', 4)
                .attr('fill', '#9ca3af')
                .attr('font-size', '10px')
                .text(type);
        });

        // Event markers
        const midY = height / 2;

        g.selectAll('.event-marker')
            .data(events)
            .join('path')
            .attr('class', 'event-marker')
            .attr('d', d => this._symbolForType(d.type))
            .attr('transform', d => `translate(${x(d.time)},${midY})`)
            .attr('fill', d => this._colorForType(d.type))
            .on('mouseover', (event, d) => {
                tooltip.transition().duration(150).style('opacity', 0.97);
                const time = `${Math.floor(d.time / 60)}:${Math.floor(d.time % 60).toString().padStart(2, '0')}`;
                tooltip.html(
                    `<strong>${d.type}</strong>` +
                    `<div>${d.label}</div>` +
                    `<div>t = ${time}</div>`
                )
                    .style('left', `${event.pageX + 12}px`)
                    .style('top', `${event.pageY - 32}px`);
            })
            .on('mousemove', (event) => {
                tooltip
                    .style('left', `${event.pageX + 12}px`)
                    .style('top', `${event.pageY - 32}px`);
            })
            .on('mouseout', () => {
                tooltip.transition().duration(200).style('opacity', 0);
            });
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
