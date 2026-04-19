/**
 * Emotion Timeline — per-speaker arousal/valence/dominance line chart
 * with claim event markers on x-axis.
 * Fetches /api/emotion/timeline
 */

class EmotionTimeline {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 20, right: 100, bottom: 40, left: 50 };
        this.colorScale = d3.scaleOrdinal(d3.schemeTableau10);
        this.metric = 'arousal'; // arousal | valence | dominance
        this._tooltip = null;
    }

    async fetch() {
        try {
            const resp = await fetch('/api/emotion/timeline');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const body = await resp.json();
            return body.timeline || [];
        } catch (err) {
            console.warn('[EmotionTimeline] fetch failed:', err);
            return [];
        }
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 520, height: 200 };
        const rect = el.getBoundingClientRect();
        return {
            width: (rect.width > 0 ? rect.width : 520) - this.margin.left - this.margin.right,
            height: 190,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();

        // Toggle buttons
        const btnBar = this.container.append('div').attr('class', 'metric-toggle');
        for (const m of ['arousal', 'valence', 'dominance']) {
            btnBar.append('button')
                .attr('class', `metric-btn${this.metric === m ? ' active' : ''}`)
                .text(m)
                .on('click', () => {
                    this.metric = m;
                    this.render(data);
                });
        }

        if (!data || data.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for emotion data...');
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

        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d.timestamp_ms))
            .range([0, width]);

        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        // Grid lines
        g.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(y).tickSize(-width).tickFormat('').ticks(4));

        // Axes
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(
                d3.axisBottom(x)
                    .ticks(6)
                    .tickFormat(d => `${(d / 1000).toFixed(0)}s`)
            );

        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(1)));

        // Y axis label
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -38)
            .attr('x', -height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '10px')
            .text(this.metric);

        // Tooltip
        if (!this._tooltip) {
            this._tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);
        }
        const tooltip = this._tooltip;

        // Group by speaker
        const speakers = d3.group(data, d => d.speaker_id);
        const speakerList = Array.from(speakers.keys()).sort();

        for (const [speaker, points] of speakers) {
            const sorted = points.slice().sort((a, b) => a.timestamp_ms - b.timestamp_ms);
            const color = this.colorScale(speaker);
            const metric = this.metric;

            const line = d3.line()
                .x(d => x(d.timestamp_ms))
                .y(d => y(d[metric] !== undefined ? d[metric] : 0.5))
                .curve(d3.curveMonotoneX);

            g.append('path')
                .datum(sorted)
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 2)
                .attr('d', line);

            // Dots at each data point
            g.selectAll(`.dot-${speaker}`)
                .data(sorted)
                .join('circle')
                .attr('cx', d => x(d.timestamp_ms))
                .attr('cy', d => y(d[metric] !== undefined ? d[metric] : 0.5))
                .attr('r', 3)
                .attr('fill', color)
                .on('mouseover', (event, d) => {
                    tooltip.transition().duration(150).style('opacity', 0.97);
                    tooltip.html(
                        `<strong>${d.speaker_id}</strong>` +
                        `<div>${metric}: ${d[metric]?.toFixed(3) ?? 'n/a'}</div>` +
                        `<div>t = ${(d.timestamp_ms / 1000).toFixed(1)}s</div>` +
                        (d.emotion_tag ? `<div>tag: ${d.emotion_tag}</div>` : '')
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
        }

        // Claim markers at bottom of chart (triangles on x-axis)
        const claimData = data.filter(d => d.emotion_tag === 'claim' || d.is_claim);
        if (claimData.length > 0) {
            const claimSymbol = d3.symbol().type(d3.symbolTriangle).size(36)();
            g.selectAll('.claim-marker')
                .data(claimData)
                .join('path')
                .attr('class', 'claim-marker')
                .attr('d', claimSymbol)
                .attr('transform', d => `translate(${x(d.timestamp_ms)},${height + 10})`);
        }

        // Legend (right side)
        const legendG = svg.append('g')
            .attr('transform', `translate(${width + this.margin.left + 8}, ${this.margin.top})`);

        speakerList.forEach((spk, i) => {
            legendG.append('line')
                .attr('x1', 0).attr('x2', 16).attr('y1', i * 18).attr('y2', i * 18)
                .attr('stroke', this.colorScale(spk))
                .attr('stroke-width', 2);
            legendG.append('text')
                .attr('x', 20).attr('y', i * 18 + 4)
                .attr('fill', '#9ca3af')
                .attr('font-size', '10px')
                .text(spk.length > 10 ? spk.slice(0, 10) + '…' : spk);
        });
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
