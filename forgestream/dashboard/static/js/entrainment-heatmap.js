/**
 * Entrainment Heatmap — speaker-pair correlation matrix
 * Fetches /api/emotion/entrainment, renders latest snapshot as heatmap
 */

class EntrainmentHeatmap {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
        this._tooltip = null;
    }

    async fetch() {
        try {
            const resp = await fetch('/api/emotion/entrainment');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const body = await resp.json();
            const snapshots = body.snapshots || [];
            return snapshots.length ? snapshots[snapshots.length - 1] : null;
        } catch (err) {
            console.warn('[EntrainmentHeatmap] fetch failed:', err);
            return null;
        }
    }

    render(snapshot) {
        this.container.selectAll('*').remove();

        if (!snapshot || !snapshot.speaker_pairs || snapshot.speaker_pairs.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for entrainment data...');
            return;
        }

        // Build speaker set
        const speakers = new Set();
        snapshot.speaker_pairs.forEach(p => {
            speakers.add(String(p.speaker_a));
            speakers.add(String(p.speaker_b));
        });
        const speakerList = Array.from(speakers).sort();
        const n = speakerList.length;

        const cellSize = Math.min(64, Math.floor(180 / n));
        const labelPad = 48;
        const svgW = labelPad + n * cellSize + 20;
        const svgH = labelPad + n * cellSize + 20;

        // Tooltip
        if (!this._tooltip) {
            this._tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);
        }
        const tooltip = this._tooltip;

        const svg = this.container.append('svg')
            .attr('width', svgW)
            .attr('height', svgH);

        const g = svg.append('g').attr('transform', `translate(${labelPad},${labelPad})`);

        // Column headers (top)
        speakerList.forEach((s, i) => {
            g.append('text')
                .attr('x', i * cellSize + cellSize / 2)
                .attr('y', -8)
                .attr('text-anchor', 'middle')
                .attr('fill', '#9ca3af')
                .attr('font-size', '11px')
                .text(s.length > 6 ? s.slice(0, 6) + '…' : s);
        });

        // Row headers (left)
        speakerList.forEach((s, i) => {
            g.append('text')
                .attr('x', -8)
                .attr('y', i * cellSize + cellSize / 2 + 4)
                .attr('text-anchor', 'end')
                .attr('fill', '#9ca3af')
                .attr('font-size', '11px')
                .text(s.length > 6 ? s.slice(0, 6) + '…' : s);
        });

        // Build lookup: pair key → correlation value
        const lookup = {};
        snapshot.speaker_pairs.forEach(p => {
            const val = p.f0_correlation ?? p.correlation ?? 0;
            lookup[`${p.speaker_a}-${p.speaker_b}`] = val;
            lookup[`${p.speaker_b}-${p.speaker_a}`] = val;
        });

        // Cells
        for (let row = 0; row < n; row++) {
            for (let col = 0; col < n; col++) {
                const sa = speakerList[row];
                const sb = speakerList[col];
                const val = row === col ? 1.0 : (lookup[`${sa}-${sb}`] ?? 0);
                const absVal = Math.abs(val);

                g.append('rect')
                    .attr('x', col * cellSize)
                    .attr('y', row * cellSize)
                    .attr('width', cellSize - 2)
                    .attr('height', cellSize - 2)
                    .attr('fill', this.colorScale(absVal))
                    .attr('rx', 4)
                    .on('mouseover', (event) => {
                        tooltip.transition().duration(150).style('opacity', 0.97);
                        tooltip.html(
                            `<strong>${sa} ↔ ${sb}</strong>` +
                            `<div>F0 correlation: ${val.toFixed(3)}</div>`
                        )
                            .style('left', `${event.pageX + 12}px`)
                            .style('top', `${event.pageY - 28}px`);
                    })
                    .on('mousemove', (event) => {
                        tooltip
                            .style('left', `${event.pageX + 12}px`)
                            .style('top', `${event.pageY - 28}px`);
                    })
                    .on('mouseout', () => {
                        tooltip.transition().duration(200).style('opacity', 0);
                    });

                g.append('text')
                    .attr('x', col * cellSize + cellSize / 2 - 1)
                    .attr('y', row * cellSize + cellSize / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', cellSize > 48 ? '12px' : '9px')
                    .attr('fill', absVal > 0.55 ? '#fff' : '#333')
                    .text(val.toFixed(2));
            }
        }

        // Snapshot timestamp label
        if (snapshot.timestamp_ms) {
            const ts = new Date(snapshot.timestamp_ms).toLocaleTimeString();
            this.container.append('div')
                .style('font-size', '10px')
                .style('color', '#6b7280')
                .style('margin-top', '6px')
                .text(`Latest snapshot: ${ts}`);
        }
    }

    async update() {
        const snapshot = await this.fetch();
        this.render(snapshot);
    }
}
