/**
 * Knowledge Graph — D3 force-directed graph
 * Fetches /api/graph and renders concepts as nodes, co-occurrences as edges.
 */

class KnowledgeGraph {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 10, right: 10, bottom: 10, left: 10 };
        this.colorScale = d3.scaleOrdinal(d3.schemeTableau10);
        this.simulation = null;
        this._tooltip = null;
        this._initialized = false;
    }

    async fetch() {
        try {
            const resp = await fetch('/api/graph');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (err) {
            console.warn('[KnowledgeGraph] fetch failed:', err);
            return { concepts: [], edges: [], requirements: [], artifacts: [] };
        }
    }

    _getSize() {
        const el = document.getElementById(this.containerId);
        if (!el) return { width: 900, height: 400 };
        const rect = el.getBoundingClientRect();
        return {
            width: rect.width > 0 ? rect.width : 900,
            height: rect.height > 0 ? rect.height : 400,
        };
    }

    render(data) {
        this.container.selectAll('*').remove();
        this._initialized = false;

        const { concepts, edges } = data;

        if (!concepts || concepts.length === 0) {
            this.container
                .append('div')
                .attr('class', 'empty-state')
                .text('Waiting for knowledge graph data...');
            return;
        }

        const { width, height } = this._getSize();

        // Build D3-compatible nodes and links
        const nodes = concepts.map(c => ({
            id: c.name,
            confidence: c.confidence || 0.5,
            count: c.count || 1,
            domain: c.domain || 'general',
        }));

        // Deduplicate edges and convert to node references
        const nodeIndex = new Map(nodes.map(n => [n.id, n]));
        const links = [];
        const seen = new Set();
        for (const e of (edges || [])) {
            const key = [e.source, e.target].sort().join('|');
            if (!seen.has(key) && nodeIndex.has(e.source) && nodeIndex.has(e.target)) {
                seen.add(key);
                links.push({ source: e.source, target: e.target, weight: e.weight || 0.5 });
            }
        }

        // Build SVG
        const svg = this.container
            .append('svg')
            .attr('width', '100%')
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('overflow', 'hidden');

        // Zoom layer
        const zoomLayer = svg.append('g').attr('class', 'zoom-layer');

        svg.call(
            d3.zoom()
                .scaleExtent([0.3, 4])
                .on('zoom', (event) => {
                    zoomLayer.attr('transform', event.transform);
                })
        );

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-5 -5 10 10')
            .attr('refX', 10)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M-5,-4L5,0L-5,4')
            .attr('fill', '#374151');

        // Links
        const link = zoomLayer.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => 1 + 3 * (d.weight || 0.3));

        // Tooltip
        if (!this._tooltip) {
            this._tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);
        }
        const tooltip = this._tooltip;

        // Nodes (group)
        const node = zoomLayer.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('class', 'node')
            .call(
                d3.drag()
                    .on('start', (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on('drag', (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on('end', (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    })
            )
            .on('mouseover', (event, d) => {
                tooltip.transition().duration(150).style('opacity', 0.97);
                tooltip.html(
                    `<strong>${d.id}</strong>` +
                    `<div>Confidence: ${(d.confidence * 100).toFixed(0)}%</div>` +
                    `<div>Mentions: ${d.count}</div>`
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

        node.append('circle')
            .attr('r', d => 5 + 15 * (d.confidence || 0.5))
            .attr('fill', d => this.colorScale(d.domain || d.id.charAt(0)));

        node.append('text')
            .attr('dy', d => -(7 + 15 * (d.confidence || 0.5)))
            .attr('text-anchor', 'middle')
            .text(d => d.id.length > 18 ? d.id.slice(0, 16) + '…' : d.id);

        // Force simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-120))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => 10 + 15 * (d.confidence || 0.5)))
            .on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });

        this._initialized = true;
    }

    async update() {
        const data = await this.fetch();
        this.render(data);
    }
}
