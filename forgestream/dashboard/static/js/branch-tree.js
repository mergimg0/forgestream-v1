/**
 * BranchTree — D3 tree layout visualizing BRANCH_POINT events.
 *
 * Renders a hierarchical tree where each node is a conversation branch,
 * with edges showing parent → child drift events.
 */
class BranchTree {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        if (!this.container) return;
        this.width = this.container.clientWidth || 400;
        this.height = 200;
        this._initSvg();
    }

    _initSvg() {
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', '100%')
            .attr('height', this.height);
        this.g = this.svg.append('g').attr('transform', 'translate(40, 20)');
    }

    async update() {
        try {
            const resp = await fetch('/api/branches');
            if (!resp.ok) return;
            const data = await resp.json();
            this._render(data.branches || []);
        } catch (e) {
            console.warn('BranchTree update failed:', e);
        }
    }

    _render(branches) {
        this.g.selectAll('*').remove();

        if (branches.length === 0) {
            this.g.append('text')
                .attr('x', 10).attr('y', 30)
                .attr('fill', '#888').attr('font-size', '12px')
                .text('No branches yet');
            return;
        }

        // Build hierarchy: find root nodes (no parent or parent not in list)
        const ids = new Set(branches.map(b => b.new_branch_id || b.id));
        const roots = branches.filter(b => !b.parent_branch_id || !ids.has(b.parent_branch_id));

        // Build D3 hierarchy from branch list
        const nodeMap = {};
        branches.forEach(b => {
            const id = b.new_branch_id || b.id;
            nodeMap[id] = { id, data: b, children: [] };
        });
        branches.forEach(b => {
            const id = b.new_branch_id || b.id;
            const pid = b.parent_branch_id;
            if (pid && nodeMap[pid]) {
                nodeMap[pid].children.push(nodeMap[id]);
            }
        });

        const rootNodes = Object.values(nodeMap).filter(n =>
            !n.data.parent_branch_id || !nodeMap[n.data.parent_branch_id]
        );

        const syntheticRoot = { id: '__root__', data: {}, children: rootNodes };
        const hierarchy = d3.hierarchy(syntheticRoot, d => d.children);
        const treeLayout = d3.tree().size([this.width - 80, this.height - 60]);
        const treeData = treeLayout(hierarchy);

        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, 1]);

        // Draw links
        this.g.selectAll('.branch-link')
            .data(treeData.links().filter(l => l.source.data.id !== '__root__'))
            .enter().append('line')
            .attr('class', 'branch-link')
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
            .attr('stroke', '#555').attr('stroke-width', 1.5);

        // Draw nodes (skip synthetic root)
        const nodes = treeData.descendants().filter(d => d.data.id !== '__root__');
        const nodeG = this.g.selectAll('.branch-node')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'branch-node')
            .attr('transform', d => `translate(${d.x},${d.y})`);

        nodeG.append('circle')
            .attr('r', 6)
            .attr('fill', d => colorScale(d.data.data.potential_score || 0.3))
            .attr('stroke', '#aaa').attr('stroke-width', 1);

        nodeG.append('title')
            .text(d => d.data.data.description || d.data.id);

        nodeG.append('text')
            .attr('x', 8).attr('y', 4)
            .attr('font-size', '9px').attr('fill', '#ccc')
            .text(d => {
                const desc = d.data.data.description || '';
                return desc.length > 20 ? desc.slice(0, 20) + '…' : desc;
            });
    }
}
