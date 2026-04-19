/**
 * SOSConvergence — epsilon gauge + axiom indicators + convergence estimate.
 *
 * Fetches /api/trust-region and renders:
 *   - An epsilon gauge (arc from 0 to trust_region_epsilon_base)
 *   - Three axiom status lights (monotone, bounded_step, constraint)
 *   - Consecutive improvement count and convergence estimate
 */
class SOSConvergence {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        if (!this.container) return;
        this.width = this.container.clientWidth || 300;
        this.height = 180;
        this._initSvg();
    }

    _initSvg() {
        this.svg = d3.select('#' + this.containerId)
            .append('svg')
            .attr('width', '100%')
            .attr('height', this.height);
    }

    async update() {
        try {
            const resp = await fetch('/api/trust-region');
            if (!resp.ok) return;
            const data = await resp.json();
            this._render(data);
        } catch (e) {
            console.warn('SOSConvergence update failed:', e);
        }
    }

    _render(data) {
        this.svg.selectAll('*').remove();
        const g = this.svg.append('g').attr('transform', 'translate(0,0)');

        const epsilon = data.epsilon || 0.3;
        const consecutive = data.consecutive_improvements || 0;
        const axioms = data.axiom_status || {};
        const epsilonBase = 0.3;

        // --- Epsilon gauge (semi-circle arc) ---
        const cx = 90, cy = 100, r = 55;
        const startAngle = -Math.PI;
        const endAngle = 0;
        const frac = Math.min(1, epsilon / epsilonBase);
        const fillAngle = startAngle + frac * (endAngle - startAngle);

        // Background arc
        const arcBg = d3.arc()
            .innerRadius(r - 12)
            .outerRadius(r)
            .startAngle(startAngle)
            .endAngle(endAngle);
        g.append('path')
            .attr('d', arcBg())
            .attr('transform', `translate(${cx},${cy})`)
            .attr('fill', '#333');

        // Fill arc
        const arcFill = d3.arc()
            .innerRadius(r - 12)
            .outerRadius(r)
            .startAngle(startAngle)
            .endAngle(fillAngle);
        const epsilonColor = epsilon < 0.1 ? '#2a9d8f' : epsilon < 0.2 ? '#e9c46a' : '#e76f51';
        g.append('path')
            .attr('d', arcFill())
            .attr('transform', `translate(${cx},${cy})`)
            .attr('fill', epsilonColor);

        // Epsilon label
        g.append('text')
            .attr('x', cx).attr('y', cy - 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '18px')
            .attr('fill', '#eee')
            .text('\u03b5 ' + epsilon.toFixed(3));

        g.append('text')
            .attr('x', cx).attr('y', cy + 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#888')
            .text('trust region');

        g.append('text')
            .attr('x', cx).attr('y', cy + 25)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('fill', '#aaa')
            .text(consecutive + ' consec. improvements');

        // --- Axiom indicator lights ---
        const axiomList = [
            { key: 'monotone', label: 'Monotone' },
            { key: 'bounded_step', label: 'Bounded \u0394' },
            { key: 'constraint', label: 'Bounded [0,1]' },
        ];

        const ax0 = 170;
        axiomList.forEach((ax, i) => {
            const y = 30 + i * 44;
            const ok = axioms[ax.key] !== false;
            const color = ok ? '#2a9d8f' : '#e76f51';

            g.append('circle')
                .attr('cx', ax0 + 8).attr('cy', y)
                .attr('r', 7)
                .attr('fill', color);

            g.append('text')
                .attr('x', ax0 + 22).attr('y', y + 4)
                .attr('font-size', '11px')
                .attr('fill', '#ccc')
                .text(ax.label);

            g.append('text')
                .attr('x', ax0 + 22).attr('y', y + 16)
                .attr('font-size', '9px')
                .attr('fill', ok ? '#2a9d8f' : '#e76f51')
                .text(ok ? 'ok' : 'violated');
        });
    }
}
