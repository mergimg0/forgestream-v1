/**
 * SeedGarden — card-based display of SEED events.
 *
 * Shows each seed cluster as a card with status badge:
 *   active   — recent seed (< 3 meetings old)
 *   dormant  — older seed
 *   promoted — keywords referenced by a REQUIREMENT event
 */
class SeedGarden {
    constructor(containerId) {
        this.containerId = containerId;
    }

    async update() {
        try {
            const resp = await fetch('/api/seeds');
            if (!resp.ok) return;
            const data = await resp.json();
            this._render(data.seeds || []);
        } catch (e) {
            console.warn('SeedGarden update failed:', e);
        }
    }

    _createCard(seed) {
        const statusColors = {
            active: '#2a9d8f',
            dormant: '#6c757d',
            promoted: '#e9c46a',
        };
        const statusIcons = {
            active: '\u25ce',
            dormant: '\u25cb',
            promoted: '\u2605',
        };

        const card = document.createElement('div');
        card.className = 'seed-card';
        card.style.cssText = [
            'display:inline-block',
            'margin:4px',
            'padding:6px 10px',
            'border-radius:6px',
            'background:#1e1e2e',
            'border:1px solid #333',
            'font-size:11px',
            'vertical-align:top',
            'max-width:180px',
        ].join(';');

        const status = seed.status || 'active';
        const color = statusColors[status] || '#888';
        const icon = statusIcons[status] || '\u25cb';
        const nodes = (seed.cluster_nodes || []).slice(0, 5).join(', ');
        const novelty = ((seed.novelty_score || 0) * 100).toFixed(0);

        const headerRow = document.createElement('div');
        headerRow.style.cssText = 'font-weight:bold;margin-bottom:3px';

        const iconSpan = document.createElement('span');
        iconSpan.style.color = color;
        iconSpan.textContent = icon + ' ';

        const domainSpan = document.createElement('span');
        domainSpan.style.color = '#ddd';
        domainSpan.textContent = seed.domain_guess || 'unknown';

        const statusSpan = document.createElement('span');
        statusSpan.style.cssText = 'float:right;font-size:9px;color:#888';
        statusSpan.textContent = status;

        headerRow.appendChild(iconSpan);
        headerRow.appendChild(domainSpan);
        headerRow.appendChild(statusSpan);

        const nodesDiv = document.createElement('div');
        nodesDiv.style.cssText = 'color:#aaa;word-break:break-word';
        nodesDiv.textContent = nodes;

        const noveltyDiv = document.createElement('div');
        noveltyDiv.style.cssText = 'color:#666;font-size:9px;margin-top:2px';
        noveltyDiv.textContent = 'novelty ' + novelty + '%';

        card.appendChild(headerRow);
        card.appendChild(nodesDiv);
        card.appendChild(noveltyDiv);
        return card;
    }

    _render(seeds) {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        while (container.firstChild) container.removeChild(container.firstChild);

        if (seeds.length === 0) {
            const msg = document.createElement('p');
            msg.style.cssText = 'color:#888;font-size:12px';
            msg.textContent = 'No seeds detected yet';
            container.appendChild(msg);
            return;
        }

        seeds.forEach(seed => container.appendChild(this._createCard(seed)));
    }
}
