/**
 * ProofQueue — panel showing pending Lean 4 proof obligations.
 *
 * Renders each obligation as a card with:
 *   - claim text
 *   - Lean 4 stub (code block)
 *   - status badge
 *   - formalization confidence
 */
class ProofQueue {
    constructor(containerId) {
        this.containerId = containerId;
    }

    async update() {
        try {
            const resp = await fetch('/api/proof-obligations');
            if (!resp.ok) return;
            const data = await resp.json();
            this._render(data.obligations || []);
        } catch (e) {
            console.warn('ProofQueue update failed:', e);
        }
    }

    _statusColor(status) {
        switch (status) {
            case 'proved':   return '#2a9d8f';
            case 'failed':   return '#e76f51';
            case 'reviewing': return '#e9c46a';
            case 'dismissed': return '#555';
            default:         return '#457b9d';  // pending
        }
    }

    _createCard(ob) {
        const card = document.createElement('div');
        card.className = 'proof-obligation-card';
        card.style.cssText = [
            'margin:6px 0',
            'padding:8px 12px',
            'border-radius:6px',
            'background:#1e1e2e',
            'border-left:3px solid #457b9d',
            'font-size:12px',
        ].join(';');

        // Header: status badge + confidence
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:6px';

        const badge = document.createElement('span');
        badge.style.cssText = [
            'padding:2px 8px',
            'border-radius:10px',
            'font-size:10px',
            'font-weight:bold',
            'text-transform:uppercase',
            'color:#fff',
            `background:${this._statusColor(ob.status)}`,
        ].join(';');
        badge.textContent = ob.status || 'pending';

        const confSpan = document.createElement('span');
        confSpan.style.cssText = 'color:#888;font-size:10px';
        const fConf = ob.formalization_confidence != null
            ? Math.round(ob.formalization_confidence * 100) + '%'
            : '';
        confSpan.textContent = fConf ? 'form. conf: ' + fConf : '';

        header.appendChild(badge);
        header.appendChild(confSpan);

        // Claim text
        const claimEl = document.createElement('div');
        claimEl.style.cssText = 'color:#e2e8f0;margin-bottom:6px;line-height:1.4';
        claimEl.textContent = ob.claim_text || '';

        // Lean 4 stub code block
        const pre = document.createElement('pre');
        pre.style.cssText = [
            'background:#0d0d1a',
            'color:#a8d8a8',
            'padding:8px',
            'border-radius:4px',
            'overflow-x:auto',
            'font-size:11px',
            'margin:0',
            'white-space:pre-wrap',
            'word-break:break-word',
        ].join(';');
        const code = document.createElement('code');
        code.textContent = ob.lean4_stub || '';
        pre.appendChild(code);

        // Speaker attribution
        const meta = document.createElement('div');
        meta.style.cssText = 'color:#666;font-size:10px;margin-top:4px';
        meta.textContent = ob.speaker ? 'speaker: ' + ob.speaker : '';

        card.appendChild(header);
        card.appendChild(claimEl);
        card.appendChild(pre);
        if (ob.speaker) card.appendChild(meta);

        return card;
    }

    _render(obligations) {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        while (container.firstChild) container.removeChild(container.firstChild);

        if (obligations.length === 0) {
            const msg = document.createElement('p');
            msg.style.cssText = 'color:#888;font-size:12px';
            msg.textContent = 'No proof obligations detected';
            container.appendChild(msg);
            return;
        }

        obligations.forEach(ob => container.appendChild(this._createCard(ob)));
    }
}
