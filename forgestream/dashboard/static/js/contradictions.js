/**
 * Contradictions — panel showing detected CONTRADICTION events.
 *
 * Renders each contradiction as a card with:
 *   - concept_a vs concept_b
 *   - explanation text
 *   - probing questions list
 */
class Contradictions {
    constructor(containerId) {
        this.containerId = containerId;
    }

    async update() {
        try {
            const resp = await fetch('/api/contradictions');
            if (!resp.ok) return;
            const data = await resp.json();
            this._render(data.contradictions || []);
        } catch (e) {
            console.warn('Contradictions update failed:', e);
        }
    }

    _createCard(c) {
        const card = document.createElement('div');
        card.className = 'contradiction-card';
        card.style.cssText = [
            'margin:6px 0',
            'padding:8px 12px',
            'border-radius:6px',
            'background:#1e1e2e',
            'border-left:3px solid #e76f51',
            'font-size:12px',
        ].join(';');

        const title = document.createElement('div');
        title.style.cssText = 'font-weight:bold;margin-bottom:4px';

        const aSpan = document.createElement('span');
        aSpan.style.color = '#e9c46a';
        aSpan.textContent = c.concept_a || '?';

        const vsSpan = document.createElement('span');
        vsSpan.style.cssText = 'color:#888;margin:0 6px';
        vsSpan.textContent = 'vs';

        const bSpan = document.createElement('span');
        bSpan.style.color = '#e76f51';
        bSpan.textContent = c.concept_b || '?';

        title.appendChild(aSpan);
        title.appendChild(vsSpan);
        title.appendChild(bSpan);

        const explanation = document.createElement('div');
        explanation.style.cssText = 'color:#aaa;margin-bottom:4px';
        explanation.textContent = c.explanation || '';

        card.appendChild(title);
        card.appendChild(explanation);

        const questions = c.probing_questions || [];
        if (questions.length > 0) {
            const qTitle = document.createElement('div');
            qTitle.style.cssText = 'color:#888;font-size:10px;margin-top:4px';
            qTitle.textContent = 'Probing questions:';
            card.appendChild(qTitle);

            questions.slice(0, 3).forEach(q => {
                const qItem = document.createElement('div');
                qItem.style.cssText = 'color:#aaa;font-size:11px;padding-left:8px';
                qItem.textContent = '\u2022 ' + q;
                card.appendChild(qItem);
            });
        }

        return card;
    }

    _render(contradictions) {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        while (container.firstChild) container.removeChild(container.firstChild);

        if (contradictions.length === 0) {
            const msg = document.createElement('p');
            msg.style.cssText = 'color:#888;font-size:12px';
            msg.textContent = 'No contradictions detected';
            container.appendChild(msg);
            return;
        }

        contradictions.forEach(c => container.appendChild(this._createCard(c)));
    }
}
