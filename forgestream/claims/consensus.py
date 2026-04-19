"""Claim consensus — deduplicate claims across multiple extraction runs."""

from __future__ import annotations


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_consensus(
    runs: list[list[dict]],
    jaccard_threshold: float = 0.4,
    min_runs: int = 1,
) -> list[dict]:
    """Build consensus claims from multiple extraction runs.

    Clusters claims across runs by topic_keyword Jaccard similarity.
    Returns one representative claim per cluster, with boosted confidence.

    Args:
        runs: List of extraction runs, each a list of claim dicts.
        jaccard_threshold: Minimum Jaccard similarity to consider claims as matching.
        min_runs: Minimum number of runs a claim cluster must appear in.

    Returns:
        List of consensus claim dicts with added 'consensus_confidence' and 'run_count' fields.
    """
    # Flatten all claims with run index
    tagged: list[tuple[int, dict]] = []
    for run_idx, claims in enumerate(runs):
        for claim in claims:
            tagged.append((run_idx, claim))

    # Greedy clustering by keyword overlap
    clusters: list[list[tuple[int, dict]]] = []
    used: set[int] = set()

    for i, (run_i, claim_i) in enumerate(tagged):
        if i in used:
            continue
        cluster = [(run_i, claim_i)]
        used.add(i)
        kw_i = set(claim_i.get("topic_keywords", []))

        for j, (run_j, claim_j) in enumerate(tagged):
            if j in used:
                continue
            kw_j = set(claim_j.get("topic_keywords", []))
            if _jaccard(kw_i, kw_j) >= jaccard_threshold:
                cluster.append((run_j, claim_j))
                used.add(j)

        clusters.append(cluster)

    # Build consensus: one representative per cluster
    consensus = []
    for cluster in clusters:
        run_indices = {run_idx for run_idx, _ in cluster}
        if len(run_indices) < min_runs:
            continue

        # Representative = highest confidence claim in cluster
        best = max(cluster, key=lambda x: x[1].get("confidence", 0.0))
        representative = dict(best[1])

        # Merge keywords from all cluster members
        all_keywords: set[str] = set()
        for _, claim in cluster:
            all_keywords.update(claim.get("topic_keywords", []))
        representative["topic_keywords"] = sorted(all_keywords)

        # Boost confidence based on cross-run agreement
        base_conf = representative.get("confidence", 0.5)
        run_fraction = len(run_indices) / len(runs)
        representative["consensus_confidence"] = min(1.0, base_conf * (0.5 + 0.5 * run_fraction))
        representative["run_count"] = len(run_indices)

        consensus.append(representative)

    return consensus
