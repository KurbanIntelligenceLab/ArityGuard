"""
arq.baselines -- the comparison methods of Section 6 (E4).

Each is reported at its own call budget AND at a budget matched to ARQ, with
baselines free to spend the extra however helps them most. If ARQ's advantage
disappears under matching, that is the finding.

None of these emits a certificate except feed_partition, whose certificate is
computed over the delivered feed and is the one the paper argues cannot fire
once a task needs more than one item at a time.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .agent import Agent
from .certificate import ABSTAIN


def undefended(agent: Agent, items: Sequence[str], claim: str) -> Dict[str, object]:
    """Read the whole feed in one call. One call, no certificate."""
    return {"verdict": agent.verdict(list(items), claim), "calls": 1,
            "certified": None, "method": "undefended"}


def feed_partition(agent: Agent, items: Sequence[str], claim: str, omega: int,
                   k_prime: int) -> Dict[str, object]:
    """RobustRAG-style isolate-then-aggregate over the DELIVERED feed.

    Partition k items into disjoint groups of omega, take a verdict per group,
    and certify when the winner leads by more than the corruption budget
    k_prime. This is the mechanism the paper compares against; the certificate
    is deterministic because the contaminated group count is bounded by k'.
    """
    if omega < 1:
        raise ValueError("omega must be at least 1")
    groups = [list(items[i:i + omega]) for i in range(0, len(items), omega)]
    verdicts = [agent.verdict(g, claim) for g in groups]
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    c1 = ranked[0][1]
    c2 = ranked[1][1] if len(ranked) > 1 else 0
    return {"verdict": ranked[0][0] if c1 - c2 > k_prime else None,
            "certified": c1 - c2 > k_prime, "calls": len(groups),
            "n_groups": len(groups), "c1": c1, "c2": c2, "threshold": k_prime,
            "method": f"feed_partition(omega={omega})"}


def majority_vote(agent: Agent, items: Sequence[str], claim: str,
                  K: int = 5) -> Dict[str, object]:
    """K independent readings of the whole feed, plurality wins. The downstream
    layer used by the attack paper. No certificate."""
    verdicts = [agent.verdict(list(items), claim) for _ in range(K)]
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return {"verdict": max(counts, key=counts.get), "calls": K,
            "certified": None, "method": f"majority_vote(K={K})"}


def llm_judge(agent: Agent, judge: Agent, items: Sequence[str], claim: str,
              K: int = 5) -> Dict[str, object]:
    """K analyst readings, then a separate judge sees the reports and the raw
    evidence. The attack paper's stronger downstream layer. No certificate."""
    reports = [agent.verdict(list(items), claim) for _ in range(K)]
    summary = list(items) + [f"analyst {i} says {r}" for i, r in enumerate(reports)]
    return {"verdict": judge.verdict(summary, claim), "calls": K + 1,
            "certified": None, "method": f"llm_judge(K={K})"}


def influence_detector(agent: Agent, items: Sequence[str], claim: str,
                       threshold: float = 0.5) -> Dict[str, object]:
    """Leave-one-out influence audit in the style of COMA: score each item by
    whether removing it flips the verdict, then abstain if any single item is
    pivotal. A DETECTOR, not a certificate: it returns a score and a decision,
    with no guarantee over unseen adversary behaviour."""
    full = agent.verdict(list(items), claim)
    flips = 0
    for i in range(len(items)):
        held = [s for j, s in enumerate(items) if j != i]
        if held and agent.verdict(held, claim) != full:
            flips += 1
    pivotal = flips / max(1, len(items))
    return {"verdict": None if pivotal > threshold else full,
            "certified": None, "calls": 1 + len(items),
            "pivotal_fraction": pivotal, "method": "influence_detector"}


def query_decomposition(agent: Agent, items: Sequence[str], claim: str,
                        subclaims: Sequence[str], k_prime: int = 1) -> Dict[str, object]:
    """The strategy RobustRAG proposes for multi-hop: split into single-hop
    sub-questions, certify each at arity one, then compose.

    `subclaims` must be supplied. Where no ground-truth decomposition exists,
    which is the synthesis case the paper is about, pass an empty list; the
    method then reports that it does not apply rather than inventing a
    decomposition.
    """
    if not subclaims:
        return {"verdict": None, "certified": False, "calls": 0,
                "method": "query_decomposition(no decomposition available)",
                "applies": False}
    hop_results = []
    calls = 0
    for sc in subclaims:
        r = feed_partition(agent, items, sc, omega=1, k_prime=k_prime)
        calls += r["calls"]
        hop_results.append(r)
    composed = list(items) + [f"established: {sc} = {r['verdict']}"
                              for sc, r in zip(subclaims, hop_results)]
    final = agent.verdict(composed, claim)
    calls += 1
    all_hops_certified = all(r["certified"] for r in hop_results)
    return {"verdict": final, "certified": all_hops_certified, "calls": calls,
            "hops": len(subclaims), "applies": True,
            "note": "the composition step itself is not certified",
            "method": "query_decomposition"}
