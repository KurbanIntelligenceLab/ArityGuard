"""
RobustRAG's own QA data (Xiang et al., github.com/inspire-group/RobustRAG,
data/realtimeqa.json and data/open_nq.json, cloned into data/RobustRAG/),
turned into E4b-style pools so the delivered-feed and pool-resampling
certificates can be compared on the benchmark the RobustRAG certificate was
introduced on (paper: "RobustRAG's own datasets").

Per question:
  * authentic items: the first `n_passages` retrieved passages, in RobustRAG's
    retrieval order, as "title. text" (questions with fewer are skipped);
  * claim: our agent verifies claims, not open answers, so each question
    becomes 'The answer to the question "<q>" is <a>.' Half the chosen
    questions (seeded) use the first correct answer (gold TRUE), half use
    RobustRAG's own target incorrect answer (gold FALSE);
  * adversary: a placeholder items, never shown to the agent, with the
    worst-case vote of e4b_feed_vs_pool (any group holding one votes against
    gold). RobustRAG's poisoned passages are not used: the worst case is
    stronger than any particular poison, and it is what the certificates
    must survive.
biogen.json (long-form generation) has no verdict to certify and is not used.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from arq.pool import Pool, build_pool  # noqa: E402
from hotpotqa_adapter import ADVERSARY_PLACEHOLDER  # noqa: E402

DATASETS = ("realtimeqa", "open_nq")


def _claim(question: str, answer: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return f'The answer to the question "{q}" is {answer.strip()}.'


def build_robustrag_pools(data_dir: str = "data/RobustRAG/data",
                          datasets: Sequence[str] = DATASETS,
                          n_examples: int = 40, n_passages: int = 30,
                          adv_counts: Sequence[int] = (1, 2, 4),
                          seed: int = 0) -> Dict[str, Dict[int, List[Pool]]]:
    out = {}
    for ds in datasets:
        path = os.path.join(data_dir, f"{ds}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: git clone "
                                    "https://github.com/inspire-group/RobustRAG data/RobustRAG")
        with open(path) as fh:
            rows = json.load(fh)
        eligible = []
        for r in rows:
            passages = [f"{c.get('title', '').strip()}. {c.get('text', '').strip()}"
                        for c in r["context"] if (c.get("text") or "").strip()]
            correct = r["correct answer"]
            if isinstance(correct, str):
                correct = [correct]
            if len(passages) >= n_passages and correct and r.get("incorrect answer"):
                eligible.append((r, passages[:n_passages], correct[0]))
        if len(eligible) < n_examples:
            raise RuntimeError(f"{ds}: {len(eligible)} eligible questions, need {n_examples}")
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(eligible), n_examples, replace=False))
        truth = np.zeros(n_examples, bool)
        truth[rng.permutation(n_examples)[:n_examples // 2]] = True
        out[ds] = {a: [] for a in adv_counts}
        for k, (idx, is_true) in enumerate(zip(chosen, truth)):
            r, passages, correct = eligible[idx]
            claim = _claim(r["question"], correct if is_true else r["incorrect answer"])
            answer = "TRUE" if is_true else "FALSE"
            for a in adv_counts:
                rho = a / (len(passages) + a)
                pool = build_pool(authentic=passages,
                                  promoted=[f"{ADVERSARY_PLACEHOLDER}{i}" for i in range(a)],
                                  rhos=[rho], query=claim, answer=answer, seed=seed * 1000 + k)
                if len(pool.adversarial[rho]) != a:
                    raise RuntimeError(f"rho rounding gave {len(pool.adversarial[rho])}, wanted {a}")
                pool.meta.update({"dataset": ds, "example_id": r.get("id"),
                                  "n_adversarial": a, "ground_truth_arity": 1})
                out[ds][a].append(pool)
    return out
