"""
data.twowiki_adapter -- turns the 2WikiMultihopQA validation split into
arq.pool.Pool objects for E2 (construct validity, the development notes task A3/D1).

Schema confirmed 2 Oct 2026 against `data/2wikimultihopqa/2wiki_validation
.jsonl` (saved via HF `datasets.load_dataset('voidful/2WikiMultihopQA',
split='validation')`, 12,576 examples), per-example shape:

    {"_id": str, "type": "compositional"|"comparison"|"bridge_comparison"
            |"inference",
     "question": str, "answer": str,
     "context": [[title, [sentence,...]], ...],
     "supporting_facts": [[title, sent_id], ...],
     "evidences": [[subject, relation, object], ...]}

Note `context`'s shape differs from HotpotQA's (a list of [title,
sentences] PAIRS here, vs. HotpotQA's parallel title/sentences arrays) --
handled directly, not coerced to match hotpotqa_adapter.py's shape.

RESTRICTED TO yes/no QUESTIONS, same judgment call and same reasoning as
hotpotqa_adapter.py: 1,295 of 12,576 examples have answer in {"yes","no"}
(confirmed 2 Oct 2026), all of type "comparison" (866) or "bridge_
comparison" (429) -- "compositional" and "inference" types never have a
yes/no answer in this split and are excluded entirely, not force-converted.
Claim text is the question's own wording (see hotpotqa_adapter.py's
_claim_from_yesno_question, reused here) with gold TRUE/FALSE taken
directly from answer=="yes"/"no" -- no fabricated or corrupted claims.

TWO GROUND-TRUTH ARITIES IN THIS ONE DATASET -- this is the point of
including 2Wiki in E2 (the development notes: "chains of known length (arity >2
cases)"), and is handled by building TWO SEPARATE POOL SETS, not one:
  - type=="comparison": supporting_facts always spans exactly 2 distinct
    titles (confirmed 2 Oct 2026: Counter over all 866 comparison yes/no
    examples == {2: 866}) -- TWOWIKI_COMPARISON_ARITY = 2.
  - type=="bridge_comparison": supporting_facts always spans exactly 4
    distinct titles (confirmed 2 Oct 2026: Counter over all 429
    bridge_comparison yes/no examples == {4: 429}) -- TWOWIKI_BRIDGE_ARITY
    = 4. This is the dataset's actual "arity > 2" case the development notes asks
    for; e2_construct_validity's per-dataset design (the development notes task D1)
    means this adapter's two subsets are passed to e2_construct_validity
    as two separate dataset entries ("2wiki_comparison",
    "2wiki_bridge_comparison"), each validated against its own
    ground-truth arity, rather than averaged into one misleading number.

No adversarial content exists in this dataset either -- see
hotpotqa_adapter.py's module docstring for why pools are built with an
empty `promoted` list and a fixed internal _NO_ADVERSARY_RHO rather than
the caller's `rhos` (E2 only needs clean groups; task_arity depends only
on q(t), not rho). The same fix applies here: a real ValueError was hit
(2 Oct 2026) when CoPHEME-scale rhos were first tried against this
dataset's larger pools (hundreds of context sentences), since build_pool's
rounding does not reliably give 0 adversarial items at those pool sizes.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from arq.pool import Pool, build_pool  # noqa: E402
from hotpotqa_adapter import _claim_from_yesno_question  # noqa: E402

TWOWIKI_COMPARISON_ARITY = 2
TWOWIKI_BRIDGE_ARITY = 4

MIN_ITEMS = 8

# See hotpotqa_adapter.py's _NO_ADVERSARY_RHO for why this is fixed and
# small rather than reusing the caller's rhos.
_NO_ADVERSARY_RHO = 1e-6


def _load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch it first: datasets.load_dataset("
            "'voidful/2WikiMultihopQA', split='validation').to_json(path) "
            "-- see the development notes task A3.")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"{path} is empty")
    return out


def _build_subset(records: List[dict], qtype: str, ground_truth_arity: int,
                  rhos: Sequence[float], seed: int) -> List[Pool]:
    subset = [r for r in records
             if r.get("type") == qtype and r.get("answer") in ("yes", "no")]
    if not subset:
        raise RuntimeError(
            f"Zero yes/no examples of type {qtype!r} found -- expected "
            f"{'866' if qtype == 'comparison' else '429'} as of the 2 Oct "
            "2026 HF download. Schema may have changed.")

    pools: List[Pool] = []
    skipped = 0
    for i, r in enumerate(subset):
        sf_titles = {t for t, _ in r["supporting_facts"]}
        if len(sf_titles) != ground_truth_arity:
            raise ValueError(
                f"example {r['_id']} (type={qtype}): supporting_facts "
                f"spans {len(sf_titles)} titles, expected exactly "
                f"{ground_truth_arity} -- contradicts the dataset-wide "
                "invariant this adapter was built against (see module "
                "docstring); re-verify the download before trusting "
                "ground-truth arity on this subset.")

        items = [s for _, sentences in r["context"] for s in sentences
                 if s.strip()]
        if len(items) < MIN_ITEMS:
            skipped += 1
            continue

        claim = _claim_from_yesno_question(r["question"])
        answer = "TRUE" if r["answer"] == "yes" else "FALSE"

        pool = build_pool(authentic=items, promoted=[],
                          rhos=[_NO_ADVERSARY_RHO],
                          query=claim, answer=answer, seed=seed + i)
        pool.meta["dataset"] = f"2wiki_{qtype}"
        pool.meta["ground_truth_arity"] = ground_truth_arity
        pool.meta["example_id"] = r["_id"]
        pools.append(pool)

    if skipped:
        print(f"NOTE: skipped {skipped} 2Wiki {qtype} example(s) with "
              f"fewer than {MIN_ITEMS} context sentences.", file=sys.stderr)
    if not pools:
        raise RuntimeError(f"Zero usable {qtype} pools built")
    return pools


def build_twowiki_controlled_pools(data_dir: str, distractor_levels=(0, 4, 16),
                                   n_examples: int = 40, seed: int = 0,
                                   transform=None, suffix: str = "") -> dict:
    """Controlled-density pools for both 2Wiki subtypes; see
    hotpotqa_adapter.build_controlled_pools for the design."""
    from hotpotqa_adapter import build_controlled_pools
    records = _load_jsonl(os.path.join(data_dir, "2wiki_validation.jsonl"))
    out = {}
    for qtype, arity in (("comparison", TWOWIKI_COMPARISON_ARITY),
                         ("bridge_comparison", TWOWIKI_BRIDGE_ARITY)):
        examples = [{"id": r["_id"], "question": r["question"],
                     "answer": r["answer"],
                     "context": [(t, s) for t, s in r["context"]],
                     "supporting": [(t, s) for t, s in r["supporting_facts"]]}
                    for r in records
                    if r.get("type") == qtype and r.get("answer") in ("yes", "no")]
        out.update(build_controlled_pools(examples, f"2wiki_{qtype}{suffix}", arity,
                                          distractor_levels, n_examples, seed,
                                          transform))
    return out


def build_twowiki_pools(data_dir: str, rhos: Sequence[float], seed: int = 0
                        ) -> Dict[str, List[Pool]]:
    """Build two separate pool sets: 'comparison' (arity 2) and
    'bridge_comparison' (arity 4), keyed exactly as e2_construct_validity's
    pools_by_dataset expects (see that function's docstring).

    Returns {"2wiki_comparison": [...], "2wiki_bridge_comparison": [...]}
    so callers can pass this dict's values straight into
    e2_construct_validity alongside HotpotQA's single pool set, each
    checked against its own ground_truth_arity.
    """
    path = os.path.join(data_dir, "2wiki_validation.jsonl")
    records = _load_jsonl(path)
    return {
        "2wiki_comparison": _build_subset(
            records, "comparison", TWOWIKI_COMPARISON_ARITY, rhos, seed),
        "2wiki_bridge_comparison": _build_subset(
            records, "bridge_comparison", TWOWIKI_BRIDGE_ARITY, rhos, seed),
    }
