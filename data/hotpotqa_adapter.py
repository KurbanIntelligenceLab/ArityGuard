"""
data.hotpotqa_adapter -- turns the HotpotQA (distractor) validation split into
arq.pool.Pool objects for E2 (construct validity, the development notes task A2/D1).

Schema confirmed 2 Oct 2026 against `data/hotpotqa/hotpotqa_distractor_
validation.jsonl` (saved via HF `datasets.load_dataset('hotpot_qa',
'distractor', split='validation')`, 7,405 examples), per-example shape:

    {"id": str, "question": str, "answer": str, "type": "bridge"|"comparison",
     "level": str,
     "supporting_facts": {"title": [str,...], "sent_id": [int,...]},
     "context": {"title": [str,...], "sentences": [[str,...],...]}}

RESTRICTED TO yes/no COMPARISON QUESTIONS (confirmed 2 Oct 2026: 458 of 7,405
examples have answer in {"yes","no"}). This is a deliberate, judgment-call
scope restriction, not an oversight: arq.agent.Agent.verdict needs a
TRUE/FALSE/ABSTAIN claim-verification task, not an open-ended "what is the
answer" extraction task, and HotpotQA's non-yes/no majority (bridge
questions asking "who/what/where...") has no natural TRUE/FALSE claim
without fabricating a corrupted distractor claim -- which this adapter
deliberately avoids rather than inventing. A yes/no comparison question
converts to a claim with NO rewording risk: "Were X and Y of the same Z?" /
answer "yes" literally IS the claim "X and Y were of the same Z", gold
TRUE; answer "no" makes the identical claim text gold FALSE. This keeps
every claim and its gold label drawn directly from the dataset's own
annotation, with no invented content.

GROUND-TRUTH ARITY: every yes/no example's supporting_facts spans EXACTLY 2
distinct titles (confirmed 2 Oct 2026 by direct inspection: Counter(len(set(
supporting_facts.title)) for all 7,405 examples) == {2: 7405}, i.e. this
holds dataset-wide, not just for the yes/no subset) -- HOTPOTQA_GROUND_TRUTH_
ARITY = 2 is this dataset's annotated instrument for E2 to validate tau
against (the development notes Sec. 4 E2, the development notes task D1).

AUTHENTIC POOL: each example's `context` (10 Wikipedia paragraphs, the
"distractor" set HotpotQA itself provides) becomes the Pool's authentic
items, one item per SENTENCE (not per paragraph) so that a t-item group
drawn by arq.pool.draw_groups is comparably fine-grained to CoPHEME's
per-tweet items. The 2 supporting-fact sentences (one per required title)
are exactly the "clean, relevant evidence" a t=2 draw needs to answer
correctly; the remaining ~8 distractor paragraphs' sentences are authentic
in content (真, not adversarial) but irrelevant to this specific claim --
matching the same "mostly irrelevant content surrounding the needle"
structure that motivated CoPHEME's relevance filter (see copheme_adapter.py
module docstring), except HERE it is handled by construction (only 10
paragraphs total, not hundreds) rather than needing a separate relevance
filter.

No adversarial/promoted content exists in HotpotQA itself -- E2 only needs
CLEAN groups (arq.experiments.e2_construct_validity draws at rho_min, the
smallest configured rho, same as e1_calibrate). Every pool is built via
build_pool with an EMPTY promoted list, so `rhos` is IGNORED here and
replaced internally with a single fixed _NO_ADVERSARY_RHO chosen small
enough that build_pool's `need = round(rho*n/(1-rho))` rounds to 0 for
every pool size this dataset can produce (confirmed 2 Oct 2026 by a real
ValueError: CoPHEME-scale rhos like 0.0038 do NOT reliably round to 0 once
a pool has a few hundred items, which many HotpotQA/2Wiki pools do --
caught by an end-to-end test against the real downloaded data, not by
inspection). This is a pool with no real adversary, not a stand-in for
CoPHEME's adversarial pools -- E2 validates tau as an arity-recovery
instrument, which requires no adversary at all, per its own definition
(task_arity depends only on q(t), not on rho). The `rhos` PARAMETER IS KEPT
in this function's signature only so callers (run_e2.py) can pass the same
--rhos flag used elsewhere in this repo without a special case; it has no
effect on the pools actually built.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from arq.pool import Pool, build_pool  # noqa: E402

HOTPOTQA_GROUND_TRUTH_ARITY = 2

# A pool needs at least this many authentic sentence-items to draw a group at
# the largest configured t (t up to 8, matching e1_calibrate/e2's default
# t_values) -- examples with fewer are skipped, not silently kept undersized.
MIN_ITEMS = 8

# E2 pools carry no real adversary (see module docstring) -- this rho is only
# ever used internally so build_pool's adversarial-set bookkeeping has a
# valid (always-empty) entry at every pool size this dataset produces.
# 1e-6 * n stays below build_pool's round() threshold even at n in the
# low thousands (round(1e-6 * 2000 / (1-1e-6)) == 0), far beyond any real
# HotpotQA/2Wiki context size.
_NO_ADVERSARY_RHO = 1e-6


def _load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch it first: "
            "datasets.load_dataset('hotpot_qa', 'distractor', "
            "split='validation').to_json(path) -- see the development notes task A2.")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"{path} is empty")
    return out


def _claim_from_yesno_question(question: str) -> str:
    """Turn a yes/no comparison QUESTION into a declarative CLAIM with no
    content change -- only surface rewording (strip the leading auxiliary
    verb / question mark), so the claim text still comes entirely from the
    dataset's own question, never fabricated. Falls back to the literal
    question text (still a valid claim to verify, just phrased as a
    question) if no known auxiliary prefix matches, rather than guessing a
    rewrite."""
    q = question.strip().rstrip("?").strip()
    lower = q.lower()
    for aux in ("were ", "was ", "are ", "is ", "did ", "do ", "does "):
        if lower.startswith(aux):
            rest = q[len(aux):]
            return f"{rest[0].upper()}{rest[1:]} ({aux.strip()})"
    return q


def build_controlled_pools(examples: List[dict], dataset: str, arity: int,
                           distractor_levels: Sequence[int], n_examples: int,
                           seed: int = 0, transform=None) -> dict:
    """Controlled-density pools for E2 (redesign, 3 Oct 2026).

    Why: the full-context design sampled uniform t-groups from ~37
    sentences, so a group held both supporting facts with probability
    t(t-1)/(37*36) (0.15% at t=2). Replaying 104 HotpotQA pools from the
    timed-out job 408674's cache gave q(t) between 0.03 and 0.18 for
    t=1..8, so that design measured dilution, not arity.

    Each example becomes one pool per level d. The pool holds `arity` gold
    items (one per supporting title, that title's supporting sentences
    joined in sent_id order) plus d distractor sentences drawn from
    non-supporting titles. The same examples are used at every d, and the
    distractor sets are nested (the d=4 set is a prefix of the d=16 set),
    so comparisons across d are paired. Examples are restricted to those
    with at least max(distractor_levels) distractor sentences; this
    favours longer contexts, which matters most for 2Wiki
    bridge_comparison (117 of 429 eligible).

    `examples` must already be normalised to
    {"id", "question", "answer", "context": [(title, [sents])...],
     "supporting": [(title, sent_id)...]}.
    Returns {f"{dataset}_d{d}": [Pool, ...]} for each d.

    transform: optional example -> example rewrite (e.g. synthetic entity
    names, data/synthetic_entities.py). It must keep every sentence list's
    length, so eligibility and the seeded question choice match the
    untransformed run exactly.
    """
    import numpy as np
    if transform is not None:
        if hasattr(transform, "prepare"):
            transform.prepare(examples)
        examples = [transform(r) for r in examples]
    max_d = max(distractor_levels)
    eligible = []
    for r in examples:
        sup_titles = []
        for t, _ in r["supporting"]:
            if t not in sup_titles:
                sup_titles.append(t)
        if len(sup_titles) != arity:
            raise ValueError(f"{r['id']}: {len(sup_titles)} supporting titles, "
                             f"expected {arity}")
        first = {}
        for t, sents in r["context"]:
            first.setdefault(t, sents)
        gold = []
        for t in sup_titles:
            ids = sorted({s for tt, s in r["supporting"] if tt == t})
            gold.append(" ".join(first[t][i].strip() for i in ids))
        distract = [s.strip() for t, sents in r["context"]
                    if t not in sup_titles for s in sents if s.strip()]
        if len(distract) >= max_d:
            eligible.append((r, gold, distract))
    if len(eligible) < n_examples:
        raise RuntimeError(f"{dataset}: only {len(eligible)} examples have "
                           f">= {max_d} distractor sentences, need {n_examples}")
    rng = np.random.default_rng(seed)
    chosen = [eligible[i] for i in sorted(rng.choice(len(eligible), n_examples,
                                                     replace=False))]
    out = {f"{dataset}_d{d}": [] for d in distractor_levels}
    for k, (r, gold, distract) in enumerate(chosen):
        order = np.random.default_rng(seed + 1000 + k).permutation(len(distract))
        claim = _claim_from_yesno_question(r["question"])
        answer = "TRUE" if r["answer"] == "yes" else "FALSE"
        for d in distractor_levels:
            items = gold + [distract[i] for i in order[:d]]
            pool = build_pool(authentic=items, promoted=[],
                              rhos=[_NO_ADVERSARY_RHO], query=claim,
                              answer=answer, seed=seed + k)
            pool.meta.update({"dataset": f"{dataset}_d{d}",
                              "ground_truth_arity": arity,
                              "distractors": d, "example_id": r["id"]})
            out[f"{dataset}_d{d}"].append(pool)
    return out


ADVERSARY_PLACEHOLDER = "ADVERSARY_ITEM_NEVER_SHOWN::"


def build_redundant_pools(base_pools: List[Pool], copies: int = 14,
                          n_distractors: int = 8,
                          adv_counts: Sequence[int] = (1, 2, 4)) -> dict:
    """Pools for E4b (delivered feed vs pool resampling), built from E2's
    d=16 controlled pools so the questions are identical to E2's.

    Authentic items: `copies` copies of each gold item plus the first
    `n_distractors` distractors (the nested prefix E2 also uses), modelling
    a supply in which each relevant fact is reported by many sources. The
    redundancy matters: it keeps q(t) above one half at t = arity in a pool
    large enough to deliver a 12- or 24-item feed from. In E2's diluted
    pools q collapses and neither mechanism could fire.

    Adversarial items are placeholders. E4b forces every group that
    contains one to vote the wrong answer (the worst case Theorem 5.2
    assumes), so their text is never shown to the agent. With 36 authentic
    items, a adversarial items give rho = a/(36+a): about 0.027, 0.053 and
    0.100 for a = 1, 2, 4. build_pool's rounding reproduces a exactly.
    Returns {a: [Pool, ...]}.
    """
    out = {a: [] for a in adv_counts}
    for k, bp in enumerate(base_pools):
        arity = bp.meta["ground_truth_arity"]
        gold, distract = bp.items[:arity], bp.items[arity:arity + n_distractors]
        if len(distract) < n_distractors:
            raise ValueError(f"{bp.meta['example_id']}: too few distractors")
        authentic = [g for g in gold for _ in range(copies)] + distract
        for a in adv_counts:
            rho = a / (len(authentic) + a)
            promoted = [f"{ADVERSARY_PLACEHOLDER}{i}" for i in range(a)]
            pool = build_pool(authentic=authentic, promoted=promoted, rhos=[rho],
                              query=bp.query, answer=bp.answer, seed=k)
            if len(pool.adversarial[rho]) != a:
                raise RuntimeError(f"rho rounding gave {len(pool.adversarial[rho])}"
                                   f" adversarial items, wanted {a}")
            pool.meta.update({k2: v for k2, v in bp.meta.items()
                              if k2 in ("example_id", "ground_truth_arity")})
            pool.meta.update({"dataset": bp.meta["dataset"].rsplit("_d", 1)[0],
                              "copies": copies, "n_adversarial": a})
            out[a].append(pool)
    return out


def build_hotpotqa_controlled_pools(data_dir: str, distractor_levels=(0, 4, 16),
                                    n_examples: int = 40, seed: int = 0,
                                    transform=None, dataset: str = "hotpotqa") -> dict:
    path = os.path.join(data_dir, "hotpotqa_distractor_validation.jsonl")
    rows = [r for r in _load_jsonl(path) if r.get("answer") in ("yes", "no")]
    examples = [{"id": r["id"], "question": r["question"], "answer": r["answer"],
                 "context": list(zip(r["context"]["title"],
                                     r["context"]["sentences"])),
                 "supporting": list(zip(r["supporting_facts"]["title"],
                                        r["supporting_facts"]["sent_id"]))}
                for r in rows]
    return build_controlled_pools(examples, dataset,
                                  HOTPOTQA_GROUND_TRUTH_ARITY,
                                  distractor_levels, n_examples, seed, transform)


def build_hotpotqa_pools(data_dir: str, rhos: Sequence[float], seed: int = 0,
                         limit: Optional[int] = None) -> List[Pool]:
    """Build one Pool per yes/no comparison example in the HotpotQA
    distractor validation split.

    `limit` caps the number of examples used (for a cheaper E2 run); None
    uses every yes/no example found (458 as of the 2 Oct 2026 download).
    Raises if data_dir doesn't contain the expected file, or if zero yes/no
    examples survive (would indicate a schema change upstream, not
    something to silently work around).
    """
    path = os.path.join(data_dir, "hotpotqa_distractor_validation.jsonl")
    records = _load_jsonl(path)
    yesno = [r for r in records if r.get("answer") in ("yes", "no")]
    if not yesno:
        raise RuntimeError(
            f"Zero yes/no comparison examples found in {path} -- expected "
            "458 as of the 2 Oct 2026 HF download. Schema may have changed.")
    if limit is not None:
        yesno = yesno[:limit]

    pools: List[Pool] = []
    skipped = 0
    for i, r in enumerate(yesno):
        sf = r["supporting_facts"]
        n_sf_titles = len(set(sf["title"]))
        if n_sf_titles != HOTPOTQA_GROUND_TRUTH_ARITY:
            # Confirmed dataset-wide invariant (see module docstring); a
            # mismatch here would mean the download/schema differs from
            # what this adapter was built against -- raise loudly rather
            # than silently skip, since the whole ground-truth-arity claim
            # rests on this being exactly 2 always.
            raise ValueError(
                f"example {r['id']}: supporting_facts spans {n_sf_titles} "
                f"titles, expected exactly {HOTPOTQA_GROUND_TRUTH_ARITY} -- "
                "this contradicts the dataset-wide invariant this adapter "
                "was built against (see module docstring); re-verify the "
                "download before trusting ground-truth arity on this "
                "dataset.")

        titles = r["context"]["title"]
        sentences = r["context"]["sentences"]
        items = [s for para in sentences for s in para if s.strip()]
        if len(items) < MIN_ITEMS:
            skipped += 1
            continue

        claim = _claim_from_yesno_question(r["question"])
        answer = "TRUE" if r["answer"] == "yes" else "FALSE"

        pool = build_pool(authentic=items, promoted=[],
                          rhos=[_NO_ADVERSARY_RHO],
                          query=claim, answer=answer, seed=seed + i)
        pool.meta["dataset"] = "hotpotqa"
        pool.meta["ground_truth_arity"] = HOTPOTQA_GROUND_TRUTH_ARITY
        pool.meta["example_id"] = r["id"]
        pool.meta["question_type"] = r["type"]
        pools.append(pool)

    if skipped:
        print(f"NOTE: skipped {skipped} HotpotQA example(s) with fewer than "
              f"{MIN_ITEMS} context sentences.", file=sys.stderr)
    if not pools:
        raise RuntimeError(f"Zero usable pools built from {path}")
    return pools
