"""
Driver invoked by 05b_e4b_feed_vs_pool_a100.sh.

E4b tests the paper's central regime (Figure 1): a certificate computed over
a k-item delivered feed, partitioned into groups of t > 1, against ARQ's pool
resampling. Both face the same worst-case selection adversary. E4 could not
test this: it handed the RobustRAG mechanism the whole pool, and at tau=1 the
two mechanisms coincided. See code/arq/experiments.py:e4b_feed_vs_pool and
data/hotpotqa_adapter.py:build_redundant_pools for the design.

Questions: the same 40 per dataset as E2 (HotpotQA and 2Wiki comparison,
both arity 2), taken from E2's d=16 pools.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e4b_feed_vs_pool  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from side_agent import make_agent  # noqa: E402
from run_e2 import build_all  # noqa: E402
from hotpotqa_adapter import build_redundant_pools  # noqa: E402

DATASETS = ("hotpotqa", "2wiki_comparison")


def build_e4b_pools(hotpotqa_dir, twowiki_dir, n_examples, seed, copies,
                    n_distractors, adv_counts):
    base = build_all(hotpotqa_dir, twowiki_dir, [0, 4, 16], n_examples, seed)
    return {ds: build_redundant_pools(base[f"{ds}_d16"], copies, n_distractors,
                                      adv_counts)
            for ds in DATASETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--chat", action="store_true",
                    help="Wrap the prompt in the chat template (side runs only; "
                         "the paper's Qwen runs use the raw prompt).")
    ap.add_argument("--hotpotqa-dir", default="data/hotpotqa")
    ap.add_argument("--twowiki-dir", default="data/2wikimultihopqa")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=40,
                    help="Must match E2's value so the questions are the same.")
    ap.add_argument("--copies", type=int, default=14)
    ap.add_argument("--n-distractors", type=int, default=8)
    ap.add_argument("--adv-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ks", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--G", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.01)
    args = ap.parse_args()

    pools = build_e4b_pools(args.hotpotqa_dir, args.twowiki_dir, args.n_examples,
                            args.seed, args.copies, args.n_distractors,
                            args.adv_counts)
    for ds, by_a in pools.items():
        p = by_a[args.adv_counts[0]][0]
        print(f"{ds}: {len(by_a[args.adv_counts[0]])} pools, n_authentic="
              f"{p.meta['n_authentic']}, rho by adversarial count: " +
              ", ".join(f"{a}->{next(iter(v[0].adversarial)):.3f}"
                        for a, v in by_a.items()))

    agent = make_agent(args.model_id, args.chat)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)
    result = e4b_feed_vs_pool(agent, pools, ts=args.ts, ks=args.ks, G=args.G,
                              outdir=args.outdir, seed=args.seed, eta=args.eta)
    final_flush(agent)

    for r in result["rows"]:
        print(f"{r['dataset']:18s} a={r['n_adversarial']} rho={r['rho']:.3f} "
              f"t={r['t']} {r['mechanism']:4s} k={str(r['k']):4s} "
              f"cert={r['certified']}/{r['n_pools']} correct={r['certified_correct']} "
              f"viol={r['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
