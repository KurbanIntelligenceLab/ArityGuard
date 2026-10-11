"""
Driver for E4c (11_e4c_robustrag_a100.sh): the E4b comparison -- delivered
feed of k items vs ARQ's pool resampling, worst-case adversary -- on
RobustRAG's own RealtimeQA and Natural Questions data
(data/robustrag_adapter.py). Writes results/robustrag/E4b_<agent>_seed<s>.json
(same schema as E4b, so report.table_feedpool reads it unchanged).
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e4b_feed_vs_pool  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from robustrag_adapter import build_robustrag_pools  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", default="data/RobustRAG/data")
    ap.add_argument("--outdir", default="results/robustrag")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=40)
    ap.add_argument("--n-passages", type=int, default=30)
    ap.add_argument("--adv-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--G", type=int, default=200)
    args = ap.parse_args()

    pools = build_robustrag_pools(args.data_dir, n_examples=args.n_examples,
                                  n_passages=args.n_passages,
                                  adv_counts=args.adv_counts, seed=args.seed)
    for ds, by_a in pools.items():
        print(f"{ds}: {len(by_a[args.adv_counts[0]])} questions, rho by count: " +
              ", ".join(f"{a}->{next(iter(v[0].adversarial)):.3f}" for a, v in by_a.items()),
              flush=True)
    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)
    res = e4b_feed_vs_pool(agent, pools, ts=args.ts, ks=args.ks, G=args.G,
                           outdir=args.outdir, seed=args.seed)
    final_flush(agent)
    for r in res["rows"]:
        print(f"{r['dataset']:11s} a={r['n_adversarial']} t={r['t']} {r['mechanism']:4s} "
              f"k={str(r['k']):4s} cert={r['certified']}/{r['n_pools']} "
              f"correct={r['certified_correct']} viol={r['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
