"""
Driver invoked by 08_e5b_ablations_a100.sh: Table 4's arity, budget and
design-choice ablations (code/arq/experiments.py:e5b_ablations), on the E4b
pools (arity 2, worst-case adversary) at the middle share, 2 promoted items
(rho = 0.053), both datasets pooled.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e5b_ablations  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from run_e4b import build_e4b_pools  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-adversarial", type=int, default=2)
    ap.add_argument("--retrieve-r", type=int, default=12)
    args = ap.parse_args()

    # Seed s uses E4b's seed-s questions (seed 0 is unchanged), so every
    # seed's clean votes overlap E4b's cache.
    by_ds = build_e4b_pools("data/hotpotqa", "data/2wikimultihopqa", 40, args.seed,
                            14, 8, [args.n_adversarial])
    pools = [p for by_a in by_ds.values() for p in by_a[args.n_adversarial]]
    print(f"{len(pools)} pools, rho={next(iter(pools[0].adversarial)):.3f}")

    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)
    out = e5b_ablations(agent, pools, outdir=args.outdir, seed=args.seed,
                        retrieve_r=args.retrieve_r)
    final_flush(agent)
    for r in out["rows"]:
        print(f"{r['condition']:18s} cert={r['certified']}/{r['n_pools']} "
              f"correct={r['certified_correct']} viol={r['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
