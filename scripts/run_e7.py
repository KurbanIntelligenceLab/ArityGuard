"""
Driver for E7, the misspecification sweep (code/arq/experiments.py:e7_misspec):
certify against an assumed share while the worst-case adversary holds a
different one, on the E4b pools and groups.

Every clean vote it needs was already cast in E4b (same pools, same group
seeds), so --cache-only runs it with no model at all: the agent raises on a
cache miss instead of loading a model. That is how the paper's E7 numbers
were produced (no GPU). Without --cache-only it falls back to LocalAgent.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from arq.agent import Agent, CachingAgent, LocalAgent  # noqa: E402
from arq.experiments import e7_misspec  # noqa: E402
from run_e4b import build_e4b_pools  # noqa: E402


class CacheOnly(Agent):
    """Renders exactly as LocalAgent does (Agent._render) but never calls a
    model: any cache miss is an error, so the run provably reuses E4b votes."""
    def __init__(self, model_id: str):
        super().__init__(name=model_id)

    def verdict(self, items, claim):
        raise RuntimeError("cache miss in --cache-only mode: this group was not "
                           "voted on in E4b; run without --cache-only on a GPU")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--G", type=int, default=200)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    inner = CacheOnly(args.model_id) if args.cache_only else LocalAgent(args.model_id)
    agent = CachingAgent(inner, args.cache)
    for seed in args.seeds:
        pools = build_e4b_pools("data/hotpotqa", "data/2wikimultihopqa", 40, seed,
                                14, 8, [1, 2, 4])
        out = e7_misspec(agent, pools, ts=args.ts, G=args.G, outdir=args.outdir,
                         seed=seed)
        v = sum(r["violations"] for r in out["rows"])
        print(f"seed {seed}: {len(out['rows'])} rows, {v} violations, "
              f"cache hits={agent.hits} misses={agent.misses}", flush=True)
    if not args.cache_only:
        agent.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
