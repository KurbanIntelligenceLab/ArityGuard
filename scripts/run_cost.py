"""
Driver invoked by 09_cost_a100.sh: measured cost per method (paper E8),
which no earlier run recorded. For each method on a few CoPHEME
relevance-filtered pools, with NO cache (so every call is real), it records
agent calls, prompt tokens (counted with the model's own tokenizer) and
wall-clock GPU seconds per query.

Dollars are not reported: the agents are local open-weight models with no
per-token price. GPU-seconds is the measured quantity.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq import baselines as bl  # noqa: E402
from arq.agent import Agent, LocalAgent  # noqa: E402
from arq.certificate import certify  # noqa: E402
from arq.pool import draw_groups, pool_size_at, reject_to_rho  # noqa: E402


class Meter(Agent):
    def __init__(self, inner: LocalAgent):
        super().__init__(name=inner.name)
        self.inner = inner
        self.calls = self.tokens = 0
        self.seconds = 0.0

    def verdict(self, items, claim):
        self.inner._load()
        self.tokens += len(self.inner._tok(self.inner._render(items, claim))["input_ids"])
        t0 = time.perf_counter()
        v = self.inner.verdict(items, claim)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return v

    def reset(self):
        self.calls = self.tokens = 0
        self.seconds = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-pools", type=int, default=4)
    ap.add_argument("--G", type=int, default=300)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.0038)
    args = ap.parse_args()

    from copheme_adapter import build_copheme_pools
    rhos = [0.001, 0.002, 0.003, 0.0038]
    pools = build_copheme_pools(args.data_dir, rhos=rhos, seed=0,
                                relevance_filter=True)[:args.n_pools]
    m = Meter(LocalAgent(args.model_id))
    m.verdict([pools[0].items[0]], pools[0].query)   # load the model before timing
    m.reset()

    methods = {
        "undefended": lambda p: bl.undefended(m, p.items, p.query),
        "feed_partition_omega1": lambda p: bl.feed_partition(m, p.items, p.query, 1, 1),
        "feed_partition_omega3": lambda p: bl.feed_partition(m, p.items, p.query, 3, 1),
        "majority_vote": lambda p: bl.majority_vote(m, p.items, p.query, K=5),
        "llm_judge": lambda p: bl.llm_judge(m, m, p.items, p.query, K=5),
        "influence_detector": lambda p: bl.influence_detector(m, p.items, p.query),
    }

    def arq(p):
        groups = draw_groups(p, args.t, args.G, max(p.adversarial), seed=0)
        kept = reject_to_rho(p, groups, args.rho, max(p.adversarial))
        vs = [m.verdict([p.items[i] for i in g], p.query) for g in kept]
        return certify(vs, pool_size_at(p, args.rho), args.rho, args.t)
    methods["arq"] = arq

    out = {"experiment": "COST", "agent": args.model_id, "n_pools": len(pools),
           "pool_sizes": [p.n for p in pools], "methods": {}}
    for name, fn in methods.items():
        per = []
        for p in pools:
            m.reset()
            fn(p)
            per.append({"calls": m.calls, "prompt_tokens": m.tokens,
                        "seconds": m.seconds})
        n = len(per)
        out["methods"][name] = {
            "calls_per_query": sum(x["calls"] for x in per) / n,
            "prompt_tokens_per_query": sum(x["prompt_tokens"] for x in per) / n,
            "gpu_seconds_per_query": sum(x["seconds"] for x in per) / n,
            "per_pool": per}
        print(f"{name:24s} calls={out['methods'][name]['calls_per_query']:.0f} "
              f"tokens={out['methods'][name]['prompt_tokens_per_query']:.0f} "
              f"sec={out['methods'][name]['gpu_seconds_per_query']:.1f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir,
                        f"COST_{args.model_id.replace('/', '_')}_seed0.json")
    out["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
