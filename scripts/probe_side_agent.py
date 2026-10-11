"""
Preflight for a new model in the SIDE runs: does it answer the production
prompt with a parseable verdict? Runs ~24 real verdicts (no cache) on groups
of the kinds the side jobs use: multi-hop feed-vs-pool groups (t = 1..3) and
CoPHEME groups that include a montage post. Exit 0 if every answer parses.

  python3 scripts/probe_side_agent.py --model-id M [--chat]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))

from side_agent import make_agent, probe  # noqa: E402


def cases(n_multi=16, n_coal=8):
    from run_e4b import build_e4b_pools
    import run_side_coalition as rc
    rng = np.random.default_rng(0)
    out = []
    pools = build_e4b_pools("data/hotpotqa", "data/2wikimultihopqa", 40, 0,
                            14, 8, [1])
    flat = [p for ds in pools.values() for p in ds[1]]
    for i in range(n_multi):
        p = flat[(i * 7) % len(flat)]
        t = 1 + i % 3
        n_auth = p.meta["n_authentic"]
        g = rng.choice(n_auth, size=t, replace=False)
        out.append(([p.items[j] for j in g], p.query))
    cpools = rc.build_pools("data", 0.1, 0)
    for i in range(n_coal):
        p = cpools[i % len(cpools)]
        adv = p.adversarial[0.1]
        g = [adv[0], int(rng.integers(p.meta["n_authentic"]))]
        out.append(([p.items[j] for j in g], p.query))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    cs = cases()
    if args.model_id == "none":          # build-only check, no model
        print(f"{len(cs)} probe cases built")
        return 0
    bad = probe(make_agent(args.model_id, args.chat), cs)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
