"""
Driver invoked by 03_e2_construct_validity_a100.sh.

E2 asks whether tau measures arity (the development notes Sec. 4 E2, the development notes D1).

REDESIGNED 3 Oct 2026. The first version sampled uniform t-groups from each
question's full ~37-sentence context. Job 408674 timed out at about 5% of the
7B run (a full run would take about 19 days), and replaying its cache over
104 complete HotpotQA pools gave q(t) of 0.03 to 0.18 for t=1..8. A random
group almost never holds both supporting facts (0.15% at t=2), so that design
measured dilution, not arity.

This version uses controlled-density pools (see
hotpotqa_adapter.build_controlled_pools): `arity` gold items (one per
supporting title) plus d distractor sentences, with d in --distractors. The
same questions appear at every d, and the distractor sets are nested.
  d = 0  : tests whether tau recovers semantic arity (2, 2, 4). Only these
           datasets can trigger the FALSIFICATION flag.
  d > 0  : shows how tau grows as the pool is diluted. Under uniform pool
           sampling this is what ARQ actually faces, so it is reported as a
           result, not treated as a failure.

Datasets in the result: hotpotqa_d{d}, 2wiki_comparison_d{d},
2wiki_bridge_comparison_d{d}. Ground-truth arity is 2, 2 and 4.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e2_construct_validity  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from side_agent import make_agent  # noqa: E402


def build_all(hotpotqa_dir, twowiki_dir, distractors, n_examples, seed):
    from hotpotqa_adapter import build_hotpotqa_controlled_pools
    from twowiki_adapter import build_twowiki_controlled_pools
    pools = build_hotpotqa_controlled_pools(hotpotqa_dir, distractors,
                                            n_examples, seed)
    pools.update(build_twowiki_controlled_pools(twowiki_dir, distractors,
                                                n_examples, seed))
    return pools


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
    ap.add_argument("--t-values", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="Capped per dataset at the pool size.")
    ap.add_argument("--groups-per-t", type=int, default=80)
    ap.add_argument("--distractors", type=int, nargs="+", default=[0, 4, 16])
    ap.add_argument("--n-examples", type=int, default=40,
                    help="Questions per dataset, shared across distractor "
                         "levels.")
    args = ap.parse_args()

    for d in (args.hotpotqa_dir, args.twowiki_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} does not exist; see the adapters' "
                                    "module docstrings for the fetch command.")

    pools_by_dataset = build_all(args.hotpotqa_dir, args.twowiki_dir,
                                 args.distractors, args.n_examples, args.seed)
    ground_truth_arity = {name: pools[0].meta["ground_truth_arity"]
                          for name, pools in pools_by_dataset.items()}
    falsification = {name for name in pools_by_dataset if name.endswith("_d0")}
    print("Built pools: " + ", ".join(
        f"{name}={len(p)} (arity {ground_truth_arity[name]}, n={p[0].n})"
        for name, p in pools_by_dataset.items()))

    agent = make_agent(args.model_id, args.chat)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)

    result = e2_construct_validity(
        agent, pools_by_dataset, ground_truth_arity,
        t_values=args.t_values, n_groups=args.groups_per_t,
        outdir=args.outdir, seed=args.seed,
        falsification_datasets=falsification,
    )
    final_flush(agent)

    summary = {name: {k: v for k, v in d.items() if k != "trials"}
               for name, d in result["datasets"].items()}
    print(json.dumps({**{k: v for k, v in result.items() if k != "datasets"},
                      "datasets": summary}, indent=2))
    if "FALSIFICATION" in result:
        print("\n*** FALSIFICATION CONDITION HIT. Read the development notes. ***\n",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
