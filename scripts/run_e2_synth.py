"""
Driver for E2-synthetic (10_e2_synth_a100.sh): E2's d=0 arity test with every
supporting entity renamed to an invented name (data/synthetic_entities.py),
the paper's control for parametric memory.

Same 40 questions per dataset and seed as E2 (the rewrite keeps sentence
counts, so the seeded choice is identical); only d=0 is run, because the
question is whether tau recovers the annotated arity once memory cannot
supply a missing fact. d=0 pools hold 2 or 4 items, so a seed costs a few
thousand calls; all seeds run in one job.

Writes results/synthetic_entities/E2_<agent>_seed<s>.json.
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
from arq.experiments import e2_construct_validity  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from side_agent import make_agent  # noqa: E402


def build_synth(hotpotqa_dir, twowiki_dir, n_examples, seed):
    from hotpotqa_adapter import build_hotpotqa_controlled_pools
    from twowiki_adapter import build_twowiki_controlled_pools
    from synthetic_entities import make_transform
    # Levels [0, 4, 16] only so eligibility matches E2; d=0 is kept.
    pools = build_hotpotqa_controlled_pools(
        hotpotqa_dir, [0, 4, 16], n_examples, seed,
        transform=make_transform(seed), dataset="hotpotqa_synth")
    pools.update(build_twowiki_controlled_pools(
        twowiki_dir, [0, 4, 16], n_examples, seed,
        transform=make_transform(seed), suffix="_synth"))
    return {k: v for k, v in pools.items() if k.endswith("_d0")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--chat", action="store_true",
                    help="Wrap the prompt in the chat template (side runs only; "
                         "the paper's Qwen runs use the raw prompt).")
    ap.add_argument("--hotpotqa-dir", default="data/hotpotqa")
    ap.add_argument("--twowiki-dir", default="data/2wikimultihopqa")
    ap.add_argument("--outdir", default="results/synthetic_entities")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t-values", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--groups-per-t", type=int, default=80)
    ap.add_argument("--n-examples", type=int, default=40)
    args = ap.parse_args()

    agent = make_agent(args.model_id, args.chat)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)
    status = 0
    for seed in args.seeds:
        name = f"E2_{args.model_id.replace('/', '_')}_seed{seed}.json"
        if os.path.exists(os.path.join(args.outdir, name)):
            print(f"seed {seed} done, skipping")
            continue
        pools = build_synth(args.hotpotqa_dir, args.twowiki_dir,
                            args.n_examples, seed)
        arity = {k: v[0].meta["ground_truth_arity"] for k, v in pools.items()}
        print(f"seed {seed}: " + ", ".join(f"{k}={len(v)} (n={v[0].n})"
                                          for k, v in pools.items()), flush=True)
        res = e2_construct_validity(agent, pools, arity, t_values=args.t_values,
                                    n_groups=args.groups_per_t,
                                    outdir=args.outdir, seed=seed,
                                    falsification_datasets=set(pools))
        final_flush(agent)
        for k, d in res["datasets"].items():
            print(f"  {k}: tau={d.get('tau')} arity={d.get('ground_truth_arity')} "
                  f"q={d.get('q_point')}", flush=True)
        if "FALSIFICATION" in res:
            status = 3
    return status


if __name__ == "__main__":
    raise SystemExit(main())
