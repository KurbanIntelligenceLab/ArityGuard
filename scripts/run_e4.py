"""
Driver invoked by 05_e4_baselines_a100.sh.

E4 compares baselines.py's six methods against ARQ, each at its own budget
and at a budget matched to ARQ's own call count (the development notes Sec. 4 E4,
the development notes task D2): "If ARQ's advantage disappears under matching, that
is the finding." Also records feed_partition's own certificate firing rate
on montage-attacked feeds (the development notes's explicit instruction). Written
2 Oct 2026 -- like E2, this is genuinely new driver code
(code/arq/experiments.py's e4_baselines), not a thin wrapper.

Uses the SAME CoPHEME pools E1/E3/E5 already use (the development notes: CoPHEME is
"the threat's native setting" -- montage-attacked feeds are exactly what
build_copheme_pools produces), built at a single ARQ rho/t/G configuration
matching run_e5.py's feasible defaults (CoPHEME's attack plans cap feasible
rho at ~0.004, same constraint as E1/E3/E5 -- see copheme_adapter.py).
--relevance-filter defaults on, matching the construct-valid 16-pool set
used by every other relevance-filtered run in this repo.

llm_judge's second agent: this script reuses the SAME LocalAgent instance
for both the analyst and judge roles by default (the development notes D2's own note
that a genuinely separate judge agent may need its own GPU) --
e4_baselines records judge_is_agent=True on every llm_judge entry so this
is never silently mistaken for a true two-agent comparison. --judge-model-id
overrides this if a second model is explicitly wanted and GPU memory
allows it.

query_decomposition needs a ground-truth sub-claim decomposition per query,
which CoPHEME's hypothesis/attack-plan schema does not provide (no natural
multi-hop structure the way HotpotQA/2Wiki do) -- this script passes an
empty subclaims_by_query, so every CoPHEME query falls through to
baselines.py's documented "no decomposition available" response rather
than a fabricated split. This is CoPHEME's own synthesis-task nature, not
a gap in this script.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e4_baselines  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--judge-model-id", default=None,
                    help="Second agent for llm_judge. Default: reuse "
                         "--model-id for both roles (see module docstring).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arq-t", type=int, default=1,
                    help="Same default as run_e5.py's --t (the one "
                         "construct-valid tau=1 crossing found so far).")
    ap.add_argument("--arq-G", type=int, default=300,
                    help="ARQ's own call budget per query, matched budget "
                         "for baselines is derived from this (not E5's "
                         "G=1000 -- that was a stress-test-specific "
                         "rescaling, not a baseline-comparison default).")
    ap.add_argument("--arq-rho", type=float, default=0.0038,
                    help="Same CoPHEME feasibility ceiling as run_e1.py/"
                         "run_e5.py's --rho default.")
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.001, 0.002, 0.003, 0.0038])
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--relevance-filter", action="store_true", default=True,
                    help="On by default -- matches the construct-valid "
                         "16-pool set used by every other relevance-"
                         "filtered run in this repo. Pass "
                         "--no-relevance-filter for the full 413-pool set.")
    ap.add_argument("--no-relevance-filter", dest="relevance_filter",
                    action="store_false")
    args = ap.parse_args()

    if args.arq_rho not in args.rhos:
        raise ValueError(f"--arq-rho {args.arq_rho} not in --rhos {args.rhos}")

    try:
        from copheme_adapter import build_copheme_pools  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "data/copheme_adapter.py not found or does not export "
            "build_copheme_pools()."
        ) from e

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"{args.data_dir} does not exist. Download CoPHEME first "
            "(the development notes Sec. 3) -- this script never substitutes data.")

    pools = build_copheme_pools(args.data_dir, rhos=args.rhos, seed=args.seed,
                                relevance_filter=args.relevance_filter)
    if not pools:
        raise RuntimeError(f"build_copheme_pools returned zero pools from "
                           f"{args.data_dir}")
    print(f"Built {len(pools)} pools (seed={args.seed}, "
          f"relevance_filter={args.relevance_filter})")

    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)

    judge = None
    if args.judge_model_id:
        judge = LocalAgent(args.judge_model_id)

    result = e4_baselines(
        agent, pools, arq_t=args.arq_t, arq_G=args.arq_G, arq_rho=args.arq_rho,
        outdir=args.outdir, seed=args.seed, eta=args.eta, judge=judge,
        subclaims_by_query={},
    )

    final_flush(agent)

    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
