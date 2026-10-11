"""
Driver invoked by 07_e6_report_cpu.sh.

E6 (the development notes Sec. 4 E6, the development notes task D3) aggregates cost/transfer
data that OTHER experiments already measured, plus a small fresh pass to
pull real verbatim examples. See code/arq/experiments.py's
e6_cost_transfer docstring for the full scope discussion -- summary:
  - cost per method: read directly from every results/E4_*_seed*.json's
    real per-query "calls" fields (e4_baselines is the only experiment
    that already records this; E1/E3/E5 do not).
  - tau transfer across agents: read directly from every
    results/E1_*_seed*.json found.
  - tokens/dollars per call: NOT reported -- no existing instrumentation
    tracks this, and this driver does not fabricate it (see
    e6_cost_transfer's docstring).
  - verbatim examples (fires/declines/certified-insufficient-evidence/
    certified-but-wrong): requires one small fresh pass over real pools
    with a real agent, since no prior result file retains full per-group
    verdict text. --skip-examples omits this (e.g. for a quick
    aggregation-only check); otherwise this script builds the same
    relevance-filtered CoPHEME pools E3/E4/E5 use and runs a cheap scan
    (small G) looking for one instance of each category.

Unlike run_e1.py/run_e3.py/run_e5.py, this script needs NO GPU for the
aggregation-only part -- it only loads a model if --skip-examples is NOT
passed. Runs on cpu-all by default (07_e6_report_cpu.sh); pass
--skip-examples there is genuinely no model available.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq.experiments import e6_cost_transfer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-examples", action="store_true",
                    help="Aggregation only (cost/transfer); no model is "
                         "loaded, no fresh calls are made. Examples field "
                         "is left absent, not fabricated.")
    ap.add_argument("--example-model-id", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Model to use for the verbatim-example pass "
                         "(ignored if --skip-examples).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--example-t", type=int, default=1)
    ap.add_argument("--example-G", type=int, default=50,
                    help="Small G -- this pass only needs to FIND one "
                         "instance of each category, not measure a rate; "
                         "no need for E5's full G=1000 or even E1's G=300.")
    ap.add_argument("--example-rho", type=float, default=0.0038)
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.001, 0.002, 0.003, 0.0038])
    args = ap.parse_args()

    example_agent = example_pools = None
    if not args.skip_examples:
        try:
            from copheme_adapter import build_copheme_pools  # noqa: E402
        except ImportError as e:
            raise ImportError(
                "data/copheme_adapter.py not found. Pass --skip-examples "
                "to aggregate cost/transfer data only, with no examples."
            ) from e
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(
                f"{args.data_dir} does not exist. Pass --skip-examples to "
                "aggregate cost/transfer data only.")
        example_pools = build_copheme_pools(
            args.data_dir, rhos=args.rhos, seed=args.seed,
            relevance_filter=True)
        from arq.agent import LocalAgent  # noqa: E402
        example_agent = LocalAgent(args.example_model_id)

    result = e6_cost_transfer(
        results_dir=args.results_dir, outdir=args.outdir, seed=args.seed,
        example_agent=example_agent, example_pools=example_pools,
        example_t=args.example_t, example_G=args.example_G,
        example_rho=args.example_rho,
    )

    print(json.dumps(result, indent=2, default=str))

    if result["cost_per_method"] is None:
        print("\nNOTE: no results/E4_*_seed*.json found -- cost_per_method "
              "is null. Run and complete E4 first for real cost data.",
              file=sys.stderr)
    if not result["tau_by_agent"]:
        print("NOTE: no results/E1_*_seed*.json found -- tau_by_agent is "
              "empty.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
