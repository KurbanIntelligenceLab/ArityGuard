"""
Driver invoked by 06_e5_stress_h200.sh.

06_e5_stress_h200.sh originally called `python3 -m arq.run e5 --agent
"$MODEL" --seed 0 --outdir results/` with NO --data argument -- arq.run's
own CLI (code/arq/run.py) prints "No --data given" and exits 2 immediately
whenever --data is empty, before ever reaching the NotImplementedError
that --data triggers. That invocation could never have produced a result;
it was a real gap, not a "deliberate stub to work around," and needed the
same direct-call treatment run_e1.py/run_e3.py already got for E1/E3
(the development notes C5/D1-D4, "arq.run still raises NotImplementedError on
--data -- the SLURM scripts bypass it by calling drivers directly").
06_e5_stress_h200.sh updated in the same change to call this script
instead of `python -m arq.run e5`.

E5 runs the assumption stress tests (code/arq/experiments.py:e5_stress):
augmenting-adversary rows at rho_assumed == rho_true and rho_assumed >
rho_true, plus displacing-adversary rows at two thresholds. Rows 3 and 5
(the displacing rows) are PREDICTED to show violations -- e5_stress's own
FALSIFICATION check fires if the "displacing, threshold Dbar" row shows
ZERO violations despite certifying, which would indicate the displacing
condition isn't actually implemented, not that the mechanism is sound
there.

Requires TWO pool sets, built from the SAME underlying CoPHEME hypotheses
so the augmenting/displacing rows are comparable: `pools` (the standard
augmenting-adversary construction) and `pools_displacing` (same claims,
built via copheme_adapter's displacing=True, which switches to
arq.pool.build_pool_displacing per its own docstring). Both use the same
--seed so the underlying claim/pool ordering matches.

rho defaults to 0.0038 (not the paper's 0.02-0.30 sweep) -- max feasible
rho against real CoPHEME attack plans is ~0.004 (only 7 montage posts per
event; see the development notes sec 1b and run_e1.py's own --rhos default), same
constraint applies here.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e5_stress  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=int, default=1,
                    help="Arity to stress-test at. Default 1, matching "
                         "the one construct-valid tau=1 crossing found "
                         "so far (14B, relevance-filtered, contradicts "
                         "prompt, 29 Sep 2026 -- see the development notes).")
    ap.add_argument("--G", type=int, default=1000,
                    help="Groups drawn per pool at rho_max for each "
                         "stress condition. Raised 300->1000 (30 Sep "
                         "2026, alongside --rho's bump to the CoPHEME "
                         "ceiling) to give the displacing-adversary "
                         "stress test (row 'displacing, threshold Dbar') "
                         "a real chance to observe a violation -- at "
                         "rho=0.002/G=300 the expected number of "
                         "contaminated groups per pool was only ~1-27 "
                         "depending on pool size, while Dbar itself was "
                         "4-39 votes, making a violation a genuine rare "
                         "tail event rather than evidence the displacing "
                         "condition is unimplemented (see the development notes' "
                         "30 Sep 2026 entries for the full arithmetic). "
                         "This is a statistical-power fix, not a "
                         "guarantee of observing a violation -- Dbar is "
                         "a concentration bound specifically constructed "
                         "to stay ahead of the expected contaminated "
                         "count, so larger G raises the number of trials "
                         "without proportionally raising the odds of any "
                         "single one landing in the tail.")
    ap.add_argument("--rho", type=float, default=0.0038,
                    help="Adversary share for the stress conditions. "
                         "Raised 0.002->0.0038 (30 Sep 2026) -- the max "
                         "feasible rho against real CoPHEME attack plans "
                         "(~0.004, see the development notes sec 1b, run_e1.py's "
                         "--rhos default) -- to raise the contaminated-"
                         "vote fraction directly, the more reliable "
                         "lever for giving the displacing-adversary "
                         "stress test a fair chance (see --G's help "
                         "text). NOT the paper's 0.02-0.30 sweep, which "
                         "real CoPHEME attack plans cannot supply.")
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.001, 0.0019, 0.002, 0.003, 0.0038],
                    help="Pool-construction rho grid (both pools and "
                         "pools_displacing are built at these). --rho "
                         "must be one of these values, AND --rho/2 must "
                         "also be one of these values (or smaller than "
                         "all of them) -- see the --rho help text. "
                         "0.0019 (== 0.0038/2) added 30 Sep 2026 "
                         "specifically so the new --rho default's "
                         "'augmenting, rho_a > rho_t' row "
                         "(max(min(pool_rhos), rho/2)) has a valid grid "
                         "value to land on -- confirmed by the same "
                         "KeyError-avoidance logic that caught this for "
                         "the previous rho=0.002 default.")
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--relevance-filter", action="store_true",
                    help="Restrict pools to claim-relevant evidence, same "
                         "flag as run_e1.py/run_e3.py. Matches the "
                         "16-pool set where the tau=1 crossing was found; "
                         "omit for the full 413-pool set.")
    args = ap.parse_args()

    if args.rho not in args.rhos:
        raise ValueError(f"--rho {args.rho} not in --rhos {args.rhos}")
    half = args.rho / 2
    if half > min(args.rhos) and half not in args.rhos:
        raise ValueError(
            f"--rho/2 ({half}) is neither in --rhos {args.rhos} nor "
            f"below its minimum ({min(args.rhos)}) -- e5_stress's "
            "'augmenting, rho_a > rho_t' row computes "
            "max(min(pool_rhos), rho/2), which reject_to_rho requires "
            "to be a value the pools were built at (else KeyError). Pick "
            "a smaller --rho, or add --rho/2's value to --rhos.")

    try:
        from copheme_adapter import build_copheme_pools  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "data/copheme_adapter.py not found or does not export "
            "build_copheme_pools(). Write it per the development notes task A1 "
            "before running this script."
        ) from e

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"{args.data_dir} does not exist. Download CoPHEME first "
            "(the development notes Sec. 3) -- this script never substitutes data.")

    pools = build_copheme_pools(args.data_dir, rhos=args.rhos, seed=args.seed,
                                relevance_filter=args.relevance_filter)
    pools_displacing = build_copheme_pools(
        args.data_dir, rhos=args.rhos, seed=args.seed, displacing=True,
        relevance_filter=args.relevance_filter)
    if not pools or not pools_displacing:
        raise RuntimeError(
            f"build_copheme_pools returned zero pools "
            f"(augmenting={len(pools)}, displacing={len(pools_displacing)}) "
            f"from {args.data_dir}")
    print(f"Built {len(pools)} augmenting pools, "
          f"{len(pools_displacing)} displacing pools (seed={args.seed})")

    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)

    result = e5_stress(
        agent, pools, pools_displacing, t=args.t, G=args.G, rho=args.rho,
        outdir=args.outdir, seed=args.seed, eta=args.eta,
    )

    final_flush(agent)

    print(json.dumps(result, indent=2))

    for row in result["rows"]:
        if "FALSIFICATION" in row:
            print(f"\n*** FALSIFICATION CONDITION HIT on row "
                  f"{row['condition']!r}. STOP. Read the development notes. ***\n",
                  file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
