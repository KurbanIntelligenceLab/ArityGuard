"""
Driver invoked by 02_e1_calibrate_a100.sh.

arq.run's CLI raises NotImplementedError as soon as --data is given (this is
deliberate -- see code/arq/run.py -- it refuses to guess a dataset format).
This script is the "call the driver directly" path the development notes C5 and
the development notes both point to: it builds Pool objects via the CoPHEME adapter
(written as part of task A1) and calls e1_calibrate directly.

Requires the CoPHEME adapter to exist at data/copheme_adapter.py exposing
    build_copheme_pools(data_dir: str, rhos: list[float], seed: int) -> list[Pool]
per the development notes task A1. This script does not write that adapter -- it is a
thin wrapper that fails loudly if the adapter or the data are missing.

NOTE (8 Sep 2026, see the development notes sec. "1b. CoPHEME reality check"): the
paper's configured rho_grid (0.02-0.30, code/configs/default.json) is NOT
achievable against the real CoPHEME/Lying_with_Truth data -- each event's
attack plan has only 7 montage posts, capping feasible rho at roughly 0.004.
The default --rhos below reflects that real constraint, not the paper's
sweep. As of 8 Sep 2026 only 3 of 6 events had a usable attack plan on disk
(charliehebdo, ferguson, ottawashooting); generation for the other 3 was
started 9 Sep 2026 (see scripts/07{b,c,d}_generate_montage_*.sh) and may
still be in progress -- n_pools grows as those jobs complete. See
copheme_adapter.py's module docstring for current status.

NOTE (10 Sep 2026): after the 3 new events finished generating (410 new
hypotheses, n_pools 3 -> 413), a full E1 run is ~991k calls per model
(n_pools x len(t_values) x groups_per_t) -- two orders of magnitude beyond
the original 3-pool run. See scripts/resume_agent.py for how this script
survives a SLURM walltime kill at that scale (periodic cache flush + resume
on re-submission) without modifying arq's core e1_calibrate driver.

NOTE (17 Sep 2026): the 413-pool run came back tau=null -- diagnosed as
the model correctly abstaining on claims where a random draw from the full
authentic pool (up to 1821 items) essentially never contains anything
relevant to the specific claim, not a reasoning failure (see the development notes).
--relevance-filter (passed through to copheme_adapter.build_copheme_pools)
restricts each pool to claim-relevant evidence before sampling groups.
This drops n_pools sharply -- measured 17 Sep 2026: 16 of 413 survive both
the relevance filter and a duplicate-claim dedup (CoPHEME repeats the same
claim conclusion across many source tweets) -- but at n=16 a full run is
only ~38k calls/model, far below the resume infrastructure's original
motivating scale. Not decided yet whether this filtered run should be the
one reported in the paper (see the development notes's task-design discussion) --
this flag exists so it CAN be run once that's decided, not as an
endorsement of the result.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e1_calibrate  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--groups-per-t", type=int, default=300)
    ap.add_argument("--rhos", type=float, nargs="+",
                     default=[0.001, 0.002, 0.003, 0.0038],
                     help="Max feasible rho against real CoPHEME attack plans "
                          "is ~0.004 (only 7 promoted posts per event) -- see "
                          "the development notes sec 1b. Not the paper's 0.02-0.30 sweep.")
    ap.add_argument("--relevance-filter", action="store_true",
                     help="Restrict pools to claim-relevant evidence "
                          "(see the development notes 17 Sep 2026 tau=null diagnosis). "
                          "Drops n_pools from 413 to ~16.")
    args = ap.parse_args()

    try:
        from copheme_adapter import build_copheme_pools  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "data/copheme_adapter.py not found or does not export "
            "build_copheme_pools(). Write it per the development notes task A1 before "
            "running this script."
        ) from e

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"{args.data_dir} does not exist. Download CoPHEME first "
            "(the development notes Sec. 3) -- this script never substitutes data."
        )

    pools = build_copheme_pools(args.data_dir, rhos=args.rhos, seed=args.seed,
                                relevance_filter=args.relevance_filter)
    if not pools:
        raise RuntimeError(f"build_copheme_pools returned zero pools from {args.data_dir}")

    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)

    result = e1_calibrate(
        agent, pools,
        t_values=args.t_values,
        n_groups=args.groups_per_t,
        outdir=args.outdir,
        seed=args.seed,
    )

    final_flush(agent)

    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))

    if "FALSIFICATION" in result:
        print("\n*** FALSIFICATION CONDITION HIT. STOP. Read the development notes C1. ***\n",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
