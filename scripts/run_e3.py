"""
Driver invoked by 04_e3_curve_a100.sh.

*** RECONSTRUCTED 29 Sep 2026 ***
An earlier version of this file was overwritten without being read first
-- a real process mistake, not a design choice, and there is no git
history in this repo (the development notes M5) to recover the original from. This
version was rebuilt from 04_e3_curve_a100.sh's actual invocation pattern
(three calls per model: `--split calibration`, `--split evaluation`,
`--split evaluation --forced-worst-case`) and e3_curve's own signature/
docstring in code/arq/experiments.py, which is unchanged and authoritative.
If anything here doesn't match what the original did, 04_e3_curve_a100.sh
is the source of truth for the CLI contract -- fix this file to match it,
not the other way around.

E3 tests whether the certificate's PREDICTED certified rate (from
calibration-half q, via rho_star) matches the OBSERVED certified rate on a
disjoint evaluation half -- out-of-sample, not the same pools used to
compute q. Mixing calibration and evaluation pools is exactly the leakage
e1_calibrate/e3_curve's own docstrings warn against (code/configs/
default.json's "splits" section: "calibration and evaluation halves MUST
be disjoint; q and rho* are estimated on calibration only").

Two-phase design (matching the SLURM script's two/three-call pattern):
  --split calibration:
      Builds pools, splits into calibration/evaluation halves (disjoint,
      deterministic by --seed), computes q_by_pool_cal directly (real
      per-pool accuracy at arity --t on CLEAN groups from the
      calibration half only), and WRITES it to
      <outdir>/../runs_meta/q_by_pool_cal_<tag>_seed<seed>.json (a JSON
      map fingerprint -> q). Prints it too, but the file is what the
      evaluation phase reads.
  --split evaluation [--forced-worst-case]:
      Rebuilds the SAME pools/split (deterministic, so this reproduces
      the calibration phase's split exactly), loads the persisted
      q_by_pool_cal from the calibration phase, and calls e3_curve on
      the EVALUATION half only.

Requires the CoPHEME adapter to exist at data/copheme_adapter.py exposing
    build_copheme_pools(data_dir, rhos, seed, relevance_filter=...) -> list[Pool]
Fails loudly, like run_e1.py, if the adapter, the data, or a missing
calibration-phase file are encountered -- never substitutes anything.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np  # noqa: E402

from arq.agent import LocalAgent  # noqa: E402
from arq.experiments import e3_curve  # noqa: E402
from arq.pool import is_clean  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402


def build_split(data_dir, rhos, seed, relevance_filter):
    """Deterministic calibration/evaluation split by pool index -- same
    pools land in the same half every run at a given seed (not a fresh
    random re-split per invocation), so the calibration and evaluation
    phases (separate process invocations) see the identical split."""
    from copheme_adapter import build_copheme_pools  # noqa: E402
    pools = build_copheme_pools(data_dir, rhos=rhos, seed=seed,
                                relevance_filter=relevance_filter)
    if not pools:
        raise RuntimeError(f"build_copheme_pools returned zero pools from {data_dir}")
    if len(pools) < 2:
        raise RuntimeError(
            f"only {len(pools)} pool(s) -- E3 needs a disjoint "
            "calibration/evaluation split, so at least 2 are required")
    half = len(pools) // 2
    cal_pools, eval_pools = pools[:half], pools[half:]
    if not eval_pools:
        raise RuntimeError("evaluation half is empty -- need at least 2 pools")
    return cal_pools, eval_pools


def compute_q_by_pool_cal(agent, pools, t: int, n_groups: int, seed: int) -> dict:
    """Real per-pool accuracy at arity t on clean groups, for the
    calibration half only. Mirrors e1_calibrate's own per-pool accuracy
    definition (is_clean at the pool's minimum built rho), computed here
    directly since e1_calibrate itself only reports pooled, not per-pool,
    q."""
    rng = np.random.default_rng(seed)
    q_by_pool = {}
    for p in pools:
        if p.answer is None:
            raise ValueError(
                f"pool {p.fingerprint()} has no gold answer; E3's "
                "calibration half needs labels to estimate q, same as E1")
        rho_min = min(p.adversarial)
        succ = trials = 0
        drawn = 0
        while drawn < n_groups:
            g = [int(x) for x in rng.choice(p.n, size=t, replace=False)]
            if not is_clean(p, g, rho_min):
                continue
            v = agent.verdict([p.items[i] for i in g], p.query)
            succ += int(v == p.answer)
            trials += 1
            drawn += 1
        q_by_pool[p.fingerprint()] = succ / trials
    return q_by_pool


def cal_cache_path(outdir: str, model_id: str, seed: int, relevance_filter: bool) -> str:
    tag = model_id.replace("/", "_")
    suffix = "_relfilt" if relevance_filter else ""
    meta_dir = os.path.join(outdir, "..", "runs_meta")
    os.makedirs(meta_dir, exist_ok=True)
    return os.path.join(meta_dir, f"q_by_pool_cal_{tag}{suffix}_seed{seed}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["calibration", "evaluation"], required=True,
                    help="calibration: compute and persist q_by_pool_cal. "
                         "evaluation: load it and run e3_curve on the "
                         "evaluation half.")
    ap.add_argument("--t", type=int, default=1,
                    help="Arity to evaluate at. Default 1, matching the "
                         "one construct-valid tau=1 crossing found so far "
                         "(14B, relevance-filtered, contradicts prompt, "
                         "29 Sep 2026 -- see the development notes).")
    ap.add_argument("--groups-per-pool-cal", type=int, default=300,
                    help="Groups drawn per calibration-half pool to "
                         "estimate q_by_pool_cal (calibration phase only). "
                         "Matches E1's groups-per-t default.")
    ap.add_argument("--G", type=int, default=300,
                    help="Groups drawn per evaluation-half pool at "
                         "rho_max, then reused via rejection sampling "
                         "across the rho_grid (Sec. 6 reuse; evaluation "
                         "phase only).")
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.001, 0.002, 0.003, 0.0038],
                    help="Pool-construction rhos. Matches run_e1.py's "
                         "default -- max feasible rho against real "
                         "CoPHEME attack plans is ~0.004 (the development notes "
                         "sec 1b), not the paper's 0.02-0.30 sweep.")
    ap.add_argument("--rho-grid", type=float, nargs="+",
                    default=[0.001, 0.002, 0.003, 0.0038],
                    help="rho values to evaluate the curve at (evaluation "
                         "phase only). Must all be present in --rhos.")
    ap.add_argument("--rho-max", type=float, default=0.0038,
                    help="Must be one of --rhos; groups are drawn once "
                         "here and reused for every smaller rho in "
                         "--rho-grid (evaluation phase only).")
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--forced-worst-case", action="store_true",
                    help="Tight two-sided test: assume the adversary wins "
                         "every non-clean group instead of querying the "
                         "agent on it. Evaluation phase only. See "
                         "e3_curve's own docstring.")
    ap.add_argument("--relevance-filter", action="store_true",
                    help="Restrict pools to claim-relevant evidence, same "
                         "flag as run_e1.py --relevance-filter. Matches "
                         "the 16-pool set where the tau=1 crossing was "
                         "found; omit for the full 413-pool set.")
    args = ap.parse_args()

    try:
        import copheme_adapter  # noqa: F401,E402
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

    for r in args.rho_grid:
        if r not in args.rhos:
            raise ValueError(
                f"--rho-grid value {r} not in --rhos {args.rhos}; pools "
                "are only built at the --rhos values, so every "
                "--rho-grid entry must be a subset of them")
    if args.rho_max not in args.rhos:
        raise ValueError(f"--rho-max {args.rho_max} not in --rhos {args.rhos}")

    cal_pools, eval_pools = build_split(
        args.data_dir, args.rhos, args.seed, args.relevance_filter)
    cal_path = cal_cache_path(args.outdir, args.model_id, args.seed,
                              args.relevance_filter)

    agent = LocalAgent(args.model_id)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)

    if args.split == "calibration":
        print(f"Split: {len(cal_pools)} calibration / {len(eval_pools)} "
              f"evaluation pools (disjoint, seed={args.seed})")
        print(f"Computing q_by_pool_cal on {len(cal_pools)} calibration "
              f"pools at t={args.t}, {args.groups_per_pool_cal} groups/pool...")
        q_by_pool_cal = compute_q_by_pool_cal(
            agent, cal_pools, t=args.t, n_groups=args.groups_per_pool_cal,
            seed=args.seed)
        final_flush(agent)
        with open(cal_path, "w") as fh:
            json.dump(q_by_pool_cal, fh, indent=1)
        print(f"Wrote q_by_pool_cal ({len(q_by_pool_cal)} pools) to {cal_path}")
        print(json.dumps({k: round(v, 3) for k, v in q_by_pool_cal.items()}, indent=2))
        return 0

    # --split evaluation
    if not os.path.exists(cal_path):
        raise FileNotFoundError(
            f"{cal_path} not found -- run with --split calibration first "
            "(same --model-id/--seed/--relevance-filter). Never inventing "
            "a calibration q in its place.")
    with open(cal_path) as fh:
        q_by_pool_cal = json.load(fh)
    print(f"Loaded q_by_pool_cal ({len(q_by_pool_cal)} pools) from {cal_path}")

    result = e3_curve(
        agent, eval_pools, t=args.t, G=args.G,
        rho_grid=args.rho_grid, rho_max=args.rho_max,
        outdir=args.outdir, q_by_pool_cal=q_by_pool_cal, seed=args.seed,
        eta=args.eta, forced_worst_case=args.forced_worst_case,
    )

    final_flush(agent)

    print(json.dumps(result, indent=2))

    if "FALSIFICATION" in result:
        print("\n*** FALSIFICATION CONDITION HIT. STOP. Read the development notes. ***\n",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
