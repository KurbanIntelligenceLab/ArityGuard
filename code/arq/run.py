"""
arq.run -- command-line entry point for the experiments.

    python -m arq.run e1 --agent <id> --data <dir> --outdir results/

Deliberately thin: it parses arguments, builds the agent and pools, and calls
the drivers in arq.experiments. All logic lives there so that it is testable
without a CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import APIAgent, CachingAgent, LocalAgent, MockAgent
from .experiments import e1_calibrate, e3_curve, e5_stress


def build_agent(spec: str, cache: str | None = None):
    if spec == "mock":
        a = MockAgent()
    elif spec.startswith("api:"):
        a = APIAgent(spec[4:], endpoint=os.environ.get("ARQ_ENDPOINT", ""))
    else:
        a = LocalAgent(spec)
    return CachingAgent(a, cache) if cache else a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arq.run", description=__doc__)
    ap.add_argument("experiment", choices=["e1", "e2", "e3", "e4", "e5", "e6"])
    ap.add_argument("--agent", required=True, help="'mock', 'api:<id>', or a HF model id")
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--outdir", default="results/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--forced-worst-case", action="store_true")
    ap.add_argument("--split", choices=["calibration", "evaluation"], default="evaluation")
    ap.add_argument("--match-budget", action="store_true")
    a = ap.parse_args(argv)

    with open(a.config) as fh:
        cfg = json.load(fh)
    agent = build_agent(a.agent, a.cache)

    if not a.data:
        print("No --data given. Build pools with arq.pool and pass a directory; "
              "this command will not substitute synthetic data.", file=sys.stderr)
        print("For a pipeline smoke test use:  python -m arq.smoke", file=sys.stderr)
        return 2

    raise NotImplementedError(
        f"Dataset adapters for {a.data} are not written: they depend on the exact\n"
        "release format of CoPHEME and the QA sets, which must be confirmed on\n"
        "download (see the development notes Sec. 3). Write the adapter that turns your\n"
        "files into arq.pool.Pool objects, then call the driver directly:\n"
        "    from arq.experiments import e1_calibrate\n"
        "    e1_calibrate(agent, pools, t_values, n_groups, outdir, seed)\n"
        "This command fails loudly rather than guessing a file format.")


if __name__ == "__main__":
    raise SystemExit(main())
