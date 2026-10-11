"""
arq.smoke -- end-to-end pipeline smoke test on synthetic pools and a mock agent.

    python -m arq.smoke

Exercises pool construction, rejection-sampled reuse, calibration, the curve
driver and the stress driver, writing only into results/MOCK_smoke/.

IT PRODUCES NO SCIENTIFIC RESULT. The mock agent's verdicts are a hash of the
group contents. Every output is written under a MOCK-prefixed directory, which
arq.experiments enforces rather than merely recommends.
"""
from __future__ import annotations

import os
import shutil

from .agent import MockAgent
from .experiments import e1_calibrate, e3_curve, e5_stress
from .pool import build_pool, build_pool_displacing, build_synthetic_pool

OUT = "results/MOCK_smoke"


def main() -> int:
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    agent = MockAgent()
    rhos = [0.02, 0.05, 0.10, 0.15]

    pools = []
    for i in range(6):
        auth = [f"SYNTHETIC::authentic fact {i}.{j}" for j in range(200)]
        prom = [f"SYNTHETIC::PROMOTED fact {i}.{j}" for j in range(60)]
        pools.append(build_pool(auth, prom, rhos,
                                query=f"SYNTHETIC::claim {i}", answer="TRUE", seed=i))
    disp = [build_pool_displacing(
                [f"SYNTHETIC::authentic fact {i}.{j}" for j in range(200)],
                [f"SYNTHETIC::PROMOTED fact {i}.{j}" for j in range(60)],
                rhos, query=f"SYNTHETIC::claim {i}", answer="TRUE", seed=i)
            for i in range(6)]

    print("1/4 calibration (E1)")
    r1 = e1_calibrate(agent, pools, t_values=[1, 2, 3], n_groups=40,
                      outdir=OUT, seed=0, no_evidence_control=False)
    qs = {t: round(v, 3) for t, v in r1["q_point"].items()}
    rs = r1["rho_star"]
    print(f"    q={qs} -> tau={r1['tau']}  rho*={rs:.3f}" if rs else f"    q={qs}")

    print("2/4 curve prediction (E3)")
    cal = {p.fingerprint(): 0.9 for p in pools[:3]}
    r3 = e3_curve(agent, pools[3:], t=2, G=60, rho_grid=[0.02, 0.05, 0.10],
                  rho_max=0.15, outdir=OUT, q_by_pool_cal=cal, seed=0)
    print(f"    predicted={ {k: round(v,2) for k,v in r3['predicted'].items()} }")
    print(f"    observed ={ {k: round(v,2) for k,v in r3['observed'].items()} }")

    print("3/4 stress tests (E5)")
    r5 = e5_stress(agent, pools, disp, t=2, G=60, rho=0.10, outdir=OUT, seed=0)
    for row in r5["rows"]:
        print(f"    {row['condition']:32s} certified={row['certified']:2d} "
              f"violations={row['violations']}")

    print("4/4 report generation")
    os.system(f"python -m arq.report --results {OUT} --out {OUT}/generated >/dev/null")
    print(f"    wrote {OUT}/generated/")

    print("\nSMOKE TEST COMPLETE. Pipeline runs end to end.")
    print("These numbers come from a hash-based mock agent and are NOT results.")
    print("They are written under a MOCK-prefixed directory for that reason.")
    print("\nThe two FALSIFICATION alarms above are EXPECTED here and are the")
    print("point of the smoke test: they show the checks fire. A mock agent")
    print("cannot match a curve it does not generate, and the mock displacing")
    print("adversary is not adversarial, so neither alarm indicates a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
