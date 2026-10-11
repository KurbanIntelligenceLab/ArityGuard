"""
Real-text coalition (results/side/coalition/; supplementary Table
"A coalition of real text", built by arq.report's table_coalition).

Reviewer objection addressed: outside CoPHEME the coalition is simulated
(contaminated groups are forced to vote the wrong answer, and the agent never
reads the coalition's items), and on CoPHEME the real montage posts make up at
most 0.0038 of the pool, far below where the arity bound matters.

Here the coalition is real text at realistic shares. Sybil accounts repost
the montage posts of the attack plan (Hu et al.'s own data) until they hold a
share rho in {0.027, 0.053, 0.100} of each relevance-filtered CoPHEME pool,
the same shares as the feed-against-pool experiment. The agent READS every
group, contaminated or not; no vote is forced. Reposts cycle through a seeded
permutation of the plan's posts, so low shares use distinct posts first. Every
pool gets at least one coalition item; the certificate's budget ceil(rho*n)
still covers it, because ceil(rho*n) >= 1.

Mechanisms per (rho, t), t in {1, 2}:
  arq           G resampled t-groups, certify() with Dbar.
  feed k share  k-item feed in disjoint t-groups, budget ceil(rho*k).
  feed k tail   same feed, budget = (1-eta) quantile of the hypergeometric
                count of coalition items in a k-item draw (sound budget).
Feeds with k > pool size are recorded as not applicable.

Per pool we record the certificate outcome, soundness (certified verdict
strictly wins among clean groups; Proposition 1's event), whether the
certified verdict is TRUE (the montage's target: every CoPHEME claim is false
or unverified), and the TRUE share among clean vs contaminated groups, which
measures whether the reposted montage actually moves the agent.

Seeds redraw repost order, groups and feeds; the 16 hypotheses are fixed.
One process loads the model once and loops over seeds, skipping any seed
whose result exists.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)

import copheme_adapter  # noqa: E402
from arq.certificate import certify  # noqa: E402
from arq.pool import Pool, draw_groups, is_clean  # noqa: E402
from resume_agent import wrap_with_resume, final_flush  # noqa: E402
from side_agent import make_agent  # noqa: E402

RHOS = (0.027, 0.053, 0.100)


def _reposting_builder(authentic, promoted, rhos, query, answer=None, seed=0):
    """Drop-in for pool.build_pool with one rho: the coalition reposts its
    posts until it holds share rho (at least one item)."""
    (rho,) = rhos
    rng = np.random.default_rng(seed)
    order = [promoted[i] for i in rng.permutation(len(promoted))]
    need = max(1, int(round(rho * len(authentic) / (1 - rho))))
    adv = [order[i % len(order)] for i in range(need)]
    items = list(authentic) + adv
    return Pool(items=items,
                adversarial={rho: list(range(len(authentic), len(items)))},
                query=query, answer=answer,
                meta={"seed": seed, "n_authentic": len(authentic),
                      "n_coalition": need, "n_distinct_posts": len(promoted),
                      "actual_share": need / len(items)})


def build_pools(data_dir, rho, seed):
    real = copheme_adapter.build_pool
    copheme_adapter.build_pool = _reposting_builder
    try:
        return copheme_adapter.build_copheme_pools(
            data_dir, rhos=[rho], seed=seed * 1000, relevance_filter=True)
    finally:
        copheme_adapter.build_pool = real


def tail_budget(N, m, k, eta):
    """Smallest b with P[X <= b] >= 1 - eta, X ~ Hypergeom(N, m, k)."""
    total = math.comb(N, k)
    acc = 0
    for b in range(0, min(m, k) + 1):
        acc += math.comb(m, b) * math.comb(N - m, k - b)
        if acc / total >= 1 - eta:
            return b
    return min(m, k)


def plurality_sound(clean_votes, winner):
    c = Counter(clean_votes)
    return all(c[winner] > n for v, n in c.items() if v != winner)


def run_seed(agent, data_dir, seed, ts, ks, G, eta):
    rng = np.random.default_rng(seed + 7)
    rows = []
    for rho in RHOS:
        pools = build_pools(data_dir, rho, seed)
        for t in ts:
            recs = {"arq": []}
            for k in ks:
                recs[f"feed{k}_share"] = []
                recs[f"feed{k}_tail"] = []
            for j, p in enumerate(pools):
                adv = set(p.adversarial[rho])

                def votes(groups):
                    return [agent.verdict([p.items[i] for i in g], p.query)
                            for g in groups]

                groups = draw_groups(p, t, G, rho, seed=seed * 100003 + j)
                vs = votes(groups)
                cl = [is_clean(p, g, rho) for g in groups]
                clean_v = [v for v, c in zip(vs, cl) if c]
                cont_v = [v for v, c in zip(vs, cl) if not c]
                r = certify(vs, p.n, rho, t, eta)
                base = {"event": p.meta["event_id"],
                        "hypothesis": p.meta["hypothesis_index"],
                        "gold": p.answer, "n": p.n,
                        "n_coalition": p.meta["n_coalition"]}
                recs["arq"].append({
                    **base, "certified": r.certified, "verdict": r.verdict,
                    "correct": r.verdict == p.answer,
                    "sound": (not r.certified) or plurality_sound(clean_v, r.winner),
                    "target": r.certified and r.verdict == "TRUE",
                    "margin": r.margin, "threshold": r.threshold,
                    "n_clean": len(clean_v), "n_contaminated": len(cont_v),
                    "clean_counts": dict(Counter(clean_v)),
                    "contaminated_counts": dict(Counter(cont_v))})
                for k in ks:
                    if k > p.n:
                        for b in ("share", "tail"):
                            recs[f"feed{k}_{b}"].append({**base, "applicable": False})
                        continue
                    feed = [int(x) for x in rng.choice(p.n, size=k, replace=False)]
                    fg = [feed[i:i + t] for i in range(0, k - k % t, t)]
                    fv = votes(fg)
                    fclean = [v for v, g in zip(fv, fg) if not (set(g) & adv)]
                    ranked = Counter(fv).most_common()
                    c1 = ranked[0][1]
                    c2 = ranked[1][1] if len(ranked) > 1 else 0
                    budgets = {"share": math.ceil(rho * k),
                               "tail": tail_budget(p.n, len(adv), k, eta)}
                    for b, kp in budgets.items():
                        ok = c1 - c2 > kp
                        w = ranked[0][0]
                        recs[f"feed{k}_{b}"].append({
                            **base, "applicable": True, "certified": ok,
                            "verdict": w if ok else None,
                            "correct": ok and w == p.answer,
                            "sound": (not ok) or plurality_sound(fclean, w),
                            "target": ok and w == "TRUE",
                            "margin": c1 - c2, "threshold": kp,
                            "n_coalition_in_feed": len(set(feed) & adv)})
            for mech, rs in recs.items():
                app = [x for x in rs if x.get("applicable", True)]
                cert = [x for x in app if x["certified"]]
                rows.append({"rho": rho, "t": t, "mechanism": mech,
                             "n_pools": len(rs), "n_applicable": len(app),
                             "certified": len(cert),
                             "certified_correct": sum(x["correct"] for x in cert),
                             "violations": sum(not x["sound"] for x in cert),
                             "certified_target": sum(x["target"] for x in cert),
                             "per_pool": rs})
                print(f"seed={seed} rho={rho:.3f} t={t} {mech:12s} "
                      f"cert={len(cert)}/{len(app)} viol={rows[-1]['violations']} "
                      f"target={rows[-1]['certified_target']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="results/side/coalition")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--G", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--mock", action="store_true",
                    help="MockAgent pipeline test; outdir must start with MOCK")
    args = ap.parse_args()

    if args.mock:
        from arq.agent import MockAgent
        if not os.path.basename(args.outdir.rstrip("/")).startswith("MOCK"):
            raise SystemExit("--mock needs an outdir named MOCK*")
        agent = MockAgent()
    else:
        agent = make_agent(args.model_id, args.chat)
    if args.cache:
        agent = wrap_with_resume(agent, args.cache)
    os.makedirs(args.outdir, exist_ok=True)
    tag = agent.name.replace("/", "_")
    for seed in args.seeds:
        path = os.path.join(args.outdir, f"COAL_{tag}_seed{seed}.json")
        if os.path.exists(path):
            print(f"seed {seed}: {path} exists, skipping")
            continue
        rows = run_seed(agent, args.data_dir, seed, args.ts, args.ks, args.G, args.eta)
        with open(path, "w") as fh:
            json.dump({"experiment": "side_coalition", "agent": agent.name,
                       "seed": seed, "G": args.G, "eta": args.eta,
                       "rhos": list(RHOS), "rows": rows}, fh)
        print(f"wrote {path}", flush=True)
        if args.cache:
            final_flush(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
