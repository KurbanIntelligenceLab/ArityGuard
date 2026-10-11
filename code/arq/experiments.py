"""
arq.experiments -- drivers for E1..E6 of Section 6, plus the ablations.

Each driver writes one JSON file per (experiment, agent, seed) into results/.
Nothing is aggregated in memory across runs, so a crash costs one file.

FALSIFICATION CONDITIONS are checked and printed by the drivers themselves, so
an outcome that contradicts the paper is surfaced rather than buried:
  E1  tau >= 5 on a synthesis task  -> the mechanism is undeployable
  E2  recovered tau does not match a dataset's annotated ground-truth arity
      -> tau is not a valid instrument for arity, withdraw its downstream use
  E3  observed curve below predicted -> Theorem 5.2 is wrong
  E5  displacing adversary at Dbar shows NO violations -> the condition is not
      implemented, and its negative result means nothing

A run whose agent is a MockAgent may only write into a directory whose name
begins with MOCK. This is enforced, not advised.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .agent import Agent
from .certificate import (ABSTAIN, certify, clean_probability, rho_star,
                          task_arity, wilson)
from .pool import Pool, draw_groups, is_clean, pool_size_at, reject_to_rho


def _guard(agent: Agent, outdir: str) -> None:
    if agent.is_mock() and not os.path.basename(outdir.rstrip("/")).startswith("MOCK"):
        raise RuntimeError(
            "refusing to write MockAgent output into a real results directory. "
            "Use --outdir results/MOCK_... for pipeline smoke tests.")
    os.makedirs(outdir, exist_ok=True)


def _write(outdir: str, name: str, payload: dict) -> str:
    # agent.name is often a raw HF model id (e.g. "Qwen/Qwen2.5-7B-Instruct")
    # and name is built by interpolating it into a single path component
    # (e.g. f"E1_{agent.name}_seed{seed}.json"). Sanitize any path separator
    # so that slash doesn't turn into an unintended subdirectory that
    # open(..., "w") then fails to find.
    name = name.replace("/", "_").replace(os.sep, "_")
    path = os.path.join(outdir, name)
    payload["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


# ---------------------------------------------------------------------------
def e1_calibrate(agent: Agent, pools: Sequence[Pool], t_values: Sequence[int],
                 n_groups: int, outdir: str, seed: int = 0,
                 no_evidence_control: bool = True) -> dict:
    """E1: measure q(t) on CLEAN groups, derive tau, and run the two controls.

    Requires gold answers: q is an accuracy. ARQ itself needs no labels at
    inference; only this calibration does.
    """
    _guard(agent, outdir)
    if any(p.answer is None for p in pools):
        raise ValueError("E1 needs gold answers on every pool; q is an accuracy")
    rng = np.random.default_rng(seed)
    succ, trials = {t: 0 for t in t_values}, {t: 0 for t in t_values}
    contaminated_by_memory = []

    for p in pools:
        if no_evidence_control and p.answer != "ABSTAIN":
            # A query answered correctly with NO evidence is contaminated by
            # parametric memory and is reported separately (Sec. 6, E1).
            # Skipped when the gold answer is ABSTAIN: the prompt itself
            # instructs the agent to answer ABSTAIN when evidence is
            # insufficient, and zero evidence is insufficient by
            # construction, so an ABSTAIN-labeled pool would pass this
            # control regardless of whether the agent has any real memory
            # of the query -- the control is uninformative there, not just
            # weak evidence (confirmed empirically 9 Sep 2026: two
            # different model sizes flagged the identical ABSTAIN-labeled
            # pools and none of the FALSE-labeled ones, consistent with a
            # structural default rather than memorization; see the development notes).
            if agent.verdict([], p.query) == p.answer:
                contaminated_by_memory.append(p.fingerprint())
        rho_min = min(p.adversarial)
        for t in t_values:
            drawn = 0
            while drawn < n_groups:
                g = [int(x) for x in rng.choice(p.n, size=t, replace=False)]
                if not is_clean(p, g, rho_min):
                    continue
                v = agent.verdict([p.items[i] for i in g], p.query)
                succ[t] += int(v == p.answer)
                trials[t] += 1
                drawn += 1

    q = {t: wilson(succ[t], trials[t]) for t in t_values}
    tau = task_arity(succ, trials)
    out = {"experiment": "E1", "agent": agent.name, "seed": seed,
           "q_point": {t: q[t][0] for t in t_values},
           "q_lower": {t: q[t][1] for t in t_values},
           "q_upper": {t: q[t][2] for t in t_values},
           "trials": trials, "tau": tau,
           "rho_star": rho_star(tau, q[tau][1]) if tau else None,
           "memory_contaminated_queries": contaminated_by_memory,
           "n_pools": len(pools)}
    if tau is not None and tau >= 5:
        out["FALSIFICATION"] = ("tau >= 5: rho* < 0.08, no realistic deployment "
                                "is certifiable. Report as an impossibility "
                                "result and stop.")
        print("\n*** " + out["FALSIFICATION"] + " ***\n")
    _write(outdir, f"E1_{agent.name}_seed{seed}.json", out)
    return out


def e2_construct_validity(agent: Agent, pools_by_dataset: Dict[str, Sequence[Pool]],
                          ground_truth_arity: Dict[str, int], t_values: Sequence[int],
                          n_groups: int, outdir: str, seed: int = 0,
                          falsification_datasets: Optional[set] = None) -> dict:
    """E2: does tau measure arity? (construct validity)

    falsification_datasets: if given, only these datasets can trigger the
    FALSIFICATION flag. Diluted controlled pools (d > 0) are expected to
    give tau above the semantic arity; that growth is a result to report,
    not a failure of the instrument. Only the undiluted d=0 pools test
    whether tau recovers arity.

    Runs the SAME per-t clean-group calibration procedure as e1_calibrate,
    independently per dataset, then compares each dataset's recovered tau
    against its OWN annotated ground-truth arity (the development notes task D1):
    HotpotQA's supporting_facts always span exactly 2 titles (arity 2);
    2WikiMultihopQA's bridge_comparison chains span 4 (the rest span 2) --
    see data/hotpotqa_adapter.py / data/twowiki_adapter.py docstrings for how
    ground_truth_arity is derived per pool and passed in here per dataset.

    pools_by_dataset: {dataset_name: pools}, each pool's `answer` the gold
    verdict needed for the accuracy computation (same requirement as E1).
    ground_truth_arity: {dataset_name: annotated arity}, the instrument this
    experiment validates tau against -- NOT estimated from pools_by_dataset,
    must come from the adapter's own annotation.

    Recovering the annotated arity (tau == ground_truth) per dataset
    validates tau as an instrument; failing to recover it withdraws every
    downstream use of tau (the development notes Sec. 4, E2). This is reported
    per-dataset, not pooled, since the whole point is to compare tau against
    each dataset's OWN distinct ground truth.
    """
    _guard(agent, outdir)
    if set(pools_by_dataset) != set(ground_truth_arity):
        raise ValueError(
            f"pools_by_dataset datasets {sorted(pools_by_dataset)} do not "
            f"match ground_truth_arity datasets {sorted(ground_truth_arity)}")
    rng = np.random.default_rng(seed)
    datasets = {}
    mismatches = []
    for name, pools in pools_by_dataset.items():
        if any(p.answer is None for p in pools):
            raise ValueError(f"E2 needs gold answers on every {name} pool; "
                             "q is an accuracy")
        # A group cannot exceed its pool; controlled pools can hold as few
        # as `arity` items, so t is capped per dataset at the smallest pool.
        n_min = min(p.n for p in pools)
        ts = [t for t in t_values if t <= n_min]
        succ, trials = {t: 0 for t in ts}, {t: 0 for t in ts}
        for p in pools:
            rho_min = min(p.adversarial)
            for t in ts:
                drawn = 0
                while drawn < n_groups:
                    g = [int(x) for x in rng.choice(p.n, size=t, replace=False)]
                    if not is_clean(p, g, rho_min):
                        continue
                    v = agent.verdict([p.items[i] for i in g], p.query)
                    succ[t] += int(v == p.answer)
                    trials[t] += 1
                    drawn += 1
        q = {t: wilson(succ[t], trials[t]) for t in ts}
        tau = task_arity(succ, trials)
        gt = ground_truth_arity[name]
        recovered = tau == gt
        if not recovered and (falsification_datasets is None
                              or name in falsification_datasets):
            mismatches.append(name)
        datasets[name] = {
            "q_point": {t: q[t][0] for t in ts},
            "q_lower": {t: q[t][1] for t in ts},
            "q_upper": {t: q[t][2] for t in ts},
            "trials": trials, "tau": tau,
            "ground_truth_arity": gt, "recovered": recovered,
            "n_pools": len(pools), "pool_size_min": n_min,
        }
    out = {"experiment": "E2", "agent": agent.name, "seed": seed,
           "datasets": datasets}
    if mismatches:
        out["FALSIFICATION"] = (
            f"tau failed to recover the annotated ground-truth arity on "
            f"{mismatches}; tau is not a valid instrument for arity on "
            "these datasets, and every downstream use of tau is withdrawn "
            "until this is understood.")
        print("\n*** " + out["FALSIFICATION"] + " ***\n")
    _write(outdir, f"E2_{agent.name}_seed{seed}.json", out)
    return out


def e3_curve(agent: Agent, pools: Sequence[Pool], t: int, G: int,
             rho_grid: Sequence[float], rho_max: float, outdir: str,
             q_by_pool_cal: Dict[str, float], seed: int = 0,
             eta: float = 0.01, forced_worst_case: bool = False) -> dict:
    """E3: predicted vs observed certified-rate curve, out of sample.

    q_by_pool_cal comes from the CALIBRATION half; the pools passed here must be
    the disjoint EVALUATION half. Mixing them is the leakage the paper's E3 was
    redesigned to avoid.
    """
    _guard(agent, outdir)
    observed, predicted = {}, {}
    for rho in rho_grid:
        # prediction: fraction of calibration queries whose own rho* exceeds rho
        predicted[rho] = float(np.mean([rho < rho_star(t, qq)
                                        for qq in q_by_pool_cal.values()]))
        fired = 0
        for p in pools:
            groups = draw_groups(p, t, G, rho_max, seed=seed)
            kept = reject_to_rho(p, groups, rho, rho_max)
            verdicts = []
            for g in kept:
                if forced_worst_case and not is_clean(p, g, rho):
                    verdicts.append("ADVERSARY")   # tight two-sided test
                else:
                    verdicts.append(agent.verdict([p.items[i] for i in g], p.query))
            r = certify(verdicts, pool_size_at(p, rho), rho, t, eta)
            fired += int(r.certified)
        observed[rho] = fired / len(pools)

    mae = float(np.mean([abs(observed[r] - predicted[r]) for r in rho_grid]))
    out = {"experiment": "E3", "agent": agent.name, "seed": seed, "t": t, "G": G,
           "forced_worst_case": forced_worst_case,
           "predicted": predicted, "observed": observed, "curve_mae": mae}
    below = [r for r in rho_grid if observed[r] < predicted[r] - 0.05]
    if below:
        out["FALSIFICATION"] = (f"observed curve below prediction at rho={below}; "
                                "this indicts Theorem 5.2, not the mechanism")
        print("\n*** " + out["FALSIFICATION"] + " ***\n")
    _write(outdir, f"E3_{agent.name}_t{t}_seed{seed}"
                   f"{'_forced' if forced_worst_case else ''}.json", out)
    return out


def e5_stress(agent: Agent, pools: Sequence[Pool], pools_displacing: Sequence[Pool],
              t: int, G: int, rho: float, outdir: str, seed: int = 0,
              eta: float = 0.01) -> dict:
    """E5: the assumption stress tests. Rows 3 and 5 are PREDICTED to violate."""
    _guard(agent, outdir)
    rows = []

    def _one(label, ps, adversary, rho_assumed, rho_true):
        viol = certified = 0
        for p in ps:
            groups = draw_groups(p, t, G, max(p.adversarial), seed=seed)
            kept = reject_to_rho(p, groups, rho_true, max(p.adversarial))
            vs = [agent.verdict([p.items[i] for i in g], p.query) for g in kept]
            r = certify(vs, pool_size_at(p, rho_assumed), rho_assumed, t, eta,
                        adversary=adversary)
            if r.certified:
                certified += 1
                clean_v = [v for v, g in zip(vs, kept) if is_clean(p, g, rho_true)]
                if clean_v:
                    cc = Counter(clean_v)
                    if r.winner != cc.most_common(1)[0][0]:
                        viol += 1
        rows.append({"condition": label, "adversary": adversary,
                     "rho_assumed": rho_assumed, "rho_true": rho_true,
                     "n_pools": len(ps),
                     "certified": certified, "violations": viol,
                     "violation_rate": viol / max(1, certified),
                     "eta": eta})

    _one("augmenting, rho_a = rho_t", pools, "augment", rho, rho)
    _one("augmenting, rho_a > rho_t", pools, "augment", rho, max(min(pools[0].adversarial), rho / 2))
    _one("displacing, threshold Dbar", pools_displacing, "augment", rho, rho)
    _one("displacing, threshold 2Dbar", pools_displacing, "displace", rho, rho)

    disp = rows[2]
    if disp["certified"] > 0 and disp["violations"] == 0:
        rows[2]["FALSIFICATION"] = (
            "no violations under a displacing adversary at threshold Dbar. "
            "Exhaustive search predicts violations here, so the displacing "
            "condition is probably not implemented; its negative result means "
            "nothing until this is fixed.")
        print("\n*** " + rows[2]["FALSIFICATION"] + " ***\n")
    out = {"experiment": "E5", "agent": agent.name, "seed": seed, "rows": rows}
    _write(outdir, f"E5_{agent.name}_seed{seed}.json", out)
    return out


def e6_cost_transfer(results_dir: str, outdir: str, seed: int = 0,
                     example_agent: Optional[Agent] = None,
                     example_pools: Optional[Sequence[Pool]] = None,
                     example_t: int = 1, example_G: int = 300,
                     example_rho: float = 0.0038, eta: float = 0.01
                     ) -> dict:
    """E6: cost per method, tau transfer across agents, verbatim examples
    (the development notes task D3, the development notes Sec. 4 E6 / paper's E8-E10).

    Unlike E1-E5, this is primarily an AGGREGATION step over result files
    E4 (and, where present, E1) have already written -- not a fresh
    model-calling experiment for its cost/transfer parts. This is a
    deliberate scope decision, not a shortcut: neither e1_calibrate,
    e3_curve, nor e5_stress persists a per-method call count or any
    timing/token data to their result JSON (confirmed 2 Oct 2026 by
    reading every result-writing path in this file and in
    code/arq/agent.py -- CachingAgent tracks hits/misses in memory but
    never writes them to a result file, and no experiment tracks latency
    or token counts at all), so there is no existing data to aggregate
    tokens/latency from for E1/E3/E5 specifically. e4_baselines is the
    ONE experiment whose result file already carries a real, per-query,
    per-method "calls" count (baselines.py's own return dicts) -- this
    function reads that, rather than re-deriving or re-measuring it.

    `calls` per method: read directly from every results/E4_*_seed*.json
    found in results_dir, averaged per method across all per_query
    entries and all E4 result files present (so multiple agents'
    E4 runs are naturally aggregated, not just the first found).
    Tokens and dollars per call are NOT reported here -- this driver has
    no access to per-call token counts (LocalAgent does not record them;
    adding that would require changing agent.py's production call path,
    which this function does not do) or to a real API pricing table for
    the local open-weight models this repo actually runs. Reporting
    plausible-looking placeholder numbers for either would violate the
    "no number enters the paper except via a real measurement" house
    rule just as much as a hand-typed guess would -- so these fields are
    left absent from the output rather than fabricated, and the paper's
    own cost table must mark them \\RES until a token/cost-tracking
    change is made deliberately, not silently defaulted to by this
    function.

    `tau` transfer across agents: read directly from every
    results/E1_*_seed*.json found, reporting each agent's own tau/q(tau)
    side by side -- "transfer" here means literally comparing the already
    -measured tau per agent, not re-deriving anything.

    Verbatim examples: four categories per the development notes ("fires / declines
    / certified-insufficient-evidence / certified-but-wrong"). Finding a
    real instance of each requires the full per-group verdict list for
    some query, which no existing result file retains (e1_calibrate/
    e3_curve/e5_stress all report only AGGREGATE counts, by design, to
    keep result files small at the 991k-call E1 scale) -- so this is the
    one part of E6 that genuinely needs a small fresh pass over real
    pools with a real agent, via `example_agent`/`example_pools`. If
    neither is supplied, this driver reports what it found from result
    files alone and marks the examples field explicitly absent rather
    than fabricating illustrative text.
    """
    import glob

    def _load_all(pattern: str) -> List[dict]:
        out = []
        for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
            with open(path) as fh:
                out.append(json.load(fh))
        return out

    e4_results = _load_all("E4_*_seed*.json")
    e1_results = _load_all("E1_*_seed*.json")

    # --- cost per method, aggregated across every E4 result file found ---
    # e4_baselines' own per_query entries are mostly flat ({method: {calls:
    # ...}}), but feed_partition_own_budget is ONE LEVEL DEEPER, keyed by
    # omega ({method: {"omega=1": {calls: ...}, "omega=3": {...}}}) --
    # found by testing against a real E4 result file, not by inspection: a
    # flat-only check silently dropped both omega conditions' call counts
    # from the cost table entirely. Each omega is reported as its own
    # "method" (e.g. "feed_partition_own_budget[omega=1]") so the table
    # doesn't conflate the two budgets.
    call_sums: Dict[str, List[int]] = {}
    for r in e4_results:
        for entry in r.get("per_query", []):
            for method, data in entry.items():
                if method in ("query", "gold", "n_items"):
                    continue
                if isinstance(data, dict) and "calls" in data:
                    call_sums.setdefault(method, []).append(data["calls"])
                elif isinstance(data, dict):
                    for sub_key, sub_data in data.items():
                        if isinstance(sub_data, dict) and "calls" in sub_data:
                            call_sums.setdefault(f"{method}[{sub_key}]", []
                                                 ).append(sub_data["calls"])
    cost_per_method = {
        method: {"mean_calls": float(np.mean(v)), "n_queries": len(v)}
        for method, v in call_sums.items()
    }
    if not e4_results:
        cost_per_method = None  # no E4 result exists yet; nothing to aggregate

    # --- tau transfer across agents, read verbatim from each E1 result ---
    tau_by_agent = {
        r["agent"]: {"tau": r.get("tau"),
                    "q_tau": (r["q_lower"].get(str(r["tau"]), r["q_lower"].get(r["tau"]))
                              if r.get("tau") is not None else None),
                    "n_pools": r.get("n_pools")}
        for r in e1_results
    }

    # --- verbatim examples: one small fresh pass, only if an agent+pools
    # are supplied; otherwise explicitly absent, never fabricated ---
    examples = None
    if example_agent is not None and example_pools:
        examples = {"fires": None, "declines": None,
                    "certified_insufficient_evidence": None,
                    "certified_but_wrong": None}
        for p in example_pools:
            if all(examples.values()):
                break
            groups = draw_groups(p, example_t, example_G,
                                 max(p.adversarial), seed=seed)
            kept = reject_to_rho(p, groups, example_rho, max(p.adversarial))
            vs = [example_agent.verdict([p.items[i] for i in g], p.query)
                 for g in kept]
            r = certify(vs, pool_size_at(p, example_rho), example_rho,
                       example_t, eta)
            record = {"query": p.query, "gold": p.answer,
                     "verdict": r.verdict, "certified": r.certified,
                     "margin": r.margin, "threshold": r.threshold,
                     "counts": r.counts}
            if r.certified and r.verdict == p.answer and examples["fires"] is None:
                examples["fires"] = record
            elif not r.certified and examples["declines"] is None:
                examples["declines"] = record
            elif (r.certified and r.verdict == ABSTAIN
                  and examples["certified_insufficient_evidence"] is None):
                examples["certified_insufficient_evidence"] = record
            elif (r.certified and r.verdict != p.answer and r.verdict != ABSTAIN
                  and examples["certified_but_wrong"] is None):
                examples["certified_but_wrong"] = record

    out = {"experiment": "E6", "seed": seed,
          "cost_per_method": cost_per_method,
          "tau_by_agent": tau_by_agent,
          "examples": examples,
          "example_agent": example_agent.name if example_agent else None,
          "n_e4_results_aggregated": len(e4_results),
          "n_e1_results_aggregated": len(e1_results)}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"E6_seed{seed}.json")
    out["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _plurality_sound(verdicts: Sequence[str], winner: str) -> bool:
    """Proposition 5.1's event: winner strictly outpolls every other verdict
    among the given (clean) verdicts."""
    c = Counter(verdicts)
    return all(c[winner] > n for v, n in c.items() if v != winner)


def e4b_feed_vs_pool(agent: Agent, pools_by_level: Dict[str, Dict[int, Sequence[Pool]]],
                     ts: Sequence[int], ks: Sequence[int], G: int, outdir: str,
                     seed: int = 0, eta: float = 0.01) -> dict:
    """E4b: the Figure-1 comparison with a real agent. Does a certificate
    computed over a k-item delivered feed fire at t > 1, where pool
    resampling does?

    Both mechanisms face the same worst-case selection adversary: any group
    containing an adversarial item votes the wrong answer (the adversary's
    target), as Theorem 5.2 assumes. Clean groups get the agent's real
    verdict. Adversarial items are therefore never rendered to the agent
    (asserted below).

      feed: k items drawn uniformly from the pool (the delivered feed),
            partitioned into floor(k/t) disjoint groups of t; certify iff
            c1 - c2 > k', with k' = ceil(rho*k) the corruption budget a
            deployer who knows rho must assume (RobustRAG's guarantee).
      arq:  G uniform t-groups from the pool; certify(...) with Dbar.

    pools_by_level: {dataset: {n_adversarial: [Pool]}}, each pool built at
    exactly one rho. Reported per (dataset, n_adversarial, t, mechanism, k):
    certified rate, certified-and-correct rate, and soundness violations
    (certified verdict does not strictly win among the clean groups).
    """
    import math
    _guard(agent, outdir)
    rng = np.random.default_rng(seed)
    rows = []

    def vote(p, g, rho, target):
        if not is_clean(p, g, rho):
            return target
        items = [p.items[i] for i in g]
        assert not any(s.startswith("ADVERSARY_ITEM_NEVER_SHOWN::") for s in items)
        return agent.verdict(items, p.query)

    def summarise(dataset, a, rho, t, mech, k, recs):
        n = len(recs)
        cert = [r for r in recs if r["certified"]]
        rows.append({"dataset": dataset, "n_adversarial": a, "rho": rho, "t": t,
                     "mechanism": mech, "k": k, "n_pools": n,
                     "certified": len(cert),
                     "certified_correct": sum(r["correct"] for r in cert),
                     "violations": sum(not r["sound"] for r in cert),
                     "mean_groups": float(np.mean([r["groups"] for r in recs])),
                     "per_pool": recs})

    for dataset, by_a in pools_by_level.items():
        for a, pools in sorted(by_a.items()):
            rho = next(iter(pools[0].adversarial))
            for t in ts:
                arq_recs = []
                feed_recs = {k: [] for k in ks}
                for j, p in enumerate(pools):
                    target = "FALSE" if p.answer == "TRUE" else "TRUE"
                    groups = draw_groups(p, t, G, rho, seed=seed * 100003 + j)
                    vs = [vote(p, g, rho, target) for g in groups]
                    clean = [v for v, g in zip(vs, groups) if is_clean(p, g, rho)]
                    r = certify(vs, pool_size_at(p, rho), rho, t, eta)
                    arq_recs.append({"certified": r.certified, "verdict": r.verdict,
                                     "correct": r.verdict == p.answer,
                                     "sound": (not r.certified) or _plurality_sound(clean, r.winner),
                                     "groups": len(groups), "margin": r.margin,
                                     "threshold": r.threshold})
                    for k in ks:
                        feed = [int(x) for x in rng.choice(p.n, size=k, replace=False)]
                        fg = [feed[i:i + t] for i in range(0, k - k % t, t)]
                        fv = [vote(p, g, rho, target) for g in fg]
                        fclean = [v for v, g in zip(fv, fg) if is_clean(p, g, rho)]
                        ranked = Counter(fv).most_common()
                        c1 = ranked[0][1]
                        c2 = ranked[1][1] if len(ranked) > 1 else 0
                        kprime = math.ceil(rho * k)
                        ok = c1 - c2 > kprime
                        feed_recs[k].append({"certified": ok,
                                             "verdict": ranked[0][0] if ok else None,
                                             "correct": ok and ranked[0][0] == p.answer,
                                             "sound": (not ok) or _plurality_sound(fclean, ranked[0][0]),
                                             "groups": len(fg), "margin": c1 - c2,
                                             "threshold": kprime})
                summarise(dataset, a, rho, t, "arq", None, arq_recs)
                for k in ks:
                    summarise(dataset, a, rho, t, "feed", k, feed_recs[k])

    out = {"experiment": "E4b", "agent": agent.name, "seed": seed, "G": G,
           "eta": eta, "rows": rows}
    _write(outdir, f"E4b_{agent.name}_seed{seed}.json", out)
    return out


ABLATION_CONDITIONS = [
    # (key, t, G, order, abstention, sampling)
    ("t1", 1, 200, "random", "counted", "pool"),
    ("t_tau", 2, 200, "random", "counted", "pool"),
    ("t_tau_plus1", 3, 200, "random", "counted", "pool"),
    ("G50", 2, 50, "random", "counted", "pool"),
    ("G200", 2, 200, "random", "counted", "pool"),
    ("G500", 2, 500, "random", "counted", "pool"),
    ("fixed_order", 2, 200, "sorted", "counted", "pool"),
    ("abstain_discarded", 2, 200, "random", "discarded", "pool"),
    ("retrieval", 2, 200, "random", "counted", "retrieved"),
]


def _lexical_rank(items: Sequence[str], query: str) -> List[int]:
    """Indices of items ordered by word overlap with the query (a stand-in
    retriever: deterministic, no model, ties broken by index)."""
    import re
    qw = set(re.findall(r"[a-z0-9]+", query.lower()))
    score = [len(qw & set(re.findall(r"[a-z0-9]+", s.lower()))) for s in items]
    return sorted(range(len(items)), key=lambda i: (-score[i], i))


def e5b_ablations(agent: Agent, pools: Sequence[Pool], outdir: str,
                  seed: int = 0, eta: float = 0.01, retrieve_r: int = 12,
                  conditions=ABLATION_CONDITIONS) -> dict:
    """E5 ablations (Table 4's arity, budget and design-choice rows) on pools
    with a worst-case adversary, where a soundness violation can occur.

    Same adversary as e4b_feed_vs_pool: every group holding a promoted item
    votes the wrong answer, and promoted items are never shown to the agent.
    Conditions:
      t, G           arity and budget sweeps; tau = 2 is the annotated arity.
      fixed_order    each group's items in sorted index order rather than
                     random order (rng.choice already randomizes, so
                     draw_groups(randomize_order=False) would not fix it).
                     Prediction: certified rate unchanged.
      abstain_discarded  ABSTAIN votes dropped before certify(). Soundness is
                     still judged against the clean plurality over ALL clean
                     votes, abstentions included, so this measures what
                     discarding costs the guarantee.
      retrieval      groups drawn only from the top-r items of a lexical
                     retriever. The adversary is assumed to get its items
                     retrieved (worst case), so the retrieved set's share
                     exceeds the rho the certificate is told. Measures what
                     free uniform pool access buys.
    """
    _guard(agent, outdir)
    rows = []
    for key, t, G, order, abstention, sampling in conditions:
        n = cert = corr = viol = 0
        for j, p in enumerate(pools):
            rho = next(iter(p.adversarial))
            adv = set(p.adversarial[rho])
            target = "FALSE" if p.answer == "TRUE" else "TRUE"
            rng = np.random.default_rng(seed * 100003 + j)
            if sampling == "retrieved":
                authentic = [i for i in _lexical_rank(p.items, p.query) if i not in adv]
                support = sorted(adv) + authentic[:max(0, retrieve_r - len(adv))]
            else:
                support = list(range(p.n))
            groups = []
            for _ in range(G):
                g = [support[int(i)] for i in rng.choice(len(support), size=t, replace=False)]
                groups.append(sorted(g) if order == "sorted" else g)
            vs, clean = [], []
            for g in groups:
                if set(g) & adv:
                    vs.append(target)
                    continue
                items = [p.items[i] for i in g]
                assert not any(s.startswith("ADVERSARY_ITEM_NEVER_SHOWN::") for s in items)
                v = agent.verdict(items, p.query)
                vs.append(v)
                clean.append(v)
            counted = [v for v in vs if v != ABSTAIN] if abstention == "discarded" else vs
            n += 1
            if not counted:
                continue
            r = certify(counted, pool_size_at(p, rho), rho, t, eta)
            if r.certified:
                cert += 1
                corr += r.verdict == p.answer
                viol += not _plurality_sound(clean, r.winner)
        rows.append({"condition": key, "t": t, "G": G, "order": order,
                     "abstention": abstention, "sampling": sampling,
                     "n_pools": n, "certified": cert, "certified_correct": corr,
                     "violations": viol, "eta": eta})
    out = {"experiment": "E5b", "agent": agent.name, "seed": seed, "rows": rows,
           "retrieve_r": retrieve_r}
    _write(outdir, f"E5b_{agent.name}_seed{seed}.json", out)
    return out


def e7_misspec(agent: Agent, pools_by_level: Dict[str, Dict[int, Sequence[Pool]]],
               ts: Sequence[int], G: int, outdir: str, seed: int = 0,
               eta: float = 0.01) -> dict:
    """Misspecification sweep (E7): the certificate is told a share rho_t
    while the adversary actually holds rho_a, over the full grid of the E4b
    levels rather than E5's single gap.

    Same pools, groups and worst-case adversary as e4b_feed_vs_pool (groups
    are drawn at the TRUE share with the same seeds, so every clean vote is
    an E4b cache hit). The deployer sees the pool, so n is the true pool
    size; only the share it certifies against is wrong. rho_t < rho_a
    underestimates the adversary (soundness may fail); rho_t > rho_a
    overestimates it (sound, but fires less).
    """
    _guard(agent, outdir)
    rows = []
    for dataset, by_a in pools_by_level.items():
        level_rho = {a: next(iter(ps[0].adversarial)) for a, ps in by_a.items()}
        for a_true, pools in sorted(by_a.items()):
            rho_a = level_rho[a_true]
            for t in ts:
                stats = {a: [0, 0, 0] for a in level_rho}   # cert, correct, viol
                for j, p in enumerate(pools):
                    target = "FALSE" if p.answer == "TRUE" else "TRUE"
                    groups = draw_groups(p, t, G, rho_a, seed=seed * 100003 + j)
                    vs, clean = [], []
                    for g in groups:
                        if not is_clean(p, g, rho_a):
                            vs.append(target)
                            continue
                        items = [p.items[i] for i in g]
                        assert not any(s.startswith("ADVERSARY_ITEM_NEVER_SHOWN::")
                                       for s in items)
                        v = agent.verdict(items, p.query)
                        vs.append(v)
                        clean.append(v)
                    n = pool_size_at(p, rho_a)
                    for a_t, rho_t in level_rho.items():
                        r = certify(vs, n, rho_t, t, eta)
                        if r.certified:
                            s = stats[a_t]
                            s[0] += 1
                            s[1] += r.verdict == p.answer
                            s[2] += not _plurality_sound(clean, r.winner)
                for a_t, (c, k, v) in sorted(stats.items()):
                    rows.append({"dataset": dataset, "t": t, "a_true": a_true,
                                 "rho_true": rho_a, "a_assumed": a_t,
                                 "rho_assumed": level_rho[a_t],
                                 "n_pools": len(pools), "certified": c,
                                 "certified_correct": k, "violations": v})
    out = {"experiment": "E7", "agent": agent.name, "seed": seed, "G": G,
           "eta": eta, "rows": rows}
    _write(outdir, f"E7_{agent.name}_seed{seed}.json", out)
    return out


def e4_baselines(agent: Agent, pools: Sequence[Pool], arq_t: int, arq_G: int,
                 arq_rho: float, outdir: str, seed: int = 0,
                 eta: float = 0.01, judge: Optional[Agent] = None,
                 subclaims_by_query: Optional[Dict[str, Sequence[str]]] = None
                 ) -> dict:
    """E4: baselines.py's six methods, each at its own budget AND at a budget
    matched to ARQ's call count (the development notes task D2, the development notes Sec. 4 E4).

    ARQ's own call count per query is arq_G (one call per drawn group, each of
    arity arq_t) -- this is the "matched budget" every baseline is compared
    against. feed_partition's omega is swept over {1, 3} per the development notes;
    its own certificate firing rate on these montage-attacked pools is the
    one other method in baselines.py that returns a certificate, and is
    recorded per-method per the development notes's own instruction, not aggregated
    away.

    majority_vote's K and feed_partition's own-budget omega are chosen so
    their OWN call count is close to but not exceeding arq_G, then each
    method is ALSO run again at a budget explicitly matched to arq_G
    (feed_partition: omega=1, giving arq_G groups when len(items)==arq_G;
    majority_vote: K=arq_G). Where a method's own natural budget already
    equals the matched one (feed_partition at omega=1), both entries are
    still recorded (not silently merged) so the table's two-budget shape
    stays uniform across methods, per the development notes's "every method at its
    own budget and at a budget matched to ARQ" framing.

    `judge` is a second Agent for llm_judge (the development notes notes this may need
    its own GPU if run concurrently with `agent`); if None, llm_judge uses
    `agent` for both roles, which is a weaker but always-runnable fallback,
    flagged in the result rather than silently assumed equivalent.
    `subclaims_by_query` supplies query_decomposition's ground-truth
    decomposition per query text; queries absent from it get
    query_decomposition's own "no decomposition available" response
    (baselines.py's documented behaviour for the synthesis case), not a
    fabricated split.
    """
    from . import baselines as bl

    _guard(agent, outdir)
    if judge is None:
        judge = agent
        judge_is_agent = True
    else:
        judge_is_agent = False
    subclaims_by_query = subclaims_by_query or {}

    per_query = []
    for p in pools:
        items = list(p.items)
        gold = p.answer
        entry = {"query": p.query, "gold": gold, "n_items": len(items)}

        u = bl.undefended(agent, items, p.query)
        entry["undefended"] = {**u, "correct": u["verdict"] == gold}

        fp_rows = {}
        for omega in (1, 3):
            r = bl.feed_partition(agent, items, p.query, omega=omega,
                                  k_prime=1)
            fp_rows[f"omega={omega}"] = {**r, "correct": r["verdict"] == gold}
        entry["feed_partition_own_budget"] = fp_rows

        omega_matched = max(1, round(len(items) / arq_G))
        r = bl.feed_partition(agent, items, p.query, omega=omega_matched,
                              k_prime=1)
        entry["feed_partition_matched"] = {
            **r, "correct": r["verdict"] == gold, "omega_matched": omega_matched}

        mv_own = bl.majority_vote(agent, items, p.query, K=5)
        entry["majority_vote_own_budget"] = {
            **mv_own, "correct": mv_own["verdict"] == gold}
        mv_matched = bl.majority_vote(agent, items, p.query, K=arq_G)
        entry["majority_vote_matched"] = {
            **mv_matched, "correct": mv_matched["verdict"] == gold}

        lj = bl.llm_judge(agent, judge, items, p.query, K=5)
        entry["llm_judge"] = {**lj, "correct": lj["verdict"] == gold,
                              "judge_is_agent": judge_is_agent}

        idet = bl.influence_detector(agent, items, p.query)
        entry["influence_detector"] = {
            **idet, "correct": idet["verdict"] == gold}

        subclaims = subclaims_by_query.get(p.query, [])
        qd = bl.query_decomposition(agent, items, p.query, subclaims,
                                    k_prime=1)
        entry["query_decomposition"] = {
            **qd, "correct": qd["verdict"] == gold if qd["applies"] else None}

        groups = draw_groups(p, arq_t, arq_G, max(p.adversarial), seed=seed)
        kept = reject_to_rho(p, groups, arq_rho, max(p.adversarial))
        vs = [agent.verdict([p.items[i] for i in g], p.query) for g in kept]
        cert = certify(vs, pool_size_at(p, arq_rho), arq_rho, arq_t, eta)
        entry["arq"] = {"verdict": cert.verdict, "certified": cert.certified,
                        "calls": len(kept), "correct": cert.verdict == gold}

        per_query.append(entry)

    def _rate(key, cond=lambda e: True):
        rel = [e for e in per_query if cond(e)]
        if not rel:
            return None
        return sum(1 for e in rel if e[key]["correct"]) / len(rel)

    summary = {
        "undefended_accuracy": _rate("undefended"),
        "feed_partition_omega1_accuracy": sum(
            1 for e in per_query
            if e["feed_partition_own_budget"]["omega=1"]["correct"]
        ) / len(per_query),
        "feed_partition_omega3_accuracy": sum(
            1 for e in per_query
            if e["feed_partition_own_budget"]["omega=3"]["correct"]
        ) / len(per_query),
        "feed_partition_matched_accuracy": sum(
            1 for e in per_query if e["feed_partition_matched"]["correct"]
        ) / len(per_query),
        "feed_partition_omega1_cert_rate": sum(
            1 for e in per_query
            if e["feed_partition_own_budget"]["omega=1"]["certified"]
        ) / len(per_query),
        "majority_vote_own_accuracy": _rate("majority_vote_own_budget"),
        "majority_vote_matched_accuracy": _rate("majority_vote_matched"),
        "llm_judge_accuracy": _rate("llm_judge"),
        "influence_detector_accuracy": _rate("influence_detector"),
        "query_decomposition_accuracy": _rate(
            "query_decomposition", lambda e: e["query_decomposition"]["applies"]),
        "query_decomposition_coverage": sum(
            1 for e in per_query if e["query_decomposition"]["applies"]
        ) / len(per_query),
        "arq_accuracy": _rate("arq"),
        "arq_cert_rate": sum(1 for e in per_query if e["arq"]["certified"]
                             ) / len(per_query),
    }

    out = {"experiment": "E4", "agent": agent.name, "seed": seed,
           "arq_t": arq_t, "arq_G": arq_G, "arq_rho": arq_rho,
           "n_pools": len(pools), "summary": summary, "per_query": per_query}
    _write(outdir, f"E4_{agent.name}_seed{seed}.json", out)
    return out
