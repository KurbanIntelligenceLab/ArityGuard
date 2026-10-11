"""
arq.report -- turn results/*.json into the paper's LaTeX tables and figure data.

Every cell is copied from a results file. A missing result is emitted as the
red [RES] macro, never as a blank, a zero, or a value carried over from a
neighbouring configuration. That is the whole point of this module: it makes
fabrication require a deliberate edit rather than an oversight.

Usage
    python -m arq.report --results results/ --out paper/generated/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

from .certificate import ABSTAIN, rho_star, rho_star_displacement, task_arity, wilson

RES = r"\RES"


def _load(results_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    out = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(f)
        out.append(d)
    return out


def _cell(v, fmt="{:.3f}") -> str:
    """A value, or the placeholder. Never an invented number."""
    if v is None:
        return RES
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _finalize_table(rows: List[str], macro: str) -> str:
    """Wrap table rows as a \\newcommand, for a file meant to be \\input
    OUTSIDE a tabular (in the preamble), then invoked by name INSIDE it.

    30 Sep 2026 fix, confirmed by direct reproduction against the real
    LaTeX toolchain (see the development notes, 30 Sep 2026), not
    inferred: \\input-ing a file of raw table rows DIRECTLY INSIDE a
    tabular environment reliably breaks -- TeX's \\input appends its own
    end-of-line token after the last line it reads, which the tabular's
    alignment parsing does not expect mid-row. This reproduced with
    trivial single-row content, multi-row content, with or without a
    trailing '%' on the file's last line -- none of those "fix the
    trailing newline" variants actually worked when re-tested cleanly
    (each earlier "it worked" turned out to be a stale log from a prior
    compile, not a real pass -- caught by explicitly removing .log/.aux
    before every retest).

    The fix that DOES work under direct, clean-state testing: define the
    rows as a \\newcommand{{\\macro}}{{...}} macro in this generated
    file, \\input this file in the paper's PREAMBLE (not inside the
    table), and invoke \\macro as a single token inside the tabular. This
    sidesteps \\input's end-of-line-token issue entirely, since the
    macro's body is fully parsed and stored before the table ever sees
    it.
    """
    body = "\n".join(rows)
    return f"\\newcommand{{\\{macro}}}{{%\n{body}\n}}\n"


def _tau_cell(tau, t_measured) -> str:
    """tau, or '>t_max' when it was measured and never crossed 1/2. A
    measured-but-undefined tau is a result, not a missing one, so it must
    not print as [RES]."""
    if tau is not None:
        return str(int(tau))
    return f"$>${max(int(t) for t in t_measured)}"


def _short_agent(agent: str) -> str:
    return agent.split("/")[-1].replace("-Instruct", "")


def _common_seeds(results: List[dict], exp: str, agents: List[str], pred=None):
    """Files of experiment `exp` per agent, restricted to the seeds EVERY
    agent has, so a table never mixes seed counts across columns while some
    seeds are still running. (Before 8 Oct 2026 several tables took the first
    or last file per agent, which silently showed seed 0 in one table and
    seed 4 in another once seed files existed.)"""
    by = {a: [r for r in results if r.get("experiment") == exp and r.get("agent") == a
              and (pred is None or pred(r))] for a in agents}
    present = [{r.get("seed", 0) for r in by[a]} for a in agents]
    common = set.intersection(*present) if present and all(present) else set()
    return ({a: sorted((r for r in by[a] if r.get("seed", 0) in common),
                       key=lambda r: r.get("seed", 0)) for a in agents}, sorted(common))


def _seeds_macro(name: str, seeds) -> str:
    """'5 seeds' / '1 seed', so captions read right while seeds are landing."""
    n = len(seeds) if seeds else 0
    txt = f"{n} seed{'s' if n != 1 else ''}" if n else RES
    return f"\\newcommand{{\\{name}}}{{{txt}}}\n"


def _pooled_q(records: List[dict]):
    """Pool per-t successes and trials over seed files of one (agent,
    dataset) and recompute tau with the drivers' own rule (Wilson lower
    bound > 1/2). Returns (tau, q_lower by t, t values)."""
    succ, tri = {}, {}
    for d in records:
        for t, n in d["trials"].items():
            t = int(t)
            tri[t] = tri.get(t, 0) + int(n)
            succ[t] = succ.get(t, 0) + int(round(d["q_point"][str(t)] * int(n)))
    tau = task_arity(succ, tri)
    lower = {t: wilson(succ[t], tri[t])[1] for t in tri}
    return tau, lower, sorted(tri)


def table_bound(results: List[dict], agents: List[str]) -> str:
    """Table: seeds, tau, q(tau), rho* (derived), memory flags, curve MAE.
    E1 trials are pooled over EACH agent's own seeds (the arity probe is a
    per-agent measurement, so one agent's extra seeds need not wait for the
    other's), and the seed count is printed per row. "Memory" counts the
    hypotheses the no-evidence control flags as answerable without evidence
    (largest per-seed count), out of the hypotheses probed. The E3 MAE is averaged
    over its common seeds."""
    e3, s3 = _common_seeds(results, "E3", agents, lambda r: not r.get("forced_worst_case"))
    rows = []
    for a in agents:
        e1 = sorted((r for r in results if r.get("experiment") == "E1" and r.get("agent") == a),
                    key=lambda r: r.get("seed", 0))
        mae = (sum(r["curve_mae"] for r in e3[a]) / len(e3[a])) if e3[a] else None
        if not e1:
            rows.append(f"{_short_agent(a)} & {RES} & {RES} & {RES} & {RES} & {RES} & {_cell(mae)}\\\\")
            continue
        tau, lower, ts = _pooled_q(e1)
        if tau is None:
            tau_c, q_c, rs_c = f"$>${max(ts)}", "--", "--"
        else:
            q = lower[tau]
            tau_c, q_c, rs_c = str(tau), _cell(q), _cell(rho_star(tau, q))
        # Flags are pool fingerprints, which change with the seed (repost
        # order), so they cannot be unioned across seeds: report the largest
        # per-seed count.
        flagged = max(len(r.get("memory_contaminated_queries", [])) for r in e1)
        mem_c = f"{flagged}/{e1[0].get('n_pools', '?')}"
        rows.append(f"{_short_agent(a)} & {len(e1)} & {tau_c} & {q_c} & {rs_c} & {mem_c} & {_cell(mae)}\\\\")
    return _finalize_table(rows, "tableboundrows") + _seeds_macro("tableboundEthreeseeds", s3)


ARITY_DATASETS = [("hotpotqa", "\\textsc{HotpotQA}"),
                  ("2wiki_comparison", "2Wiki comparison"),
                  ("2wiki_bridge_comparison", "2Wiki bridge")]
DISTRACTOR_LEVELS = (0, 4, 16)


def table_arity(results: List[dict], agents: List[str],
                synth: Optional[List[dict]] = None) -> str:
    """E1b (code: E2) arity validation: tau at each distractor level d, per
    agent, pooled over the common seeds; then the same d=0 test with every
    supporting entity renamed (results/synthetic_entities/), the memory
    control. Bold = tau equals the annotated arity at d=0."""
    e2, seeds = _common_seeds(results, "E2", agents)
    sy, sseeds = _common_seeds(synth or [], "E2", agents)
    rows, srows = [], []
    for key, label in ARITY_DATASETS:
        gt = None
        cells, scells = [], []
        for a in agents:
            for d in DISTRACTOR_LEVELS:
                recs = [f["datasets"][f"{key}_d{d}"] for f in e2[a]
                        if f"{key}_d{d}" in f["datasets"]]
                if not recs:
                    cells.append(RES)
                else:
                    gt = recs[0]["ground_truth_arity"]
                    tau, _, ts = _pooled_q(recs)
                    c = str(tau) if tau is not None else f"$>${max(ts)}"
                    cells.append(f"\\textbf{{{c}}}" if (d == 0 and tau == gt) else c)
            recs = [f["datasets"][f"{key}_synth_d0"] for f in sy[a]
                    if f"{key}_synth_d0" in f["datasets"]]
            if not recs:
                scells += [RES, "", ""]
            else:
                gt = recs[0]["ground_truth_arity"]
                tau, _, ts = _pooled_q(recs)
                c = str(tau) if tau is not None else f"$>${max(ts)}"
                scells += [f"\\textbf{{{c}}}" if tau == gt else c, "", ""]
        rows.append(f"{label} & {_cell(gt, '{:d}')} & " + " & ".join(cells) + "\\\\")
        srows.append(f"{label} & {_cell(gt, '{:d}')} & " + " & ".join(scells) + "\\\\")
    return (_finalize_table(rows, "tablearityrows") + _finalize_table(srows, "tablearitysynthrows")
            + _seeds_macro("tablearityseeds", seeds) + _seeds_macro("tablearitysynthseeds", sseeds))


# (row label, key in an E4 per_query entry, optional sub-key)
METHOD_ROWS = [
    ("undefended agent", "undefended", None),
    ("RobustRAG mech., $\\omega{=}1$", "feed_partition_own_budget", "omega=1"),
    ("RobustRAG mech., $\\omega{=}3$", "feed_partition_own_budget", "omega=3"),
    ("majority vote", "majority_vote_own_budget", None),
    ("LLM judge \\cite{zheng2023judging}", "llm_judge", None),
    ("influence detector \\cite{coma2026}", "influence_detector", None),
    ("query decomposition", "query_decomposition", None),
    ("RobustRAG mech., matched", "feed_partition_matched", None),
    ("majority vote, matched", "majority_vote_matched", None),
    ("\\arq", "arq", None),
]
ATTACK_TARGET = "TRUE"  # every CoPHEME claim is the montage's target conclusion


def _pct(k: int, n: int) -> str:
    return f"{100.0 * k / n:.1f}" if n else RES


# Green cells ("\\win{...}", defined in both .tex preambles). Rule, applied
# by script and never by hand: an ARQ cell is green when ARQ is at least as
# good as EVERY other method in that column (ties count). Only columns where
# "better" is unambiguous are eligible: certified and certified-correct rates
# (higher), violations and attack success (lower). Accuracy, abstention,
# calls and cost are never highlighted.
def _win(cell: str) -> str:
    return f"\\win{{{cell}}}"


def _ours_best(ours, others, higher: bool) -> bool:
    others = [o for o in others if o is not None]
    if ours is None or not others:
        return False
    return ours >= max(others) if higher else ours <= min(others)


def table_methods(results: List[dict], agent: str, agents: Optional[List[str]] = None) -> str:
    """Table 3 rows for one agent, from results/E4_<agent>_seed*.json.

    ASR = share of queries whose returned verdict is TRUE: every CoPHEME
    claim is false or unverified and is the conclusion the montage steers
    toward, so TRUE is the attack succeeding. ASR|v conditions on a
    verdict being returned (uncertified or abstaining-by-design methods
    return None). Uncert./Cert. only for methods that issue a certificate.
    """
    macro = f"tablemethodsrows{_agent_macro_tag(agent)}"
    files, seeds = _common_seeds(results, "E4", agents or [agent])
    if not files.get(agent):
        rows = [f"{lab} & " + " & ".join([RES] * 7) + "\\\\" for lab, _, _ in METHOD_ROWS]
        return _finalize_table(rows, macro)
    pq = [q for f in files[agent] for q in f["per_query"]]
    n_queries = len(files[agent][0]["per_query"])
    stats = []
    for lab, key, sub in METHOD_ROWS:
        recs = [e[key][sub] if sub else e[key] for e in pq]
        if key == "query_decomposition" and not any(x.get("applies") for x in recs):
            stats.append((lab, key, None))
            continue
        n = len(recs)
        returned = [x for x in recs if x["verdict"] is not None]
        certifies = any(x.get("certified") is not None for x in recs)
        stats.append((lab, key, {
            "n": n, "calls": sum(x["calls"] for x in recs) / n,
            "correct": sum(1 for x, e in zip(recs, pq) if x["verdict"] == e["gold"]),
            "asr": sum(1 for x in recs if x["verdict"] == ATTACK_TARGET),
            "n_returned": len(returned),
            "abst": sum(1 for x in recs if x["verdict"] == ABSTAIN),
            "certifies": certifies,
            "cert": sum(1 for x in recs if x.get("certified"))}))

    def frac(st, num, den):
        return st[num] / st[den] if st and st[den] else None

    def asr_v(st):
        return st["asr"] / st["n_returned"] if st and st["n_returned"] else None

    # Reference: a policy that always answers ABSTAIN. It costs no call and
    # reaches the target conclusion never; it shows how much of ARQ's attack
    # resistance a pure abstainer also gets.
    n_all = len(pq)
    ref = {"n": n_all, "calls": 0.0,
           "correct": sum(1 for e in pq if e["gold"] == ABSTAIN),
           "asr": 0, "n_returned": n_all, "abst": n_all, "certifies": False, "cert": 0}
    stats.insert(len(stats) - 1, ("always ABSTAIN (reference)", "always_abstain", ref))
    others = [st for _, key, st in stats if key != "arq" and st]
    rows = []
    for lab, key, st in stats:
        if st is None:
            rows.append(f"{lab} & \\multicolumn{{7}}{{c}}{{\\emph{{no hop "
                        f"structure in \\textsc{{CoPHEME}}; does not apply}}}}\\\\")
            continue
        n = st["n"]
        asr_c = _pct(st["asr"], n)
        asrv_c = _pct(st["asr"], st["n_returned"])
        cert_c = _pct(st["cert"], n) if st["certifies"] else None
        if key == "arq":
            if _ours_best(frac(st, "asr", "n"), [frac(o, "asr", "n") for o in others], False):
                asr_c = _win(asr_c)
            if _ours_best(asr_v(st), [asr_v(o) for o in others], False):
                asrv_c = _win(asrv_c)
            if cert_c and _ours_best(frac(st, "cert", "n"),
                                     [frac(o, "cert", "n") for o in others if o["certifies"]], True):
                cert_c = _win(cert_c)
        cert_cells = (f"{_pct(n - st['cert'], n)} & {cert_c}" if st["certifies"]
                      else "-- & --")
        rows.append(f"{lab} & {st['calls']:.0f} & {_pct(st['correct'], n)} & {asr_c} & "
                    f"{asrv_c} & {_pct(st['abst'], n)} & {cert_cells}\\\\")
    rows.append(f"\\multicolumn{{8}}{{@{{}}r@{{}}}}{{\\scriptsize $N={len(pq)}$: "
                f"{n_queries} queries $\\times$ {len(seeds)} seed"
                f"{'s' if len(seeds) != 1 else ''}}}\\\\")
    return _finalize_table(rows, macro)


SPLIT_METHODS = [("undefended agent", "undefended", None),
                 ("majority vote", "majority_vote_own_budget", None),
                 ("LLM judge", "llm_judge", None),
                 ("influence detector", "influence_detector", None),
                 ("RobustRAG mech., $\\omega{=}1$", "feed_partition_own_budget", "omega=1"),
                 ("\\arq", "arq", None)]


def _holds_montage_post(n_items: int, rho: float) -> bool:
    """Whether a CoPHEME pool of n_items (authentic + coalition) built by
    pool.build_pool at share rho holds at least one coalition item:
    build_pool appends round(rho * n_auth / (1 - rho)) items."""
    for need in (0, 1, 2):
        n_auth = n_items - need
        if n_auth > 0 and int(round(rho * n_auth / (1 - rho))) == need:
            return need > 0
    raise ValueError(f"cannot recover the coalition count for n={n_items}")


def table_methods_split(results: List[dict], agents: List[str]) -> str:
    """Supplementary: the montage-attack methods split two ways, over the
    common seeds. (1) By gold label: correct answers on FALSE claims and on
    unverified (ABSTAIN) claims. (2) By whether the hypothesis's pool holds a
    montage post at the tested share: the TRUE rate (the attack's target)
    with and without one."""
    files, seeds = _common_seeds(results, "E4", agents)
    grows, mrows = [], []
    for lab, key, sub in SPLIT_METHODS:
        gcells, mcells = [], []
        for a in agents:
            pq = [(f["arq_rho"], q) for f in files[a] for q in f["per_query"]]
            if not pq:
                gcells += [RES] * 2; mcells += [RES] * 2
                continue
            def rec(q):
                return q[key][sub] if sub else q[key]
            for gold in ("FALSE", ABSTAIN):
                sel = [q for _, q in pq if q["gold"] == gold]
                gcells.append(_pct(sum(rec(q)["verdict"] == gold for q in sel), len(sel)))
            for present in (True, False):
                sel = [q for r, q in pq if _holds_montage_post(q["n_items"], r) == present]
                mcells.append(_pct(sum(rec(q)["verdict"] == ATTACK_TARGET for q in sel), len(sel)))
        grows.append(f"{lab} & " + " & ".join(gcells) + "\\\\")
        mrows.append(f"{lab} & " + " & ".join(mcells) + "\\\\")
    counts = []
    for a in agents[:1]:
        pq = [(f["arq_rho"], q) for f in files[a] for q in f["per_query"]]
        hyp = {q["query"]: _holds_montage_post(q["n_items"], r) for r, q in pq}
        counts = [sum(hyp.values()), len(hyp) - sum(hyp.values())]
    rows = [g[:-2] + " & " + m.split(" & ", 1)[1] for g, m in zip(grows, mrows)]
    out = _finalize_table(rows, "tablesplitrows")
    out += _seeds_macro("tablesplitseeds", seeds)
    if counts:
        out += (f"\\newcommand{{\\tablemontagewith}}{{{counts[0]}}}\n"
                f"\\newcommand{{\\tablemontagewithout}}{{{counts[1]}}}\n")
    return out


def _hypergeom_quantile(N: int, K: int, k: int, eta: float = 0.01) -> int:
    """Smallest d with P[#adversarial items in a k-item feed <= d] >= 1-eta,
    for a feed drawn without replacement from N items, K adversarial. This
    is the budget a delivered-feed certificate needs in order to be sound
    with probability 1-eta when only the share is known."""
    from math import comb
    total, acc = comb(N, k), 0
    for d in range(min(K, k) + 1):
        acc += comb(K, d) * comb(N - K, k - d)
        if acc / total >= 1 - eta:
            return d
    return min(K, k)


def _feedpool_rows(e4b: Dict[str, List[dict]], agents: List[str], t_min: int) -> List[str]:
    variants = [("\\arq{}, $G{=}200$", "arq", None, "run"),
                ("feed $k{=}12$, share budget", "feed", 12, "run"),
                ("feed $k{=}12$, tail budget", "feed", 12, "eta"),
                ("feed $k{=}24$, share budget", "feed", 24, "run"),
                ("feed $k{=}24$, tail budget", "feed", 24, "eta")]
    vals = {}   # (variant index, agent) -> (cert rate, corr rate, violations)
    for vi, (label, mech, k, budget) in enumerate(variants):
        for a in agents:
            files = e4b.get(a)
            if not files:
                vals[(vi, a)] = None
                continue
            n = cert = corr = viol = 0
            for r, row in ((f, rw) for f in files for rw in f["rows"]):
                if row["t"] < t_min or row["mechanism"] != mech or row["k"] != k:
                    continue
                N = int(round(row["n_adversarial"] / row["rho"]))
                thr = (_hypergeom_quantile(N, row["n_adversarial"], k, r["eta"])
                       if budget == "eta" else None)
                for rec in row["per_pool"]:
                    n += 1
                    c = rec["certified"] if thr is None else rec["margin"] > thr
                    if c:
                        cert += 1
                        corr += bool(rec["correct"])
                        viol += not rec["sound"]
            vals[(vi, a)] = (cert / n, corr / n, viol, cert, corr, n) if n else None
    rows = []
    for vi, (label, *_rest) in enumerate(variants):
        cells = []
        for a in agents:
            v = vals[(vi, a)]
            if v is None:
                cells += [RES] * 3
                continue
            cs = [_pct(v[3], v[5]), _pct(v[4], v[5]), str(v[2])]
            if vi == 0:     # ARQ: green where it is at least as good as every feed
                oth = [vals[(j, a)] for j in range(1, len(variants)) if vals[(j, a)]]
                for m, higher in ((0, True), (1, True), (2, False)):
                    if _ours_best(v[m], [o[m] for o in oth], higher):
                        cs[m] = _win(cs[m])
            cells += cs
        rows.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return rows


def table_feedpool(results: List[dict], agents: List[str], t_min: int = 2,
                   robustrag: Optional[List[dict]] = None) -> str:
    """E4b: ARQ vs a delivered-feed certificate, pooled over t >= t_min,
    both datasets and all shares, over the seeds every agent has. The feed is
    shown with two budgets: the share-derived ceil(rho*k) the run used, and
    the eta-sound hypergeometric quantile, evaluated offline from each pool's
    stored margin (a feed that certifies under the larger budget also
    certifies under the smaller, so its stored verdict and soundness flag
    apply). A second block repeats it on RobustRAG's own RealtimeQA and NQ
    data (results/robustrag/, arity 1, pooled over t in {1, 2})."""
    e4b, seeds = _common_seeds(results, "E4b", agents)
    rr, rseeds = _common_seeds(robustrag or [], "E4b", agents)
    return (_finalize_table(_feedpool_rows(e4b, agents, t_min), "tablefeedpoolrows")
            + _seeds_macro("tablefeedpoolseeds", seeds)
            + _finalize_table(_feedpool_rows(rr, agents, 1), "tablefeedpoolrobustragrows")
            + _seeds_macro("tablefeedpoolrobustragseeds", rseeds))


def table_third_model(third: List[dict], t_min: int = 2) -> str:
    """Supplementary: the feed-against-pool comparison (E4b) repeated with an
    agent from a third model family (results/side/third_model/), same rows,
    pooling and green rule as Table tab:feedpool."""
    agents = sorted({r["agent"] for r in third if r.get("experiment") == "E4b"})
    e4b, seeds = _common_seeds(third, "E4b", agents) if agents else ({}, [])
    return (_finalize_table(_feedpool_rows(e4b, agents or ["none"], t_min), "tablethirdrows")
            + _seeds_macro("tablethirdseeds", seeds))


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of b vs c discordant
    pairs at 1/2, twice the smaller tail (capped at 1). Integer arithmetic,
    so tiny p-values stay exact."""
    from fractions import Fraction
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def _holm(ps: List[float]) -> List[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def _paired_vs_tail(files: List[dict], t_min: int, k: int):
    """Discordant counts, ARQ vs the tail-budget feed of k items, paired on
    (seed, dataset, coalition size, t, question). Returns
    {outcome: (arq_only, feed_only)} for 'cert' and 'corr'."""
    arq, feed = {}, {}
    for f in files:
        for row in f["rows"]:
            if row["t"] < t_min:
                continue
            is_feed = row["mechanism"] == "feed" and row["k"] == k
            if row["mechanism"] != "arq" and not is_feed:
                continue
            if is_feed:
                N = int(round(row["n_adversarial"] / row["rho"]))
                thr = _hypergeom_quantile(N, row["n_adversarial"], k, f["eta"])
            for j, rec in enumerate(row["per_pool"]):
                key = (f.get("seed", 0), row["dataset"], row["n_adversarial"], row["t"], j)
                c = rec["certified"] if not is_feed else rec["margin"] > thr
                (feed if is_feed else arq)[key] = (c, c and bool(rec["correct"]))
    out = {}
    for oi, name in ((0, "cert"), (1, "corr")):
        b = sum(1 for key, v in arq.items() if key in feed and v[oi] and not feed[key][oi])
        c = sum(1 for key, v in arq.items() if key in feed and feed[key][oi] and not v[oi])
        out[name] = (b, c)
    return out


def table_mcnemar(results: List[dict], agents: List[str], robustrag: List[dict],
                  third: List[dict]) -> str:
    """Supplementary: exact McNemar tests of ARQ against the SOUND
    (tail-budget) feed, on every feed-against-pool data set, over the same
    seeds as the corresponding tables. Holm correction over the whole family.
    Cells: ARQ-only / feed-only discordant pairs, and Holm-adjusted p."""
    third_agents = sorted({r["agent"] for r in third if r.get("experiment") == "E4b"})
    blocks = [("feed against pool, $t\\in\\{2,3\\}$", _common_seeds(results, "E4b", agents)[0], agents, 2),
              ("RobustRAG data, $t\\in\\{1,2\\}$", _common_seeds(robustrag, "E4b", agents)[0], agents, 1),
              ("third model, feed against pool", _common_seeds(third, "E4b", third_agents)[0]
               if third_agents else {}, third_agents, 2)]
    tests = []   # (block, agent, k, outcome, b, c, p)
    for bi, (_, files, ags, t_min) in enumerate(blocks):
        for a in ags:
            if not files.get(a):
                continue
            for k in (12, 24):
                for name, (b, c) in _paired_vs_tail(files[a], t_min, k).items():
                    tests.append((bi, a, k, name, b, c, _mcnemar_exact(b, c)))
    adj = _holm([t[-1] for t in tests])

    def fmt(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3g}"
    rows = []
    for bi, (label, _, ags, _) in enumerate(blocks):
        rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\")
        for a in ags:
            for k in (12, 24):
                cells = []
                for name in ("cert", "corr"):
                    i = next((i for i, t in enumerate(tests)
                              if t[:4] == (bi, a, k, name)), None)
                    if i is None:
                        cells += [RES, RES]
                        continue
                    _, _, _, _, b, c, _ = tests[i]
                    p = fmt(adj[i]) + ("" if adj[i] < 0.05 else "$^{\\dagger}$")
                    cells += [f"{b}\\,/\\,{c}", p]
                rows.append(f"{_short_agent(a)} & $k{{=}}{k}$ & " + " & ".join(cells) + "\\\\")
    n_sig = sum(p < 0.05 for p in adj)
    return (_finalize_table(rows, "tablemcnemarrows")
            + f"\\newcommand{{\\tablemcnemarn}}{{{len(tests)}}}\n"
            + f"\\newcommand{{\\tablemcnemarsig}}{{{n_sig}}}\n"
            + f"\\newcommand{{\\tablemcnemarnonsig}}{{{len(tests) - n_sig}}}\n")


def table_third_arity(third: List[dict], synth: List[dict], t_max: int = 4) -> str:
    """Supplementary: the d=0 arity validation (E2) with the third model
    family, with real entity names (results/side/third_model/) and with
    invented names (.../synthetic_entities/). Per dataset: pooled clean-group
    accuracy q(t) for t = 1..t_max ('--' where the pool is smaller than t),
    then tau by the drivers' Wilson rule ('>t' when it never crosses 1/2).
    Bold tau = the annotated arity."""
    agents = sorted({r["agent"] for r in third + synth if r.get("experiment") == "E2"})
    real, rseeds = _common_seeds(third, "E2", agents) if agents else ({}, [])
    syn, sseeds = _common_seeds(synth, "E2", agents) if agents else ({}, [])

    def cells(files, key):
        recs = [f["datasets"][key] for a in agents for f in files.get(a, [])
                if key in f["datasets"]]
        if not recs:
            return [RES] * (t_max + 1), None
        gt = recs[0]["ground_truth_arity"]
        tau, _, ts = _pooled_q(recs)
        tri, suc = {}, {}
        for d in recs:
            for t, n in d["trials"].items():
                tri[int(t)] = tri.get(int(t), 0) + int(n)
                suc[int(t)] = suc.get(int(t), 0) + d["q_point"][t] * int(n)
        qs = [f"{suc[t] / tri[t]:.2f}" if t in tri else "--" for t in range(1, t_max + 1)]
        c = str(tau) if tau is not None else f"$>${max(ts)}"
        return qs + [f"\\textbf{{{c}}}" if tau == gt else c], gt

    rows = []
    for key, label in ARITY_DATASETS:
        rc, gt = cells(real, f"{key}_d0")
        sc, gt2 = cells(syn, f"{key}_synth_d0")
        rows.append(f"{label} & {_cell(gt or gt2, '{:d}')} & " + " & ".join(rc + sc) + "\\\\")
    return (_finalize_table(rows, "tablethirdarityrows")
            + _seeds_macro("tablethirdarityseeds", rseeds)
            + _seeds_macro("tablethirdaritysynthseeds", sseeds))


COAL_AGENTS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
               "mistralai/Mistral-7B-Instruct-v0.3"]
COAL_MECHS = ["arq", "feed12_share", "feed12_tail", "feed24_share", "feed24_tail"]


def table_coalition(coal: List[dict]) -> str:
    """Supplementary: the real-text coalition (results/side/coalition/). The
    montage posts are reposted to share rho of each relevance-filtered
    CoPHEME pool and the agent reads every group. Pooled over t in {1, 2} and
    the seeds every agent has. Per mechanism: % certified of the applicable
    queries (a feed longer than the pool does not apply), with violations in
    parentheses where nonzero. Then, for ARQ: % of queries certified TRUE
    (the montage's target) and the TRUE share among clean and among
    contaminated groups. Green: ARQ's certified rate is at least every
    feed's in that row."""
    agents = [a for a in COAL_AGENTS if any(r.get("agent") == a for r in coal)]
    files, seeds = (_common_seeds(coal, "side_coalition", agents) if agents
                    else ({}, []))
    rows, viol_share, viol_arq = [], 0, 0
    for a in agents:
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{_short_agent(a)}}}}}\\\\")
        for rho in sorted({r["rho"] for f in files[a] for r in f["rows"]}):
            agg = {m: [0, 0, 0, 0] for m in COAL_MECHS}   # n, cert, viol, target
            clean, cont = {}, {}
            for f in files[a]:
                for r in f["rows"]:
                    if r["rho"] != rho or r["mechanism"] not in agg:
                        continue
                    g = agg[r["mechanism"]]
                    g[0] += r["n_applicable"]
                    g[1] += r["certified"]
                    g[2] += r["violations"]
                    g[3] += r["certified_target"]
                    if r["mechanism"] == "arq":
                        for p in r["per_pool"]:
                            for v, c in p["clean_counts"].items():
                                clean[v] = clean.get(v, 0) + c
                            for v, c in p["contaminated_counts"].items():
                                cont[v] = cont.get(v, 0) + c
            rate = {m: (g[1] / g[0] if g[0] else None) for m, g in agg.items()}
            cells = []
            for m in COAL_MECHS:
                n, c, v, _ = agg[m]
                cell = _pct(c, n) + (f"\\,({v})" if v else "")
                if m == "arq" and _ours_best(rate[m], [rate[o] for o in COAL_MECHS[1:]], True):
                    cell = _win(cell)
                cells.append(cell)
                if m.endswith("_share"):
                    viol_share += v
                if m == "arq":
                    viol_arq += v
            n, _, _, tgt = agg["arq"]
            cells += [_pct(tgt, n), _pct(clean.get("TRUE", 0), sum(clean.values())),
                      _pct(cont.get("TRUE", 0), sum(cont.values()))]
            rows.append(f"${rho:.3f}$ & " + " & ".join(cells) + "\\\\")
    return (_finalize_table(rows or [RES + "\\\\"], "tablecoalitionrows")
            + _seeds_macro("tablecoalitionseeds", seeds)
            + f"\\newcommand{{\\tablecoalitionshareviol}}{{{viol_share}}}\n"
            + f"\\newcommand{{\\tablecoalitionarqviol}}{{{viol_arq}}}\n")


def table_misspec(results: List[dict], agents: List[str], t_min: int = 2) -> str:
    """E7 (supplementary): the certificate assumes share rho_t while the
    worst-case adversary holds rho_a, on the E4b pools, pooled over t >= t_min,
    both datasets and the common seeds. Cells: certified % and violations."""
    e7, seeds = _common_seeds(results, "E7", agents)
    levels = sorted({r["a_true"] for a in agents for f in e7[a] for r in f["rows"]})
    rho = {}
    for a in agents:
        for f in e7[a]:
            for r in f["rows"]:
                rho.setdefault(r["a_true"], r["rho_true"])
    rows = []
    for at in levels:
        cells = []
        for a in agents:
            for aa in levels:
                rs = [r for f in e7[a] for r in f["rows"] if r["t"] >= t_min
                      and r["a_true"] == at and r["a_assumed"] == aa]
                n = sum(r["n_pools"] for r in rs)
                c = sum(r["certified"] for r in rs)
                v = sum(r["violations"] for r in rs)
                cell = f"{_pct(c, n)}" + (f"\\,/\\,\\textbf{{{v}}}" if v else "\\,/\\,0")
                cells.append(f"\\underline{{{cell}}}" if aa == at else cell)
        rows.append(f"${rho.get(at, 0):.3f}$ & " + " & ".join(cells) + "\\\\")
    head = " & ".join(f"${rho[l]:.3f}$" for l in levels)
    return (_finalize_table(rows, "tablemisspecrows")
            + f"\\newcommand{{\\tablemisspechead}}{{{head}}}\n"
            + _seeds_macro("tablemisspecseeds", seeds))


ABLATION_LABELS = [("t1", "$t=1$"), ("t_tau", "$t=\\tau$ (default)"),
                   ("t_tau_plus1", "$t=\\tau{+}1$"), ("G50", "$G=50$"),
                   ("G200", "$G=200$ (default)"), ("G500", "$G=500$"),
                   ("fixed_order", "fixed presentation order"),
                   ("abstain_discarded", "abstentions discarded"),
                   ("retrieval", "retrieval-constrained sampling")]


def _cert_viol(cert, n, viol) -> str:
    return f"{cert}/{n} & {viol}/{cert}"


def _summed_rows(files: List[dict], cond: str):
    rows = [x for f in files for x in f["rows"] if x["condition"] == cond
            and x.get("n_pools") is not None]
    if not rows:
        return None
    return (sum(x["certified"] for x in rows), sum(x["n_pools"] for x in rows),
            sum(x["violations"] for x in rows))


def table_ablation(results: List[dict], agents: List[str]) -> str:
    """Table 4's arity, budget and design rows from E5b, both agents per row,
    summed over the common seeds."""
    e5b, seeds = _common_seeds(results, "E5b", agents)
    rows = []
    for key, label in ABLATION_LABELS:
        cells = []
        for a in agents:
            t = _summed_rows(e5b[a], key)
            cells.append(_cert_viol(t[0], t[1], t[2]) if t else f"{RES} & {RES}")
        rows.append(f"\\quad {label} & " + " & ".join(cells) + "\\\\")
    return (_finalize_table(rows[:6], "tableablationarityrows") +
            _finalize_table(rows[6:], "tableablationdesignrows") +
            _seeds_macro("tableablationseeds", seeds))


STRESS_LATEX = {
    "augmenting, rho_a = rho_t": "adding, $\\rho_a=\\rho_t$",
    "augmenting, rho_a > rho_t": "adding, $\\rho_a<\\rho_t$",
    "displacing, threshold Dbar": "evicting, $\\bar D$ $\\ddagger$",
    "displacing, threshold 2Dbar": "evicting, $2\\bar D$",
}


def table_stress_all(results: List[dict], agents: List[str]) -> str:
    """E5 assumption-stress rows, both agents per row, summed over the
    common seeds."""
    e5, seeds = _common_seeds(results, "E5", agents)
    rows = []
    for cond, label in STRESS_LATEX.items():
        cells = []
        for a in agents:
            t = _summed_rows(e5[a], cond)
            cells.append(_cert_viol(t[0], t[1], t[2]) if t else f"{RES} & {RES}")
        rows.append(f"\\quad {label} & " + " & ".join(cells) + "\\\\")
    return _finalize_table(rows, "tablestressallrows") + _seeds_macro("tablestressseeds", seeds)


COST_LABELS = [("undefended", "undefended agent"),
               ("feed_partition_omega1", "RobustRAG mech., $\\omega{=}1$"),
               ("feed_partition_omega3", "RobustRAG mech., $\\omega{=}3$"),
               ("majority_vote", "majority vote"), ("llm_judge", "LLM judge"),
               ("influence_detector", "influence detector"), ("arq", "\\arq")]


def table_cost(results: List[dict], agents: List[str]) -> str:
    """Measured cost per query (run_cost.py): calls, prompt tokens, GPU-s."""
    cost = {r.get("agent"): r for r in results if r.get("experiment") == "COST"}
    rows = []
    for key, label in COST_LABELS:
        cells = []
        for a in agents:
            m = cost.get(a, {}).get("methods", {}).get(key)
            cells.append(f"{m['calls_per_query']:.0f} & {m['prompt_tokens_per_query']/1000:.1f}k"
                         f" & {m['gpu_seconds_per_query']:.1f}" if m else f"{RES} & {RES} & {RES}")
        rows.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return _finalize_table(rows, "tablecostrows")


def _load_e6(results_dir: str) -> List[dict]:
    """E6 files: results/E6_seed*.json and results/e6_examples_*/E6_seed*.json."""
    paths = sorted(glob.glob(os.path.join(results_dir, "E6_seed*.json")) +
                   glob.glob(os.path.join(results_dir, "e6_examples_*", "E6_seed*.json")))
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def _tex_escape(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
                 ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


EXAMPLE_LABELS = [("fires", "fires, correct"), ("declines", "declines"),
                  ("certified_insufficient_evidence", "certifies ABSTAIN"),
                  ("certified_but_wrong", "certified, wrong")]


def table_examples(e6_files: List[dict]) -> str:
    """Verbatim E6 examples, one row per (agent, category) found. A category
    no file found is simply absent, never filled in. 'Wrong' is computed
    against gold, so a certified ABSTAIN on a FALSE claim is shown as
    certified-but-wrong, matching the paper's own definition."""
    rows = []
    for f in e6_files:
        ex = f.get("examples") or {}
        agent = _short_agent(f.get("example_agent") or "unknown agent")
        agent = agent.replace("Qwen2.5-", "")
        for key, lab in EXAMPLE_LABELS:
            r = ex.get(key)
            if not r:
                continue
            verdict = r["verdict"] if r["verdict"] is not None else "none"
            if r["certified"] and r["verdict"] != r["gold"]:
                lab = lab + "; \\emph{wrong}"
            rows.append(f"{agent} & {lab} & {_tex_escape(r['query'])} & {r['gold']} & "
                        f"{verdict} & {r['margin']}/{r['threshold']}\\\\")
    if not rows:
        rows = [f"{RES} & {RES} & {RES} & {RES} & {RES} & {RES}\\\\"]
    return _finalize_table(rows, "tableexamplesrows")


_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def _agent_macro_tag(agent: str) -> str:
    """A LaTeX command name may only contain letters -- map every digit to
    a word and drop everything else, so each agent's table_stress macro is
    unique and expandable.

    2 Oct 2026 fix: the original just stripped non-letters, which silently
    collapsed "Qwen2.5-7B-Instruct" and "Qwen2.5-14B-Instruct" to the SAME
    macro name ("QwenQwenBInstruct") -- confirmed by a real collision when
    both 7B and 14B's table_stress files were generated together (job
    406852 produced results for both models for the first time). Spelling
    out digits keeps distinct model ids distinct instead of merging them.
    """
    out = []
    for c in agent:
        if c.isalpha():
            out.append(c)
        elif c in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[c])
    return "".join(out) or "agent"


def table_stress(results: List[dict], agent: str) -> str:
    """Table 4: the assumption stress rows.

    Columns are Cert.$/N$, Viol.$/N$(among certified), eta -- matching
    what e5_stress actually measures (certification + soundness). Per-
    condition accuracy against gold answers is NOT tracked by e5_stress
    (it only compares the certified winner against the clean-group
    plurality, a soundness check, not an accuracy one) -- simplified out
    of the paper's table 30 Sep 2026 rather than left as an invented
    or silently-blank column.
    """
    # e5_stress's own condition strings (code/arq/experiments.py) use
    # plain-text underscores ("rho_a", "rho_t") -- fine for JSON/terminal
    # output, but a bare '_' outside LaTeX math mode is a fatal "Missing
    # $ inserted" error (confirmed 30 Sep 2026 by a real build failure,
    # not assumed). Map each of e5_stress's 4 fixed condition labels to
    # a LaTeX-safe rendering rather than escaping underscores blindly,
    # since these specific labels read better with proper math-mode
    # rho_a/rho_t than with an escaped "rho\_a".
    CONDITION_LATEX = {
        "augmenting, rho_a = rho_t": "adding, $\\rho_a=\\rho_t$",
        "augmenting, rho_a > rho_t": "adding, $\\rho_a<\\rho_t$",
        "displacing, threshold Dbar": "evicting, $\\bar D$",
        "displacing, threshold 2Dbar": "evicting, $2\\bar D$",
    }
    macro = f"tablestressrows{_agent_macro_tag(agent)}"
    e5 = next((r for r in results if r.get("experiment") == "E5"
               and r.get("agent") == agent), None)
    if e5 is None:
        rows = [f"{c} & {RES} & {RES} & {RES}\\\\" for c in
                ["adding, $\\rho_a=\\rho_t$",
                 "adding, $\\rho_a<\\rho_t$",
                 "evicting, $\\bar D$", "evicting, $2\\bar D$"]]
        return _finalize_table(rows, macro)
    rows = []
    for r in e5["rows"]:
        cond = CONDITION_LATEX.get(r["condition"], r["condition"])
        flag = " $\\ddagger$" if "FALSIFICATION" in r else ""
        n = r.get("n_pools")
        rows.append(f"{cond}{flag} & {_cell(r['certified'],'{:d}')}"
                    f"/{_cell(n,'{:d}')} & {_cell(r['violations'],'{:d}')}"
                    f"/{_cell(r['certified'],'{:d}')} & {_cell(r['eta'])}\\\\")
    return _finalize_table(rows, macro)


def figure_boundary_data(q_values=(1.0, 0.8), taus=(1, 2, 3, 4, 5)) -> str:
    """Figure 2 coordinates. Closed form, no experimental data, so this is the
    one generated artefact that is complete before any run."""
    lines = ["% Figure 2 coordinates, computed from the closed form"]
    for q in q_values:
        pts = " ".join(f"({t},{rho_star(t,q):.3f})" for t in taus)
        lines.append(f"% augment q={q}: {pts}")
    for q in q_values:
        pts = " ".join(f"({t},{rho_star_displacement(t,q):.3f})" for t in taus)
        lines.append(f"% displace q={q}: {pts}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/")
    ap.add_argument("--out", default="paper/generated/")
    ap.add_argument("--agents", nargs="*", default=["open-weight A", "open-weight B",
                                                    "API model C", "API model D"])
    ap.add_argument("--e1-results", default=None,
                    help="Directory whose E1 files replace the main results "
                         "dir's E1 files for Table 2, so tau/q come from the "
                         "same pool set as the E3 curve (e.g. "
                         "results/relevance_filtered).")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = _load(a.results)
    n_res = sum(1 for r in res if r.get("experiment"))
    with open(os.path.join(a.out, "table_bound.tex"), "w") as fh:
        bound_res = res
        if a.e1_results:
            bound_res = ([r for r in res if r.get("experiment") != "E1"] +
                         [r for r in _load(a.e1_results) if r.get("experiment") == "E1"])
        fh.write(table_bound(bound_res, a.agents))
    # One table_stress_<tag>.tex per agent (30 Sep 2026 fix): the original
    # wrote a single table_stress.tex from agents[0] only, silently
    # dropping every other agent's E5 rows. Table 4's own caption commits
    # to "one row per condition" (4 fixed condition rows, not agent x
    # condition), so this keeps that structure -- one table per agent,
    # not a wider table -- and writes a file per agent instead of just
    # the first.
    for ag in a.agents:
        tag = ag.replace("/", "_")
        with open(os.path.join(a.out, f"table_stress_{tag}.tex"), "w") as fh:
            fh.write(table_stress(res, ag))
    # Back-compat: table_stress.tex still written from agents[0], so any
    # existing \input reference to the old filename keeps working.
    with open(os.path.join(a.out, "table_stress.tex"), "w") as fh:
        fh.write(table_stress(res, a.agents[0]))
    with open(os.path.join(a.out, "figure_boundary.tex"), "w") as fh:
        fh.write(figure_boundary_data())
    synth_dir = os.path.join(a.results, "synthetic_entities")
    rr_dir = os.path.join(a.results, "robustrag")
    synth = _load(synth_dir) if os.path.isdir(synth_dir) else []
    rr = _load(rr_dir) if os.path.isdir(rr_dir) else []
    with open(os.path.join(a.out, "table_arity.tex"), "w") as fh:
        fh.write(table_arity(res, a.agents, synth))
    for ag in a.agents:
        tag = ag.replace("/", "_")
        with open(os.path.join(a.out, f"table_methods_{tag}.tex"), "w") as fh:
            fh.write(table_methods(res, ag, a.agents))
    for fname, fn in (("table_ablation.tex", table_ablation),
                      ("table_stress_all.tex", table_stress_all),
                      ("table_cost.tex", table_cost)):
        with open(os.path.join(a.out, fname), "w") as fh:
            fh.write(fn(res, a.agents))
    with open(os.path.join(a.out, "table_feedpool.tex"), "w") as fh:
        fh.write(table_feedpool(res, a.agents, robustrag=rr))
    with open(os.path.join(a.out, "table_misspec.tex"), "w") as fh:
        fh.write(table_misspec(res, a.agents))
    with open(os.path.join(a.out, "table_methods_split.tex"), "w") as fh:
        fh.write(table_methods_split(res, a.agents))
    with open(os.path.join(a.out, "table_examples.tex"), "w") as fh:
        fh.write(table_examples(_load_e6(a.results)))
    # A third model family, and the real-text coalition (results/side/).
    third_dir = os.path.join(a.results, "side", "third_model")
    coal_dir = os.path.join(a.results, "side", "coalition")
    with open(os.path.join(a.out, "table_third_model.tex"), "w") as fh:
        fh.write(table_third_model(_load(third_dir) if os.path.isdir(third_dir) else []))
    third_synth_dir = os.path.join(third_dir, "synthetic_entities")
    with open(os.path.join(a.out, "table_third_arity.tex"), "w") as fh:
        fh.write(table_third_arity(
            _load(third_dir) if os.path.isdir(third_dir) else [],
            _load(third_synth_dir) if os.path.isdir(third_synth_dir) else []))
    with open(os.path.join(a.out, "table_mcnemar.tex"), "w") as fh:
        fh.write(table_mcnemar(res, a.agents, rr,
                               _load(third_dir) if os.path.isdir(third_dir) else []))
    with open(os.path.join(a.out, "table_coalition.tex"), "w") as fh:
        fh.write(table_coalition(_load(coal_dir) if os.path.isdir(coal_dir) else []))
    print(f"read {n_res} result file(s) from {a.results}")
    print(f"wrote {4 + 2 * len(a.agents)} files to {a.out}")
    if n_res == 0:
        print("NOTE: no results found, so every cell is the [RES] placeholder. "
              "This is the correct output before the experiments are run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
