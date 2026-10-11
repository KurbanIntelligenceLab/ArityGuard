"""
tests/test_certificate.py -- the verification suite for arq.certificate.

Run:  python -m tests.test_certificate        (from the package root)

These tests check ARITHMETIC and SOUNDNESS against planted or exhaustively
enumerated cases. They involve no language model and support no empirical
claim about any agent. Every empirical result in the paper remains [RES].

The four checks that matter:
  T1  exhaustive: the certificate never contradicts the clean plurality
  T2  exhaustive: it DOES break under a displacing adversary at threshold Dbar
      (88 counterexamples expected at G<=7), and 2*Dbar repairs it
  T3  the closed-form bound matches a direct evaluation of the firing condition
  T4  Monte Carlo soundness against a worst-case adversary
"""
from __future__ import annotations

import itertools
import random
import sys
from math import comb

sys.path.insert(0, ".")  # run from code/
from arq.certificate import (                                    # noqa: E402
    ABSTAIN, certify, clean_probability, contamination_bound, feasible, holm,
    paired_bootstrap, rho_star, rho_star_displacement, task_arity, wilson,
)

FAILURES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES += 1


def _splits(total: int, k: int):
    if k == 1:
        yield (total,); return
    for x in range(total + 1):
        for rest in _splits(total - x, k - 1):
            yield (x,) + rest


# ---------------------------------------------------------------------------
def t1_exhaustive_soundness() -> None:
    print("\n== T1. exhaustive: certified winner strictly leads among clean groups ==")
    bad = checked = 0
    for G in range(2, 9):
        for nv in (2, 3):
            for D in range(G + 1):
                for Dbar in range(D, G + 1):
                    for cl in _splits(G - D, nv):
                        for ad in _splits(D, nv):
                            tot = [cl[i] + ad[i] for i in range(nv)]
                            order = sorted(range(nv), key=lambda i: -tot[i])
                            v1, c1 = order[0], tot[order[0]]
                            c2 = tot[order[1]]
                            checked += 1
                            if c1 - c2 > Dbar:      # augmenting threshold
                                if any(cl[u] >= cl[v1] for u in range(nv) if u != v1):
                                    bad += 1
    check("no counterexample under an augmenting adversary", bad == 0,
          f"{checked} configurations, {bad} violations")


def t2_displacement_breaks() -> None:
    print("\n== T2. exhaustive: Dbar is UNSOUND under displacement; 2*Dbar repairs ==")
    for thresh_mult, expect_break in ((1, True), (2, False)):
        bad = checked = 0
        for G in range(2, 8):
            for D in range(G + 1):
                for Dbar in range(D, G + 1):
                    for cl_full in _splits(G, 2):
                        for ev in _splits(min(D, G), 2):
                            if ev[0] > cl_full[0] or ev[1] > cl_full[1] or sum(ev) != D:
                                continue
                            for ad in _splits(D, 2):
                                tot = [cl_full[i] - ev[i] + ad[i] for i in range(2)]
                                o = sorted(range(2), key=lambda i: -tot[i]); v1 = o[0]
                                checked += 1
                                if tot[o[0]] - tot[o[1]] > thresh_mult * Dbar:
                                    if cl_full[1 - v1] >= cl_full[v1]:
                                        bad += 1
        label = f"threshold {thresh_mult}*Dbar under displacement"
        if expect_break:
            check(label + " breaks, as the paper predicts", bad == 88,
                  f"{bad} counterexamples (paper reports 88)")
        else:
            check(label + " is sound", bad == 0, f"{checked} configurations")


def t3_bound_matches_condition() -> None:
    print("\n== T3. closed-form bound vs a direct evaluation of the firing condition ==")
    mism = 0
    for q in (0.51, 0.6, 0.7, 0.8, 0.9, 1.0):
        for i in range(1, 2001):
            P = i / 2000
            cstar, cadv = q * P, (1 - P) + (1 - q) * P
            fires = (max(cstar, cadv) - min(cstar, cadv)) > (1 - P)
            if fires != feasible(P, q, "augment"):
                mism += 1
            if fires and cstar < cadv:
                mism += 1               # must never fire on a wrong winner
    check("augmenting bound P_c > 2/(2q+1) matches, no wrong-winner firing", mism == 0)

    mism = 0
    for q in (0.6, 0.8, 1.0):
        for i in range(1, 2001):
            P = i / 2000
            cstar, cadv = q * P, (1 - P) + (1 - q) * P
            fires = (max(cstar, cadv) - min(cstar, cadv)) > 2 * (1 - P)
            if fires != feasible(P, q, "displace"):
                mism += 1
    check("displacing bound P_c > 3/(2(q+1)) matches", mism == 0)

    vals = {(1, 1.0): 0.3333, (2, 1.0): 0.1835, (3, 1.0): 0.1264,
            (4, 1.0): 0.0964, (5, 1.0): 0.0779, (3, 0.9): 0.1061}
    ok = all(abs(rho_star(t, q) - v) < 5e-4 for (t, q), v in vals.items())
    check("rho_star reproduces the values printed in the paper", ok)
    check("rho_star_displacement(3, 1.0) == 0.0914",
          abs(rho_star_displacement(3, 1.0) - 0.0914) < 5e-4)
    check("q <= 1/2 admits no solution", rho_star(3, 0.5) == 0.0)


def t4_monte_carlo() -> None:
    print("\n== T4. Monte Carlo soundness against a worst-case adversary ==")
    rng = random.Random(20271008)
    n, t, G, eta = 400, 3, 200, 0.01
    fired = viol = 0
    for _ in range(3000):
        rho = rng.choice([0.02, 0.05, 0.10, 0.15])
        q = rng.choice([0.8, 0.9, 1.0])
        Pc = clean_probability(n, rho, t)
        verdicts = []
        for _ in range(G):
            if rng.random() < Pc:
                verdicts.append("CORRECT" if rng.random() < q else ABSTAIN)
            else:
                verdicts.append("ADVERSARY")        # worst case
        r = certify(verdicts, n, rho, t, eta)
        if r.certified:
            fired += 1
            clean = [v for v in verdicts if v != "ADVERSARY"]
            cc = {v: clean.count(v) for v in set(clean)}
            if not cc or r.winner != max(cc, key=cc.get):
                viol += 1
    check("no certified verdict contradicted the clean plurality", viol == 0,
          f"fired {fired}/3000, violations {viol}")


def t5_units_and_edges() -> None:
    print("\n== T5. boundary cases and input validation ==")
    check("rho=0 gives P_c=1", clean_probability(100, 0.0, 3) == 1.0)
    check("rho=1 gives P_c=0", clean_probability(100, 1.0, 3) == 0.0)
    check("t larger than the clean pool gives P_c=0",
          clean_probability(100, 0.98, 3) == 0.0)
    check("exact P_c is <= (1-rho)^t (the approximation is optimistic)",
          clean_probability(200, 0.2, 3) <= 0.8 ** 3)
    check("Dbar=0 when nothing is contaminated", contamination_bound(50, 1.0) == 0)
    check("Dbar=G when everything is", contamination_bound(50, 0.0) == 50)
    for name, fn in (("empty verdicts", lambda: certify([], 100, 0.1, 3)),
                     ("None verdict", lambda: certify(["A", None], 100, 0.1, 3)),
                     ("rho out of range", lambda: clean_probability(100, 1.5, 3)),
                     ("bad adversary name", lambda: certify(["A"], 100, 0.1, 3,
                                                            adversary="nope"))):
        try:
            fn(); check(f"raises on {name}", False, "no exception")
        except (ValueError, TypeError):
            check(f"raises on {name}", True)
    _, lo, hi = wilson(8, 10)
    check("Wilson interval brackets the point estimate", lo <= 0.8 <= hi,
          f"[{lo:.3f}, {hi:.3f}]")
    check("task_arity uses the LOWER bound (noise raises tau)",
          task_arity({1: 6, 2: 6, 3: 30}, {1: 10, 2: 10, 3: 40}) == 3,
          "q(2)=0.60 point estimate but lower bound <= 0.5, so tau=3")
    check("task_arity returns None when no t qualifies",
          task_arity({1: 2, 2: 3}, {1: 10, 2: 10}) is None)
    r = paired_bootstrap([1, 2, 3, 4], [0, 1, 2, 3], n_boot=500)
    check("paired bootstrap recovers a constant difference",
          abs(r["diff"] - 1.0) < 1e-9)
    h = holm({"a": 0.001, "b": 0.9})
    check("Holm rejects the small p and not the large", h["a"] and not h["b"])


if __name__ == "__main__":
    t1_exhaustive_soundness()
    t2_displacement_breaks()
    t3_bound_matches_condition()
    t4_monte_carlo()
    t5_units_and_edges()
    print("\n" + "-" * 66)
    print(f"VERIFICATION SUITE: {'ALL PASS' if FAILURES == 0 else f'{FAILURES} FAILURE(S)'}")
    print("Arithmetic and soundness only. No language model was called; the")
    print("empirical results come from results/ via `make tables`.")
    sys.exit(1 if FAILURES else 0)
