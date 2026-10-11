"""
arq.certificate -- the certificate, the bound, and the statistics behind them.

This module is the one that has to be right. Everything else in the package
either feeds it or reports what it returns.

WHAT IT IMPLEMENTS (with the paper section each corresponds to)
  clean_probability   exact hypergeometric P_c                        (Alg. 1, Sec 5.3)
  contamination_bound Dbar, the (1-eta) quantile of Bin(G, 1-P_c)     (Alg. 1)
  certify             the margin test c1 - c2 > Dbar                  (Prop. 5.1)
  feasible / rho_star the bound rho < 1 - (2/(2q+1))^(1/tau)          (Thm. 5.2)
  task_arity          tau = min{t : Wilson lower bound of q(t) > 1/2} (Sec 6, E1)
  wilson, paired_bootstrap, holm                                      (Sec 6, statistics)

THREE THINGS THAT ARE EASY TO GET WRONG, AND ARE HANDLED HERE
  1. P_c uses the EXACT hypergeometric product, not (1-rho)^t. The convenient
     form OVERSTATES how often groups are clean, which understates Dbar and
     would make the certificate unsound at finite pool size.
  2. The margin threshold is Dbar under an augmenting adversary and 2*Dbar under
     a displacing one. Using Dbar when the adversary displaces is UNSOUND; an
     exhaustive search found 88 counterexamples at G<=7 (see tests/).
  3. Dbar is an exact binomial quantile, not a confidence interval. P_c is
     computed from (rho, n, t); nothing is being estimated.

NO FUNCTION HERE FABRICATES A RESULT. Missing or malformed input raises.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import comb, log, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CertificateResult", "clean_probability", "contamination_bound", "certify",
    "feasible", "rho_star", "rho_star_displacement", "task_arity",
    "wilson", "paired_bootstrap", "holm", "ABSTAIN",
]

ABSTAIN = "ABSTAIN"          # counted as a verdict, per Sec. 5 of the paper


# ---------------------------------------------------------------------------
# clean-group probability and the contamination bound
# ---------------------------------------------------------------------------
def clean_probability(n: int, rho: float, t: int) -> float:
    """P_c: probability a uniform t-subset of an n-item pool contains no
    adversarial item, when the adversary controls ceil(rho*n) of them.

    Exact hypergeometric. See the module docstring for why (1-rho)**t is unsafe.
    """
    if not (0 <= rho <= 1):
        raise ValueError(f"rho must lie in [0,1], got {rho}")
    if t < 1 or n < 1:
        raise ValueError(f"need n>=1 and t>=1, got n={n}, t={t}")
    m = int(np.ceil(rho * n))
    if n - m < t:                       # not enough clean items to fill a group
        return 0.0
    return comb(n - m, t) / comb(n, t)


def contamination_bound(G: int, p_clean: float, eta: float = 0.01) -> int:
    """Dbar: smallest d with P[Bin(G, 1-p_clean) <= d] >= 1-eta.

    Exact, by accumulating the pmf. Not a confidence interval: 1-p_clean is
    known, not estimated.

    Computed in log-space (30 Sep 2026 fix): `comb(G, d) * q**d * (1-q)**(G-d)`
    directly overflows float64 for large G (e.g. G>=3000) because
    comb(G, d) grows combinatorially faster than the q**d/(1-q)**(G-d)
    factors shrink it -- confirmed by a real OverflowError while sizing
    E5's stress test to a larger G (see the development notes, 30 Sep 2026 entry).
    log(comb(G,d)) via lgamma avoids ever forming the raw product; the pmf
    terms are summed via a running max + logsumexp-style accumulation so
    the final probability is still recovered in linear space without
    overflowing, matching the original's exact (not approximated) pmf
    accumulation semantics.
    """
    if G < 1:
        raise ValueError("G must be at least 1")
    if not (0 < eta < 1):
        raise ValueError(f"eta must lie in (0,1), got {eta}")
    q = 1.0 - p_clean
    if q <= 0.0:
        return 0
    if q >= 1.0:
        return G
    from math import lgamma
    log_q, log_1mq = log(q), log(1 - q)
    acc = 0.0
    for d in range(G + 1):
        log_pmf = (lgamma(G + 1) - lgamma(d + 1) - lgamma(G - d + 1)
                   + d * log_q + (G - d) * log_1mq)
        acc += np.exp(log_pmf)
        if acc >= 1 - eta:
            return d
    return G


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------
@dataclass
class CertificateResult:
    verdict: Optional[str]      # None when uncertified
    certified: bool
    c1: int
    c2: int
    winner: str
    margin: int
    dbar: int
    threshold: int              # dbar, or 2*dbar under displacement
    p_clean: float
    counts: Dict[str, int]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def certify(verdicts: Sequence[str], n: int, rho: float, t: int,
            eta: float = 0.01, adversary: str = "augment") -> CertificateResult:
    """Run the margin test on a list of per-group verdicts.

    adversary: "augment"     -> threshold Dbar   (Assumption 3 holds)
               "displace"    -> threshold 2*Dbar (Assumption 3 fails; Sec. 5.3)

    Abstentions are counted as a verdict, which is the conservative choice: it
    avoids depending on the agent's abstention calibration. If ABSTAIN wins,
    the result is a certified statement of insufficient evidence, not an answer.
    """
    if adversary not in ("augment", "displace"):
        raise ValueError("adversary must be 'augment' or 'displace'")
    if len(verdicts) == 0:
        raise ValueError("no verdicts supplied; re-run the queries, do not impute")
    if any(v is None for v in verdicts):
        raise ValueError("a verdict is None: a query failed. Re-run it.")

    G = len(verdicts)
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, c1 = ranked[0]
    c2 = ranked[1][1] if len(ranked) > 1 else 0

    p_clean = clean_probability(n, rho, t)
    dbar = contamination_bound(G, p_clean, eta)
    threshold = dbar if adversary == "augment" else 2 * dbar
    ok = (c1 - c2) > threshold

    return CertificateResult(
        verdict=winner if ok else None, certified=ok, c1=c1, c2=c2,
        winner=winner, margin=c1 - c2, dbar=dbar, threshold=threshold,
        p_clean=p_clean, counts=counts,
    )


# ---------------------------------------------------------------------------
# the feasibility bound (Theorem 5.2)
# ---------------------------------------------------------------------------
def feasible(p_clean: float, q: float, adversary: str = "augment") -> bool:
    """Asymptotic firing condition. augment: P_c > 2/(2q+1).
    displace: P_c > 3/(2(q+1))."""
    if not (0 <= q <= 1):
        raise ValueError(f"q must lie in [0,1], got {q}")
    thr = 2.0 / (2 * q + 1) if adversary == "augment" else 3.0 / (2 * (q + 1))
    return p_clean > thr


def rho_star(tau: int, q: float) -> float:
    """Largest adversarial share at which a certificate is possible, large-pool
    limit, augmenting adversary. Returns 0.0 when q <= 1/2 (no solution)."""
    if tau < 1:
        raise ValueError("tau must be at least 1")
    if q <= 0.5:
        return 0.0
    return 1.0 - (2.0 / (2 * q + 1)) ** (1.0 / tau)


def rho_star_displacement(tau: int, q: float) -> float:
    """Same, under a displacing adversary. Returns 0.0 when q <= 1/2."""
    if tau < 1:
        raise ValueError("tau must be at least 1")
    if q <= 0.5:
        return 0.0
    return 1.0 - (3.0 / (2 * (q + 1))) ** (1.0 / tau)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval. Returns (point, lower, upper)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError("successes must lie in [0, n]")
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def task_arity(q_successes: Dict[int, int], q_trials: Dict[int, int],
               z: float = 1.96) -> Optional[int]:
    """tau = min{t : Wilson LOWER bound of q(t) > 1/2}.

    The lower bound rather than the point estimate, so sampling noise raises tau
    instead of manufacturing a false feasibility claim. Returns None if no t
    qualifies, which means the task is out of scope at the arities measured.
    """
    for t in sorted(q_successes):
        if t not in q_trials:
            raise KeyError(f"no trial count for t={t}")
        _, lo, _ = wilson(q_successes[t], q_trials[t], z)
        if lo > 0.5:
            return t
    return None


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap on the mean difference a-b. Returns point, CI, p."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs equal-length paired samples")
    if a.size == 0:
        raise ValueError("empty sample")
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("NaN in sample: a run failed. Re-run it, do not impute.")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    draws = d[idx].mean(axis=1)
    obs = float(d.mean())
    p = float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {"diff": obs, "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "p": min(1.0, p)}


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni. Applies only to the confirmatory family declared in the
    paper; everything else is exploratory and carries no significance claim."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(order)
    out, still = {}, True
    for i, (name, p) in enumerate(order):
        if still and p <= alpha / (k - i):
            out[name] = True
        else:
            still = False
            out[name] = False
    return out
