"""
arq.pool -- preprocessing: build evidence pools with a KNOWN adversarial share.

Ground-truth rho cannot be obtained from real data, so we construct it. The
authentic pool is fixed; adversary-selected fragments are added until the
intended share is reached. Adversarial sets are NESTED,
    A_{rho_1} subset A_{rho_2} subset ...
which is what makes the rho sweep affordable: draw groups once at rho_max,
then evaluate any smaller rho by REJECTING groups that contain items outside
A_rho. Rejection sampling returns exactly the group distribution of the
rho-pool, so the reuse is sound and not an approximation (paper, Sec. 6).

Datasets are read, never written. If a file is missing this module raises; it
never falls back to synthetic data silently. Synthetic pools are available only
through build_synthetic_pool(), which stamps every item so that synthetic runs
cannot be mistaken for real ones.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SYNTHETIC_PREFIX = "SYNTHETIC::"


@dataclass
class Pool:
    items: List[str]                       # authentic items, index = id
    adversarial: Dict[float, List[int]]    # rho -> nested adversarial index sets
    query: str
    answer: Optional[str] = None           # gold verdict, None outside calibration
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        rhos = sorted(self.adversarial)
        for a, b in zip(rhos, rhos[1:]):
            if not set(self.adversarial[a]).issubset(self.adversarial[b]):
                raise ValueError(
                    f"adversarial sets are not nested at rho={a} -> {b}; the "
                    "rejection-sampling reuse in Sec. 6 depends on nesting")

    @property
    def n(self) -> int:
        return len(self.items)

    def is_synthetic(self) -> bool:
        return any(s.startswith(SYNTHETIC_PREFIX) for s in self.items)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for s in self.items:
            h.update(s.encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[dict]:
    """Read a dataset file. Raises if it is absent: a missing dataset is an
    error, not a reason to substitute anything."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it as the development notes describes. This "
            "function will not substitute synthetic or cached data.")
    out = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} is not valid JSON") from e
    if not out:
        raise ValueError(f"{path} is empty")
    return out


def build_pool(authentic: Sequence[str], promoted: Sequence[str],
               rhos: Sequence[float], query: str,
               answer: Optional[str] = None, seed: int = 0) -> Pool:
    """Assemble a pool whose adversarial share takes each value in `rhos`.

    `promoted` are the adversary's items: authentic in content, adversarial in
    provenance. They are appended to the pool, matching Assumption 3
    (augmentation, not displacement). For the displacing variant see
    build_pool_displacing().
    """
    if not authentic:
        raise ValueError("authentic pool is empty")
    rhos = sorted(rhos)
    if rhos and (rhos[0] <= 0 or rhos[-1] >= 1):
        raise ValueError("every rho must lie strictly in (0,1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(promoted))

    items = list(authentic)
    adversarial: Dict[float, List[int]] = {}
    for rho in rhos:
        # need |A| / (|authentic| + |A|) = rho  =>  |A| = rho*|auth|/(1-rho)
        need = int(round(rho * len(authentic) / (1 - rho)))
        if need > len(promoted):
            raise ValueError(
                f"rho={rho} needs {need} promoted items but only "
                f"{len(promoted)} are available; widen the pool or lower rho")
        while len(items) < len(authentic) + need:
            items.append(promoted[order[len(items) - len(authentic)]])
        adversarial[rho] = list(range(len(authentic), len(authentic) + need))
    return Pool(items=items, adversarial=adversarial, query=query,
                answer=answer, meta={"seed": seed, "n_authentic": len(authentic)})


def build_pool_displacing(authentic: Sequence[str], promoted: Sequence[str],
                          rhos: Sequence[float], query: str,
                          answer: Optional[str] = None, seed: int = 0) -> Pool:
    """Displacement variant: each promoted item EVICTS an authentic one, so the
    pool size is constant. Assumption 3 fails here and the certificate must use
    the 2*Dbar threshold (arq.certificate.certify(..., adversary='displace')).
    """
    rng = np.random.default_rng(seed)
    rhos = sorted(rhos)
    n = len(authentic)
    victims = rng.permutation(n)
    items = list(authentic)
    adversarial: Dict[float, List[int]] = {}
    for rho in rhos:
        need = int(round(rho * n))
        if need > len(promoted):
            raise ValueError(f"rho={rho} needs {need} promoted items, have {len(promoted)}")
        idx = sorted(int(v) for v in victims[:need])
        for k, pos in enumerate(idx):
            items[pos] = promoted[k]
        adversarial[rho] = idx
    return Pool(items=items, adversarial=adversarial, query=query, answer=answer,
                meta={"seed": seed, "mode": "displacing", "n_authentic": n})


def build_synthetic_pool(n_authentic: int, n_promoted: int, rhos: Sequence[float],
                         seed: int = 0) -> Pool:
    """Fabricated entities, for the memory-free control of E1 and for smoke
    tests. Every item is prefixed so a synthetic run can never be mistaken for
    a real one downstream."""
    rng = np.random.default_rng(seed)
    auth = [f"{SYNTHETIC_PREFIX}fact {i}: entity Q{rng.integers(1000)} "
            f"observed at T{i}" for i in range(n_authentic)]
    prom = [f"{SYNTHETIC_PREFIX}fact P{i}: entity Q{rng.integers(1000)} "
            f"observed at T{i}" for i in range(n_promoted)]
    return build_pool(auth, prom, rhos, query=f"{SYNTHETIC_PREFIX}query",
                      answer="TRUE", seed=seed)


# ---------------------------------------------------------------------------
def draw_groups(pool: Pool, t: int, G: int, rho_max: float,
                seed: int = 0, randomize_order: bool = True) -> List[List[int]]:
    """Draw G uniform t-subsets from the pool as it stands at rho_max.

    Order within a group is randomized by default. This is not required by
    Proposition 5.1, which treats the agent as a black box on a presented
    group; it denies the adversary a fixed presentation to optimize against.
    Set randomize_order=False for the ablation.
    """
    if t < 1 or G < 1:
        raise ValueError("need t>=1 and G>=1")
    if rho_max not in pool.adversarial:
        raise KeyError(f"pool was not built at rho={rho_max}")
    rng = np.random.default_rng(seed)
    n = pool.n
    if t > n:
        raise ValueError(f"t={t} exceeds pool size {n}")
    groups = []
    for _ in range(G):
        g = list(rng.choice(n, size=t, replace=False))
        if randomize_order:
            rng.shuffle(g)
        groups.append([int(x) for x in g])
    return groups


def reject_to_rho(pool: Pool, groups: Sequence[Sequence[int]],
                  rho: float, rho_max: float) -> List[List[int]]:
    """Reuse groups drawn at rho_max to evaluate a smaller rho, by discarding
    any group containing an item in A_{rho_max} \\ A_{rho}. Because the
    adversarial sets are nested, the survivors are distributed exactly as
    uniform draws from the rho-pool restricted to its clean-plus-A_rho items.
    """
    if rho > rho_max:
        raise ValueError("rho must not exceed rho_max")
    excluded = set(pool.adversarial[rho_max]) - set(pool.adversarial[rho])
    kept = [list(g) for g in groups if not (set(g) & excluded)]
    if not kept:
        raise ValueError(
            f"rejection left no groups at rho={rho}; draw more at rho_max or "
            "narrow the sweep. Do not proceed with an empty sample.")
    return kept


def is_clean(pool: Pool, group: Sequence[int], rho: float) -> bool:
    """Whether a group contains no adversarial item at share rho. Used only for
    ANALYSIS (measuring q, checking soundness); the mechanism never calls it."""
    if rho not in pool.adversarial:
        raise KeyError(f"pool was not built at rho={rho}")
    return not (set(group) & set(pool.adversarial[rho]))


def pool_size_at(pool: Pool, rho: float) -> int:
    """Number of items in the pool AS IT STANDS at share rho.

    After reject_to_rho, surviving groups are uniform t-subsets of
    (authentic + A_rho), not of the larger rho_max pool. Passing pool.n to
    clean_probability instead of this value understates P_c and would make the
    certificate look weaker than it is.
    """
    if rho not in pool.adversarial:
        raise KeyError(f"pool was not built at rho={rho}")
    if pool.meta.get("mode") == "displacing":
        return pool.n
    return int(pool.meta["n_authentic"]) + len(pool.adversarial[rho])
