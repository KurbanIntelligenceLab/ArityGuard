"""
data.copheme_adapter -- turns the CoPHEME release (arXiv:2601.01685,
"Lying with Truths", ACL 2026) into arq.pool.Pool objects.

Schema confirmed 8 Sep 2026 against the cloned release at
`data/Lying_with_Truth/` (github: hu-jinwei et al.). This replaces an earlier
version of this file written against a guessed schema before the real data
had been inspected -- see the development notes H3 history if you need the old one.
Research-use permission for this dataset was confirmed by the authors of this work
on 8 Sep 2026 (see the development notes H3 -- resolved; note the source repo itself
carries no LICENSE file, so that confirmation is the operative basis).

REAL LAYOUT (relative to `data/Lying_with_Truth/`):

    CoPHEME/<event>/<event>_evidence.json      -- flat list of tweet dicts
    CoPHEME/<event>/<event>_hypotheses.json    -- flat list of claim dicts
    attack_plan/GPT4.1/<event>/result_<event>_h<idx>.json
                                                -- pre-generated montage attack

`<event>` in {charliehebdo, ferguson, germanwings-crash, ottawashooting,
putinmissing, sydneysiege} -- 6 events, confirmed to match the development notes
Sec. 5's "CoPHEME has only six events" note.

Evidence records (`*_evidence.json`, one list per event, ~100-1800 items):
    {"tweet_id": str, "text": str, "veracity": "non-rumour"|"true",
     "is_rumour": bool, ...}
Only "non-rumour"/"true" tweets appear in evidence files (both denote
genuinely-happened content -- "Local Truth" per the paper's README) --
these become Pool's `authentic` items.

Hypothesis records (`*_hypotheses.json`, one list per event):
    {"tweet_id": str, "text": str, "conclusion": str,
     "veracity": "false"|"unverified", "cascade_size": float, ...}
CONFIRMED: veracity is NEVER "true" in any of the 6 events' hypothesis files
-- every hypothesis is a rumour (false or unverified), which is the point
of the dataset (there is nothing to attack in a claim already agreed true).
`conclusion` is the claim text -- becomes Pool.query.

Attack-plan records (`attack_plan/GPT4.1/<event>/result_<event>_h<idx>.json`,
one JSON object per file): the paper's threat model is that colluding agents
manipulate victim beliefs using ONLY TRUTHFUL evidence fragments, rewritten
and sequenced into a montage narrative -- NOT fabricated content. So the
adversarial ("promoted") items for a Pool are NOT literal tweets from
evidence.json; they are the attack plan's own generated posts:
    d["attack_plan"]["posts"][*]["text"]
these are truthful-in-substance, adversarially-authored and -sequenced
fragments (README: "steer victim beliefs using only truthful evidence
fragments... through strategic narrative construction"), and matching
Assumption 3's "augmenting, not displacing" framing in arq.pool.build_pool.
Gold label comes from `d["metadata"]["ground_truth"]` and
`d["metadata"]["target_conclusion"]` is the claim text (should match the
corresponding hypothesis's `conclusion`, not re-verified here).

As of 8 Sep 2026, 3 of 6 events had a pre-generated attack plan, each for
exactly one hypothesis index (h0): charliehebdo (ground_truth=false),
ferguson (ground_truth=unverified), ottawashooting (ground_truth=unverified;
note the attack_plan directory itself is misspelled "ottawshooting", missing
an "a" -- handled below). As of 9 Sep 2026, full-parity generation was
started for the other 3 events (germanwings-crash: 144 hypotheses,
putinmissing: 126, sydneysiege: 140) via
`scripts/generate_montage_batch.py` + `scripts/07{b,c,d}_generate_montage_*.sh`
(resume-safe, may take multiple SLURM submissions to finish -- check
`ls attack_plan/GPT4.1/<event>/result_<event>_h*.json | wc -l` for progress).

**This adapter discovers ALL available hypothesis indices per event by
scanning the attack_plan directory, not just h0** -- so it picks up new
hypotheses automatically as generation jobs write more files, with no code
change needed per file. `n_pools` will grow from 3 (as of 8 Sep) toward
potentially 500+ as the three new events' generation jobs complete; this was
a deliberate design decision (9 Sep 2026, see the development notes) to maximize
statistical power for E1/E3 rather than keep parity with the original
single-hypothesis-per-event shape.

GOLD-LABEL MAPPING (a judgment call, not given by the data):
    ground_truth == "false"       -> Pool.answer = "FALSE"
    ground_truth == "unverified"  -> Pool.answer = "ABSTAIN"
arq.agent.VERDICTS is only ("TRUE", "FALSE", "ABSTAIN"); CoPHEME hypotheses
are never "true" so that arm is unused here. "unverified" has no natural
TRUE/FALSE reading, so it is mapped to ABSTAIN on the view that abstention is
the epistemically honest verdict when the claim's truth is genuinely
undetermined by the evidence available. This is a modeling choice made
2026-09-08 (see the development notes) -- if it turns out to bias E1's q or tau in a
way that matters, reconsider dropping "unverified" hypotheses entirely rather
than silently keeping this mapping.

RELEVANCE FILTER (added 17 Sep 2026, see the development notes "tau=null" diagnosis):
E1 run against the unfiltered 413-pool dataset came back tau=null -- not
because the model can't reason, but because random draws from a claim's
FULL authentic pool (up to 1821 items) usually contain nothing relevant to
that specific claim (confirmed by cache replay: the model answers ABSTAIN
~99% of the time regardless of gold label, and that's the epistemically
correct response to irrelevant evidence). `build_copheme_pools(...,
relevance_filter=True)` restricts each pool's authentic items to those
matching a proper-noun/entity extracted from the claim before handing them
to build_pool/build_pool_displacing, so is_clean's uniform-random draws are
now uniform over RELEVANT evidence, not the full firehose. This is a
lexical heuristic (see _extract_entities), not ground truth: it will miss
paraphrased-but-relevant evidence (false negative) and can match on a name
used in an unrelated context (false positive). Pools with fewer relevant
items than needed for the largest configured t are skipped entirely (not
silently kept with too few items to draw from) -- see
MIN_RELEVANT_FOR_FILTER.

Two bugs were caught and fixed while building this (17 Sep 2026), both via
an automated no-duplicate-trial check, not by inspection:
1. `_extract_entities` originally treated the claim's first word as a
   possible entity purely because sentences start capitalized ("There are
   13 hostages..." -> spurious entity "There"). Fixed by dropping the
   sentence-initial token before entity matching. Costs some real entities
   that happen to start a sentence (e.g. "Merabet was killed..." now
   extracts nothing), which is a real precision/recall tradeoff, not a
   free fix.
2. CoPHEME's own hypothesis files repeat the same claim conclusion across
   many records (confirmed: sydneysiege has only 33 unique conclusions
   across 140 hypotheses, apparently one record per propagating tweet, not
   per distinct rumour). Once relevance-filtered, a repeated claim yields
   an identical (evidence, query, answer) triple -- a duplicate,
   non-independent trial. build_copheme_pools now deduplicates by exact
   claim text per event when relevance_filter=True (unaffected when False,
   so the original unfiltered 413-pool result stands as reported).

**Net effect on n_pools is large: measured 17 Sep 2026, only 16 of 413
pools survive both the relevance filter and the duplicate-claim
dedup** (9 FALSE / 7 ABSTAIN, spanning 5 of 6 events --
charliehebdo/ferguson/germanwings-crash/putinmissing/sydneysiege each
contribute 1-9, ottawashooting's single hypothesis did not survive
filtering). This is a much smaller, but far more construct-valid, sample
than the unfiltered 413 -- report this shrinkage explicitly if this
filtered result is used in the paper; it is not a neutral subsample.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from arq.pool import Pool, build_pool, build_pool_displacing  # noqa: E402

REPO_SUBDIR = "Lying_with_Truth"

# Maps the CoPHEME event directory name to the (possibly differently-spelled)
# attack_plan subdirectory name. First 3 confirmed 8 Sep 2026 by listing both
# trees; the other 3 added 9 Sep 2026 to match generate_montage_batch.py's
# output directories (no misspelling issue there since we control the output
# path ourselves). Safe to list here even before generation completes --
# _iter_attack_plans (below) globs for whatever files actually exist and
# simply finds none until a SLURM job writes some.
ATTACK_PLAN_DIR_FOR_EVENT = {
    "charliehebdo": "charliehebdo",
    "ferguson": "ferguson",
    "ottawashooting": "ottawshooting",  # sic -- typo in the upstream repo
    "germanwings-crash": "germanwings-crash",
    "putinmissing": "putinmissing",
    "sydneysiege": "sydneysiege",
}

GROUND_TRUTH_TO_ANSWER = {"false": "FALSE", "unverified": "ABSTAIN"}

# Generic capitalized words that appear in almost every tweet about a given
# event and so carry no claim-specific relevance signal on their own --
# extend per-event as needed if a new event's generic terms leak through
# (e.g. seen inflating "relevant" counts to the full pool size).
_GENERIC_CAPITALIZED_TERMS = {
    "Charlie", "Hebdo", "Paris", "First", "Attack", "Police", "French",
    "German", "Germany", "Sydney", "Ottawa", "Canada", "Canadian",
    "Putin", "Russia", "Russian", "Ferguson", "Missouri", "America",
    "American", "Twitter", "Breaking",
}

# A pool needs at least this many relevance-filtered items to be usable at
# the largest configured t (t up to 8 needs 8 distinct items to draw a
# group without replacement) -- pools with fewer are skipped, not silently
# kept undersized.
MIN_RELEVANT_FOR_FILTER = 8


def _extract_entities(claim: str) -> set:
    """Crude but transparent relevance anchor: capitalized proper-noun-like
    tokens (length >= 4) in the claim, excluding generic event-wide terms.
    See RELEVANCE FILTER in the module docstring for why this exists and
    its known failure modes (misses paraphrases, can false-positive on
    unrelated mentions of the same name).

    Deliberately IGNORES the first word of the claim: every sentence starts
    capitalized regardless of whether that word is a proper noun ("There
    are 13 hostages..."), and treating position-0 capitalization as a
    relevance signal was confirmed (17 Sep 2026) to extract "There" as a
    pseudo-entity, matching it against nearly every tweet and silently
    producing duplicate filtered pools for claims that happen to start with
    the same generic word -- caught via a fingerprint-collision check, not
    guessed at; see the development notes for the incident."""
    import re
    words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", claim)
    if words and claim.strip().startswith(words[0]):
        words = words[1:]  # drop only the true sentence-initial occurrence
    return {w for w in words if w not in _GENERIC_CAPITALIZED_TERMS}


def _filter_relevant(items: List[str], entities: set) -> List[str]:
    if not entities:
        return []
    lowered_entities = [e.lower() for e in entities]
    return [item for item in items if any(e in item.lower() for e in lowered_entities)]


def _lying_with_truth_root(data_dir: str) -> str:
    """data_dir is expected to be data/copheme (per the development notes/the development notes
    naming) or to point directly at the cloned Lying_with_Truth checkout.
    Accept either so callers don't have to know which."""
    candidate = os.path.join(data_dir, REPO_SUBDIR)
    if os.path.isdir(os.path.join(candidate, "CoPHEME")):
        return candidate
    if os.path.isdir(os.path.join(data_dir, "CoPHEME")):
        return data_dir
    raise FileNotFoundError(
        f"Could not find a CoPHEME/ directory under {data_dir} or "
        f"{candidate}. Expected the cloned Lying_with_Truth checkout "
        "(github, arXiv:2601.01685) either directly at data_dir or in a "
        f"'{REPO_SUBDIR}' subdirectory of it."
    )


def _load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as fh:
        return json.load(fh)


def _list_events(root: str) -> List[str]:
    copheme = os.path.join(root, "CoPHEME")
    events = sorted(
        d for d in os.listdir(copheme) if os.path.isdir(os.path.join(copheme, d))
    )
    if not events:
        raise RuntimeError(f"No event subdirectories found under {copheme}")
    return events


def _load_authentic_items(root: str, event: str) -> List[str]:
    path = os.path.join(root, "CoPHEME", event, f"{event}_evidence.json")
    records = _load_json(path)
    items = [r["text"] for r in records if r.get("veracity") in ("non-rumour", "true")]
    if not items:
        raise ValueError(f"{path}: zero authentic (non-rumour/true) evidence items")
    return items


def _iter_attack_plans(root: str, event: str):
    """Yield every pre-generated attack plan for `event`, in hypothesis-index
    order. Discovers whatever result_{event}_h*.json files exist -- does not
    assume h0 is the only one, so it picks up new hypotheses as
    generate_montage_batch.py writes more without any code change here."""
    plan_dir = ATTACK_PLAN_DIR_FOR_EVENT.get(event)
    if plan_dir is None:
        return
    import glob
    import re
    pattern = os.path.join(root, "attack_plan", "GPT4.1", plan_dir, f"result_{event}_h*.json")
    paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_h(\d+)\.json$", p).group(1)),
    )
    for path in paths:
        yield _load_json(path)


def build_copheme_pools(data_dir: str, rhos: Sequence[float], seed: int = 0,
                        displacing: bool = False,
                        events: Optional[Sequence[str]] = None,
                        relevance_filter: bool = False,
                        min_relevant: int = MIN_RELEVANT_FOR_FILTER) -> List[Pool]:
    """Build one Pool per available (event, hypothesis) attack plan.

    As of 9 Sep 2026 this builds a pool for EVERY hypothesis index that has
    a generated attack plan file, not just h0 -- see module docstring
    "n_pools will grow..." note. Events with no attack plans at all
    (ATTACK_PLAN_DIR_FOR_EVENT has no entry, or the directory is empty
    because generation hasn't run/finished yet) are skipped with a printed
    notice, same as before.

    Set displacing=True for the displacement-adversary variant (needed by
    E5's stress test).

    Set relevance_filter=True to restrict each pool's authentic items to
    those matching an entity extracted from the claim, per the RELEVANCE
    FILTER section of this module's docstring (added 17 Sep 2026, in
    response to the tau=null finding on the unfiltered 413-pool dataset --
    see the development notes). Pools left with fewer than `min_relevant` items after
    filtering are SKIPPED, not silently kept undersized -- a skip message
    is printed to stderr per skipped hypothesis (distinct from the
    per-event "no attack plan" notice). This substantially reduces
    n_pools and skews the answer-type/event distribution -- see the
    docstring for the measured numbers; report them if this filtered
    result is used in the paper.

    Raises loudly if data_dir doesn't contain the expected CoPHEME/ and
    attack_plan/ trees, or if a record is missing an expected field.
    """
    root = _lying_with_truth_root(data_dir)
    all_events = _list_events(root)
    wanted = list(events) if events is not None else all_events
    max_t_needed = 8  # E1's default t_values go up to 8; see run_e1.py

    builder = build_pool_displacing if displacing else build_pool
    pools: List[Pool] = []
    skipped = []
    skipped_by_relevance = []
    skipped_as_duplicate = []
    pool_seed_offset = 0
    for event in wanted:
        authentic = None  # loaded lazily, once per event, only if >=1 plan exists
        n_for_event = 0
        seen_claims = set()  # (event-scoped) claim texts already kept, relevance_filter only
        for plan in _iter_attack_plans(root, event):
            if authentic is None:
                authentic = _load_authentic_items(root, event)

            promoted = [p["text"] for p in plan["attack_plan"]["posts"]]
            if not promoted:
                raise ValueError(
                    f"{event} h{plan['metadata']['hypothesis_index']}: "
                    "attack plan has zero posts"
                )

            ground_truth = plan["metadata"]["ground_truth"]
            if ground_truth not in GROUND_TRUTH_TO_ANSWER:
                raise ValueError(
                    f"{event} h{plan['metadata']['hypothesis_index']}: "
                    f"unrecognized ground_truth {ground_truth!r}. Expected "
                    f"one of {list(GROUND_TRUTH_TO_ANSWER)} -- see module "
                    "docstring's GOLD-LABEL MAPPING section."
                )
            answer = GROUND_TRUTH_TO_ANSWER[ground_truth]
            query = plan["metadata"]["target_conclusion"]

            pool_authentic = authentic
            if relevance_filter:
                # CoPHEME's hypothesis files repeat the same claim
                # conclusion across many records (confirmed 17 Sep 2026:
                # sydneysiege has only 33 unique conclusions across 140
                # hypotheses -- likely one record per propagating tweet,
                # not per distinct rumour). Once relevance-filtered, a
                # repeated claim yields an identical pool -- a duplicate,
                # non-independent trial, not a new data point. Keep only
                # the first occurrence per event.
                if query in seen_claims:
                    skipped_as_duplicate.append(
                        (event, plan["metadata"]["hypothesis_index"])
                    )
                    n_for_event += 1
                    continue
                seen_claims.add(query)

                entities = _extract_entities(query)
                relevant = _filter_relevant(authentic, entities)
                if len(relevant) < max(min_relevant, max_t_needed):
                    skipped_by_relevance.append(
                        (event, plan["metadata"]["hypothesis_index"], len(relevant))
                    )
                    n_for_event += 1  # event itself did have a plan; not an
                                      # event-level skip, just this hypothesis
                    continue
                pool_authentic = relevant

            # Distinct seed per pool: build_pool's adversarial-set sampling
            # would otherwise draw the identical promoted-item order for
            # every hypothesis sharing the same base seed.
            pool = builder(
                authentic=pool_authentic,
                promoted=promoted,
                rhos=rhos,
                query=query,
                answer=answer,
                seed=seed + pool_seed_offset,
            )
            pool.meta["event_id"] = event
            pool.meta["hypothesis_index"] = plan["metadata"]["hypothesis_index"]
            pool.meta["ground_truth_raw"] = ground_truth
            pool.meta["relevance_filtered"] = relevance_filter
            pools.append(pool)
            pool_seed_offset += 1
            n_for_event += 1

        if n_for_event == 0:
            skipped.append(event)

    if skipped:
        print(
            f"NOTE: skipped {len(skipped)} event(s) with no pre-generated "
            f"attack plan: {skipped}. Generate them with "
            "scripts/generate_montage_batch.py to include them.",
            file=sys.stderr,
        )
    if skipped_by_relevance:
        print(
            f"NOTE: relevance_filter skipped {len(skipped_by_relevance)} "
            f"hypothesis(es) with fewer than {min_relevant} claim-relevant "
            f"evidence items (of {len(skipped_by_relevance) + len(pools)} "
            "considered). Not included in the returned pools.",
            file=sys.stderr,
        )
    if skipped_as_duplicate:
        print(
            f"NOTE: relevance_filter skipped {len(skipped_as_duplicate)} "
            "hypothesis(es) whose claim text duplicates an earlier "
            "hypothesis in the same event (CoPHEME repeats claim "
            "conclusions across multiple source tweets) -- kept only the "
            "first occurrence per event to avoid non-independent trials.",
            file=sys.stderr,
        )
    if not pools:
        raise RuntimeError(
            f"Zero pools built from {data_dir} -- no requested event had a "
            "usable attack plan."
        )
    return pools


def split_calibration_evaluation(
    pools: Sequence[Pool], calibration_fraction: float = 0.5, seed: int = 0,
) -> Tuple[List[Pool], List[Pool]]:
    """Split pools into disjoint calibration/evaluation sets, stratified by
    event_id.

    For an event with more than one pool (the normal case once
    build_copheme_pools' full-parity generation, added 9 Sep 2026, has run),
    this does a real within-event split at calibration_fraction. For an
    event with exactly one pool (still true for charliehebdo/ferguson/
    ottawashooting if their other hypotheses haven't been generated, or any
    event mid-generation), that single pool can't itself be split -- it goes
    entirely to one side via a seeded coin flip, so repeated calls across
    many such events still approximate calibration_fraction in aggregate.
    Raises if the overall result leaves either half completely empty.
    """
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0, 1)")

    by_event: Dict[str, List[Pool]] = {}
    for p in pools:
        by_event.setdefault(p.meta.get("event_id", "UNKNOWN"), []).append(p)

    import numpy as np
    rng = np.random.default_rng(seed)

    cal: List[Pool] = []
    ev: List[Pool] = []
    for event_id, group in sorted(by_event.items()):
        idx = rng.permutation(len(group))
        if len(group) == 1:
            # Can't split a single pool; assign the whole event to one side,
            # alternating by a seeded coin flip so repeated calls over many
            # single-pool events still approximate calibration_fraction.
            target = cal if rng.random() < calibration_fraction else ev
            target.extend(group)
            continue
        n_cal = max(1, int(round(len(group) * calibration_fraction)))
        cal_idx = set(idx[:n_cal].tolist())
        for i, p in enumerate(group):
            (cal if i in cal_idx else ev).append(p)

    if not cal or not ev:
        raise RuntimeError(
            f"Split produced an empty half (calibration={len(cal)}, "
            f"evaluation={len(ev)}) from {len(pools)} pools across "
            f"{len(by_event)} events. With only 3 events currently usable "
            "(one pool each), an unlucky seed can put all of them on one "
            "side -- try a different seed, or wait until more attack plans "
            "exist so a within-event split is possible."
        )
    return cal, ev
