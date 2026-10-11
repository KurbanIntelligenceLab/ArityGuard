"""
arq.agent -- verdict elicitation.

The mechanism needs one thing from an agent: a verdict from a finite set, given
a presented group. Greedy decoding is required (Assumption 4): a verdict must
be a function of its group alone and independent across groups.

THREE ADAPTERS
  LocalAgent   open-weight model served locally (vLLM / transformers)
  APIAgent     hosted model behind an HTTP API
  MockAgent    deterministic stub for smoke-testing the PIPELINE ONLY

MockAgent stamps every verdict so that a mock run cannot be mistaken for a real
one, and Runner refuses to write results from a mock agent into the real
results directory. A parse failure RAISES; it never defaults to a verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

VERDICTS = ("TRUE", "FALSE", "ABSTAIN")
MOCK_TAG = "MOCK"

# Deliberate, narrow exception to "never guess a verdict" -- NOT a general
# fuzzy matcher. Qwen2.5-14B-Instruct was observed (9 Sep 2026, E1 run
# 390753) to consistently emit "ABSTRAIN" instead of "ABSTAIN" -- confirmed
# NOT a token-truncation artifact (the full continuation was visible at a
# wide max_new_tokens budget: " ABSTRAIN\n\nThe provided evidence does not
# contain any information..."), i.e. this is a stable, reproducible spelling
# quirk of that model with this prompt, not ambiguous output. Only entries
# added here after being independently confirmed reproducible should stay;
# this must never grow into a catch-all for arbitrary near-misses -- that
# would defeat the point of raising on unparseable output. See the development notes
# for the incident this was added for.
KNOWN_VERDICT_TYPOS = {"ABSTRAIN": "ABSTAIN"}

# SEPARATE from KNOWN_VERDICT_TYPOS above -- this is a semantic equivalence
# map, not a spelling-correction map, and the distinction matters: these are
# real words the model chose deliberately, not misspellings of an intended
# word. Qwen2.5-7B-Instruct was observed (13 Sep 2026, E1 run 392717) to
# answer "DEVELOPING" (e.g. "DEVELOPING\nThe evidence provided includes
# mentions of an 'ISIS...") when the prompt's three-way choice does not fit
# -- confirmed not a truncation artifact the same way (full continuation
# visible at a widened budget, same word intact throughout). Treated as
# equivalent to ABSTAIN on the view that "situation still developing /
# inconclusive" and "insufficient evidence to conclude" express the same
# underlying epistemic state the prompt's ABSTAIN option is meant to
# capture. This is a judgment call, not a fact about the model -- if it
# turns out to systematically bias q/tau, reconsider (e.g. exclude these
# trials instead of coercing them). Only add an entry here after
# independently confirming the model is offering an off-menu alternative
# with a stable, reproducible meaning -- this must never become a catch-all
# for arbitrary strings that aren't TRUE/FALSE/ABSTAIN. See the development notes for
# the incident this was added for.
#
# Second entry added 14 Sep 2026 (E1 run 392876, 83.8% through 7B's
# calibration): Qwen2.5-7B-Instruct answered "INITIAL" (e.g.
# "INITIAL\nThe evidence provided includes an initial report suggesting a
# female..."), i.e. flagging the evidence as only a preliminary/unconfirmed
# report rather than committing to TRUE/FALSE/ABSTAIN. Same equivalence
# reasoning as DEVELOPING: "this is only an initial/unconfirmed report" is
# the same epistemic stance as "insufficient evidence to conclude." Three
# such hedging-flavored off-menu words in one run (counting the earlier
# ABSTRAIN typo) is a real pattern worth tracking -- if a fourth appears,
# stop adding entries reflexively and reconsider whether the PROMPT itself
# needs to explicitly forbid hedges, rather than patching words one at a
# time indefinitely. Deferred this time because the cache was already
# ~84% built against the current prompt; changing the prompt now would
# make the cached majority inconsistent with newly-generated trials.
KNOWN_OFF_MENU_EQUIVALENTS = {"DEVELOPING": "ABSTAIN", "INITIAL": "ABSTAIN"}

# PROMPT changed 26 Sep 2026 -- adopted the "contradicts" wording from the
# prompt/decoding investigation (scripts/prompt_variants.py, results in
# results/prompt_variants/), per the paper authors' decision. The old
# wording asked whether the claim was "supported" and only mentioned
# ABSTAIN for "insufficient" evidence -- it never told the model that
# evidence CONTRADICTING the claim (not just failing to support it) means
# FALSE. Cache-verified on Qwen2.5-7B-Instruct across the same 16
# relevance-filtered pools/seed/sampling as the old prompt: FALSE-verdict
# rate on FALSE-labeled pools rose from 0.78% to 19.78%, and the accuracy
# trend across arity reversed from declining (0.419->0.393) to rising
# (0.450->0.495, q_lower 0.480-0.481 by t=7-8). This is NOT a fix -- tau
# is still null even under this wording, q_lower does not cross 0.5 at any
# tested arity -- it is the strongest single lever found in that
# investigation, adopted as a real, substantial, reproducible improvement,
# not as a claim of full resolution. See the development notes for the
# full 5-way comparison (original 3-pool, unfiltered 413-pool, baseline
# 16-pool, this wording, and the fewshot/wider_budget variants that did
# not improve on it further).
#
# CONSEQUENCE: every existing results/E1_*.json, results/E3_*.json,
# results/E5_*.json (and their relevance_filtered/prompt_variants
# counterparts) was generated under the OLD prompt and is now stale for
# any comparison against a fresh run -- they remain valid as a record of
# what the old prompt did, but do not mix old- and new-prompt trials when
# computing tau/q. Re-run E1 under this prompt before drawing conclusions
# from it.
PROMPT = """You are a neutral analyst. Using ONLY the evidence below, decide
whether the claim is true or false.
- If the evidence supports the claim, answer TRUE.
- If the evidence CONTRADICTS or REFUTES the claim -- for example, it
  states a different number, name, or fact than the claim asserts -- answer
  FALSE. Do not answer ABSTAIN just because the evidence does not use the
  claim's exact wording; if it establishes a fact incompatible with the
  claim, that is a contradiction.
- Only if the evidence has nothing to do with the claim at all, answer
  ABSTAIN.

Evidence:
{evidence}

Claim: {claim}

Answer with exactly one word: TRUE, FALSE, or ABSTAIN.
Answer:"""


@dataclass
class Agent:
    name: str
    def verdict(self, items: Sequence[str], claim: str) -> str:
        raise NotImplementedError
    def is_mock(self) -> bool:
        return False

    @staticmethod
    def _parse(raw: str) -> str:
        """Map a model response to a verdict. Raises on anything unrecognized:
        a parse failure means the query failed, and a failed query must be
        re-run, not filled in.

        Two narrow, explicitly curated exceptions exist, checked only as a
        fallback after the strict regex fails, and only for exact strings in
        their respective maps -- neither is fuzzy matching:
          KNOWN_VERDICT_TYPOS           confirmed-reproducible misspellings
          KNOWN_OFF_MENU_EQUIVALENTS    confirmed-reproducible off-menu words
                                        treated as semantically equivalent
                                        to one of the three real verdicts
        See each map's own comment for the incident it was added for."""
        if raw is None:
            raise ValueError("agent returned None")
        upper = raw.strip().upper()
        m = re.search(r"\b(TRUE|FALSE|ABSTAIN)\b", upper)
        if m:
            return m.group(1)
        for typo, canonical in KNOWN_VERDICT_TYPOS.items():
            if re.search(rf"\b{typo}\b", upper):
                return canonical
        for word, canonical in KNOWN_OFF_MENU_EQUIVALENTS.items():
            if re.search(rf"\b{word}\b", upper):
                return canonical
        raise ValueError(
            f"unparseable verdict {raw[:80]!r}. Fix the prompt or the "
            "decoding settings; do NOT default to a verdict.")

    @staticmethod
    def _render(items: Sequence[str], claim: str) -> str:
        ev = "\n".join(f"- {s}" for s in items)
        return PROMPT.format(evidence=ev, claim=claim)


class LocalAgent(Agent):
    """Open-weight model. Greedy decoding is mandatory (Assumption 4).

    max_new_tokens=4 is tight: with some tokenizers "ABSTAIN" alone costs 3
    tokens (e.g. Qwen2.5: AB/ST/AIN), leaving only 1 token of slack for a
    leading space or an off-by-one split, so a genuine ABSTAIN verdict can be
    truncated into unparseable garbage (observed: "ABSTAIN" -> " ABSTRAIN").
    On a parse failure this retries with a wider token budget rather than
    silently defaulting to a verdict (Assumption 4's "independent across
    groups" is not violated: decoding stays greedy/deterministic throughout,
    only the token budget changes across retries, so a given group still
    maps to exactly one final verdict). If every retry still fails to parse,
    this still raises -- a persistent parse failure is real data about the
    model/prompt, not something to paper over.
    """
    def __init__(self, model_id: str, max_new_tokens: int = 4, device: str = "cuda",
                max_retries: int = 3, retry_token_growth: int = 4):
        super().__init__(name=model_id)
        self.model_id, self.max_new_tokens, self.device = model_id, max_new_tokens, device
        self.max_retries, self.retry_token_growth = max_retries, retry_token_growth
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch  # noqa
        ids = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_new_tokens,
                                       do_sample=False)      # greedy
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    def verdict(self, items, claim):
        self._load()
        p = self._render(items, claim)
        last_raw = last_err = None
        for attempt in range(self.max_retries):
            budget = self.max_new_tokens + attempt * self.retry_token_growth
            raw = self._generate(p, budget)
            try:
                return self._parse(raw)
            except ValueError as e:
                last_raw, last_err = raw, e
        raise ValueError(
            f"unparseable verdict after {self.max_retries} attempts (widening "
            f"max_new_tokens each time, final budget {budget}): {last_raw[:80]!r}. "
            "Fix the prompt or the decoding settings; do NOT default to a "
            f"verdict."
        ) from last_err


class APIAgent(Agent):
    """Hosted model. temperature=0 for the same reason."""
    def __init__(self, model_id: str, endpoint: str, api_key_env: str = "ARQ_API_KEY",
                 max_retries: int = 4):
        super().__init__(name=model_id)
        self.model_id, self.endpoint, self.max_retries = model_id, endpoint, max_retries
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. The package will not run an API "
                "agent without credentials, and will not silently fall back.")

    def verdict(self, items, claim):
        import requests  # noqa
        body = {"model": self.model_id, "temperature": 0, "max_tokens": 4,
                "messages": [{"role": "user", "content": self._render(items, claim)}]}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.endpoint, json=body, timeout=60,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return self._parse(r.json()["choices"][0]["message"]["content"])
            except Exception as e:                       # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API failed after {self.max_retries} attempts: {last}")


class MockAgent(Agent):
    """Deterministic stub for smoke-testing the pipeline.

    Its verdicts are a hash of the group contents, so they are reproducible and
    carry no information about any real model. Runner refuses to write mock
    results anywhere except a directory prefixed MOCK.
    """
    def __init__(self, adversarial_marker: str = "PROMOTED", q: float = 0.9):
        super().__init__(name=f"{MOCK_TAG}-agent")
        self.marker, self.q = adversarial_marker, q

    def is_mock(self) -> bool:
        return True

    def verdict(self, items, claim):
        if any(self.marker in s for s in items):
            return "FALSE"                                   # adversary's target
        h = int(hashlib.sha256("".join(items).encode()).hexdigest()[:8], 16)
        return "TRUE" if (h % 1000) / 1000 < self.q else "ABSTAIN"


class CachingAgent(Agent):
    """Wrap any agent with an on-disk cache keyed by (model, RENDERED PROMPT).

    The rho sweep re-scores the same groups at several rho values; without a
    cache that is a multiple of the true cost. Cache hits are logged so the
    reported call count is the number of REAL calls, not lookups.

    KEYED ON THE RENDERED PROMPT, NOT ON (items, claim) -- fixed 28 Sep
    2026. Keying on (model.name, items, claim) alone means the cache is
    blind to the PROMPT template itself: if agent.py's PROMPT changes (as
    it did that day, adopting the `contradicts` wording) but the same
    items/claim pairs were already cached under the old prompt, every one
    of those cache entries is silently served under the new prompt's run
    with zero indication anything is stale -- a run can complete with
    zero real model calls and report old-prompt answers as if they were
    new-prompt ones. Caught by exactly this happening:
    results/relevance_filtered/E1_Qwen_Qwen2.5-7B-Instruct_seed0.json came
    back byte-identical to its pre-PROMPT-change version after the
    "fresh" run (job 404363, 28 Sep 2026) -- its cache file
    (runs/e1_calibrate_relfilt/cache_Qwen_Qwen2.5-7B-Instruct.json, dated
    17 Sep) had 34,617 entries already on disk from the OLD prompt, and
    every group in this run's sampling hit that cache, so the model was
    never actually re-queried under the new prompt. This is a general
    class of bug (any future prompt/decoding change silently invalidates
    nothing), not specific to that one script's stale file -- fixed here,
    at the one choke point every experiment's caching goes through,
    rather than by archiving individual cache files each time.
    """
    def __init__(self, inner: Agent, path: str):
        super().__init__(name=inner.name)
        self.inner, self.path = inner, path
        self.hits = self.misses = 0
        self._c: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path) as fh:
                self._c = json.load(fh)

    def is_mock(self) -> bool:
        return self.inner.is_mock()

    def _key(self, items, claim) -> str:
        rendered = self.inner._render(items, claim)
        return hashlib.sha256(
            json.dumps([self.inner.name, rendered]).encode()).hexdigest()

    def verdict(self, items, claim):
        k = self._key(items, claim)
        if k in self._c:
            self.hits += 1
            return self._c[k]
        v = self.inner.verdict(items, claim)
        self._c[k] = v
        self.misses += 1
        return v

    def flush(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self._c, fh)
