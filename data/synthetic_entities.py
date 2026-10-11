"""
Synthetic-entity rewrite for the E2 arity test (paper: "synthetic-entity
set", the control for parametric memory).

E2 found that Qwen2.5-14B answers HotpotQA comparisons from ONE supporting
fact about 74% of the time, so tau came out 1 against a true arity of 2: the
missing fact came from memory. Replacing each supporting entity's name with
an invented one removes the lookup key. If tau then rises to the annotated
arity, memory was the cause.

What is replaced, per example, consistently across the question and every
context sentence (gold and distractor):
  * each supporting title, with any trailing "(disambiguation)" removed,
    by an invented name with the same number of words;
  * each capitalised word of that title (3+ letters, not a function word)
    by the matching invented word, so "Wood" in "Edward Davis Wood Jr."
    is caught as well as "Ed Wood";
  * the lead subject of the title's first sentence (the span before " (",
    " is ", " was ", ...), when it shares a word with the title, by the full
    invented name. That covers Wikipedia leads such as "Edward Davis Wood
    Jr. (October 10, 1924 ...) was ...".
Other names in the sentences (a film a director made, a city) are left
alone, so the rewrite removes the entity's own name, not every clue to it.
`leak_report` measures what is left.

Sentence counts never change, so build_controlled_pools picks the same
questions at the same seed as the real E2 run.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

import numpy as np

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v",
           "z", "br", "dr", "gr", "kr", "pl", "st", "tr", "vr"]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou"]
_CODAS = ["", "", "n", "r", "l", "s", "th", "nd", "rk"]
_STOP = {"the", "of", "and", "a", "an", "in", "on", "for", "de", "la", "le",
         "von", "van", "der", "du", "da", "di", "to", "at", "by", "with"}
# The lead subject stops at the first "(" or ",": 2Wiki writes
# "Curtis Bernhardt( 15 April 1899 - ...) was", and the dates must survive.
_LEAD = re.compile(r"^([^(,]{2,90}?)(?=\s?\(|\s(?:is|was|are|were|has|had)\s|,\s)")


def _word(rng) -> str:
    syl = int(rng.integers(2, 4))
    w = "".join(_ONSETS[rng.integers(len(_ONSETS))] + _VOWELS[rng.integers(len(_VOWELS))]
                for _ in range(syl)) + _CODAS[rng.integers(len(_CODAS))]
    return w.capitalize()


def _rng(*keys) -> np.random.Generator:
    h = hashlib.sha256("::".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _base(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][\w'.-]*", text)
            if len(w.strip(".")) >= 3 and w[0].isupper() and w.lower() not in _STOP]


def common_words(examples, min_share: float = 0.01) -> set:
    """Capitalised words found in at least min_share of all context
    paragraphs (American, Airport, University, Film, ...). They are never
    renamed: renaming "American" would rewrite the very nationality a
    comparison asks about."""
    from collections import Counter
    df, n = Counter(), 0
    for r in examples:
        for _, sents in r["context"]:
            n += 1
            df.update({w.strip(".,") for w in _content_words(" ".join(sents))})
    return {w for w, c in df.items() if c >= min_share * n}


def fake_names(example: dict, seed: int = 0, common: frozenset = frozenset()
               ) -> Dict[str, Tuple[str, Dict[str, str]]]:
    """{title: (fake full name, {real word: fake word})} for each supporting
    title, deterministic in (seed, example id, title). Only rare capitalised
    words are mapped; the fake full name is the title with those words
    swapped, so it reads the same wherever the title appears."""
    out = {}
    titles = []
    for t, _ in example["supporting"]:
        if t not in titles:
            titles.append(t)
    for t in titles:
        rng = _rng(seed, example["id"], t)
        mapping = {}
        for w in _content_words(_base(t)):
            key = w.strip(".,")
            if len(key) >= 3 and key not in common and key not in mapping:
                mapping[key] = _word(rng)
        out[t] = (_sub_words(_base(t), mapping), mapping)
    return out


def _sub_words(text: str, mapping: Dict[str, str]) -> str:
    for real in sorted(mapping, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(real)}(?![\w])", mapping[real], text)
    return text


def synthesize(example: dict, seed: int = 0, common: frozenset = frozenset()) -> dict:
    names = fake_names(example, seed, common)
    full = {_base(t): f for t, (f, _) in names.items()}
    words: Dict[str, str] = {}
    for _, (_, m) in names.items():
        words.update(m)

    def rewrite(text: str) -> str:
        for real in sorted(full, key=len, reverse=True):
            text = text.replace(real, full[real])
        return _sub_words(text, words)

    context = []
    for t, sents in example["context"]:
        sents = list(sents)
        if t in names and sents:
            m = _LEAD.match(sents[0])
            if m and set(_content_words(m.group(1))) & set(names[t][1]):
                sents[0] = names[t][0] + sents[0][m.end(1):]
        context.append((rewrite(t), [rewrite(s) for s in sents]))
    supporting = [(rewrite(t), i) for t, i in example["supporting"]]
    return {**example, "question": rewrite(example["question"]),
            "context": context, "supporting": supporting,
            "synthetic_names": {t: f for t, (f, _) in names.items()}}


class Transform:
    """Callable for build_controlled_pools(transform=...). prepare() sees the
    whole example list first, to find the words that must not be renamed."""
    def __init__(self, seed: int = 0, min_share: float = 0.01):
        self.seed, self.min_share, self.common = seed, min_share, frozenset()

    def prepare(self, examples) -> None:
        self.common = frozenset(common_words(examples, self.min_share))

    def __call__(self, example: dict) -> dict:
        return synthesize(example, self.seed, self.common)


def make_transform(seed: int = 0, min_share: float = 0.01) -> Transform:
    return Transform(seed, min_share)


def leak_report(pools) -> dict:
    """Share of pools whose claim or items still contain a word of an
    original supporting title (meta['original_titles'] must be set)."""
    leaked = 0
    for p in pools:
        real = set()
        for t in p.meta.get("original_titles", []):
            real |= {w.strip(".,") for w in _content_words(_base(t))}
        text = " ".join(p.items) + " " + p.query
        if any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text) for w in real):
            leaked += 1
    return {"pools": len(pools), "with_any_original_title_word": leaked}
