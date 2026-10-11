"""
Fetch the two known-arity QA datasets to the paths the adapters expect, and
check them byte for byte against the files the paper's results were run on.

    python scripts/download/fetch_qa_datasets.py            # from the repo root

  data/hotpotqa/hotpotqa_distractor_validation.jsonl   HotpotQA, distractor, validation
  data/2wikimultihopqa/2wiki_validation.jsonl          2WikiMultihopQA, validation

Both come from the Hugging Face Hub at pinned revisions and are written with
`Dataset.to_json` defaults, which reproduces the original files exactly (MD5
below, checked 9 Oct 2026 with datasets 5.0.1). An existing file with the right
MD5 is left alone.
"""
from __future__ import annotations

import hashlib
import os
import sys

DATASETS = [
    # (hub repo, config, revision, output path, md5 of the file used in the paper)
    ("hotpotqa/hotpot_qa", "distractor", "1908d6afbbead072334abe2965f91bd2709910ab",
     "data/hotpotqa/hotpotqa_distractor_validation.jsonl", "0931d3fb140f15de6280c829aaaaacb7"),
    ("voidful/2WikiMultihopQA", None, "16852fde9d85cba158cf7e6517e7a3f9415a28c0",
     "data/2wikimultihopqa/2wiki_validation.jsonl", "4e7c8a4b32d5001e5f160ff1c100d22e"),
]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import datasets  # imported here so --help works without it

    bad = 0
    for repo, cfg, rev, out, want in DATASETS:
        if os.path.exists(out) and md5(out) == want:
            print(f"ok (already present): {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"downloading {repo} ({cfg or 'default'}) @ {rev[:10]} -> {out}")
        args = (repo, cfg) if cfg else (repo,)
        ds = datasets.load_dataset(*args, split="validation", revision=rev)
        ds.to_json(out)
        got = md5(out)
        if got == want:
            print(f"ok: {out} (md5 {got})")
        else:
            bad += 1
            print(f"MD5 MISMATCH for {out}: got {got}, expected {want}. The records may "
                  "still be identical (JSON escaping differs across library versions); "
                  "compare parsed records before trusting a rerun.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
