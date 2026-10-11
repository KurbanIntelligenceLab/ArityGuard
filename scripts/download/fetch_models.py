"""
Optional: pre-download the three agents into the Hugging Face cache, so GPU
jobs do not download on first use (useful on clusters whose compute nodes have
no internet). Nothing is written into the repository.

    python scripts/download/fetch_models.py               # all three
    python scripts/download/fetch_models.py --only Qwen/Qwen2.5-7B-Instruct

Set HF_HOME to choose the cache location. The models are not gated; no token
is needed. Revisions are not pinned: the paper's runs used whatever `main`
resolved to in September-October 2026.
"""
from __future__ import annotations

import argparse

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=MODELS)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    for m in a.only:
        print(f"downloading {m}")
        path = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                    "*.txt", "tokenizer*"])
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
