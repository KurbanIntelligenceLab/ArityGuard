"""
Resume-aware wrapper around Lying_with_Truth's generate_montage_plan.py.

Why this exists rather than calling generate_montage_plan.py directly:

1. NO CHECKPOINTING upstream. run_batch() in generate_montage_plan.py has no
   skip-if-exists check and no per-hypothesis error handling -- a crash at
   hypothesis k loses no prior work (each hypothesis is saved immediately
   after completing) but ALSO does not stop; on RE-invocation you'd redo
   already-completed hypotheses unless you manually pass --start-idx. This
   script scans the output directory for existing
   result_{event}_h{i}.json files and skips any index already present,
   so re-submitting the same SLURM job after a crash or a walltime kill
   picks up where it left off with zero manual bookkeeping.
2. NO PER-HYPOTHESIS ERROR HANDLING upstream. One uncaught exception in
   Director.manage_debate (e.g. a single bad generation) kills the entire
   batch, including hypotheses that were never attempted. This script
   catches per-hypothesis and logs failures to a sidecar
   .failed_indices.json file, then continues to the next hypothesis.
3. HARDCODED 2-GPU ASSUMPTION upstream. generate_montage_plan.py sets
   CUDA_VISIBLE_DEVICES="0,1" and llm_client.py's huggingface path defaults
   to tensor_parallel_size=2, visible_devices="0,1" -- both wrong for a
   1-GPU SLURM allocation (--gres=gpu:1, matching every other job in this
   project). This script does NOT import generate_montage_plan.py itself
   (avoiding its module-level CUDA_VISIBLE_DEVICES override and its
   `from huggingface_hub import login; login('')` call, which passes an
   invalid empty-string token and can raise depending on cached-credential
   state) -- it re-implements the same call sequence directly against
   src.utils.llm_client / src.production.{director,writer,editor} with
   single-GPU-correct settings.

Usage (see scripts/07b_generate_montage_*.sh for the SLURM wrapper):
    python3 generate_montage_batch.py --event germanwings-crash \
        --output-dir "attack_plan/GPT4.1/germanwings-crash" \
        --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "Lying_with_Truth")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

# Deliberately NOT importing generate_montage_plan.py -- see module docstring.
from src.utils.llm_client import get_llm_client  # noqa: E402
from src.utils.data_loader import CollusiveDataset  # noqa: E402
from src.production.director import Director  # noqa: E402
from src.production.writer import Writer  # noqa: E402
from src.production.editor import Editor  # noqa: E402
import yaml  # noqa: E402


def load_configs(repo_root: str):
    with open(os.path.join(repo_root, "configs/settings.yaml")) as f:
        settings = yaml.safe_load(f)
    with open(os.path.join(repo_root, "configs/agents.yaml")) as f:
        agents = yaml.safe_load(f)
    return settings, agents


def completed_indices(output_dir: str, event_name: str) -> set:
    out = set()
    if not os.path.isdir(output_dir):
        return out
    prefix, suffix = f"result_{event_name}_h", ".json"
    for name in os.listdir(output_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            idx_str = name[len(prefix):-len(suffix)]
            if idx_str.isdigit():
                out.add(int(idx_str))
    return out


def generate_one(event_name, hypothesis_index, settings, agents_config,
                 llm_client, dataset, output_dir, verbose=True):
    """One hypothesis's full Director/Writer/Editor debate, matching
    generate_montage_plan.py's run_single + save_attack_plan exactly."""
    evidence_pool = dataset.get_evidence_pool(event_name)
    hypotheses = dataset.get_target_hypotheses(event_name)
    if hypothesis_index >= len(hypotheses):
        raise IndexError(
            f"hypothesis index {hypothesis_index} out of range "
            f"(max {len(hypotheses) - 1}) for {event_name}"
        )
    target_hypothesis = hypotheses[hypothesis_index]

    writer = Writer(llm_client=llm_client,
                    system_prompt=agents_config["production_team"]["writer"]["system_prompt"])
    editor = Editor(llm_client=llm_client,
                    system_prompt=agents_config["production_team"]["editor"]["system_prompt"])
    director = Director(
        llm_client=llm_client, writer=writer, editor=editor,
        system_prompt=agents_config["production_team"]["director"]["system_prompt"],
        max_debate_rounds=settings["production"]["max_debate_rounds"],
        max_revision_rounds=settings["production"]["max_revision_rounds"],
    )
    attack_plan = director.manage_debate(
        target_hypothesis=target_hypothesis, evidence_pool=evidence_pool, verbose=verbose
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "metadata": {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "event": event_name,
            "hypothesis_index": hypothesis_index,
            "target_conclusion": target_hypothesis.conclusion,
            "ground_truth": target_hypothesis.veracity,
            "adjudication_enabled": False,
        },
        "attack_plan": {
            "narrative": attack_plan.narrative.narrative_text,
            "narrative_metadata": attack_plan.narrative.metadata,
            "num_posts": len(attack_plan.post_sequence.posts),
            "posts": attack_plan.post_sequence.posts,
            "approval_metadata": attack_plan.approval_metadata,
        },
        "victim_feed": {
            "total_posts": len(attack_plan.post_sequence.posts),
            "attack_posts": len(attack_plan.post_sequence.posts),
            "noise_posts": 0,
            "posts": attack_plan.post_sequence.posts,
        },
        "victim_verdicts": [],
        "analysis": {},
        "adjudication": None,
    }
    filepath = output_path / f"result_{event_name}_h{hypothesis_index}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return filepath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--start-idx", type=int, default=None)
    ap.add_argument("--end-idx", type=int, default=None)
    ap.add_argument("--use-vllm", action="store_true",
                    help="Use vLLM backend (faster for many sequential calls); "
                         "default is the transformers-based local client.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    settings, agents_config = load_configs(REPO_ROOT)
    # settings.yaml's dataset.data_root ("data") assumes a data/<event>/...
    # layout that does not exist in this clone -- the real CoPHEME data
    # lives directly at <repo>/CoPHEME/<event>/... (confirmed 9 Sep 2026).
    # CollusiveDataset needs a directory containing event subdirs directly,
    # so point at CoPHEME/ rather than trusting the settings.yaml value.
    dataset = CollusiveDataset(os.path.join(REPO_ROOT, "CoPHEME"))

    total = len(dataset.get_target_hypotheses(args.event))
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else total - 1
    if start < 0 or end >= total or start > end:
        print(f"ERROR: invalid range [{start}, {end}] for {total} hypotheses in {args.event}",
              file=sys.stderr)
        return 1

    done = completed_indices(args.output_dir, args.event)
    todo = [i for i in range(start, end + 1) if i not in done]
    print(f"=== {args.event}: {len(done & set(range(start, end + 1)))} already done, "
          f"{len(todo)} remaining in range [{start}, {end}] ===")
    if not todo:
        print("Nothing to do.")
        return 0

    if args.use_vllm:
        llm_client = get_llm_client(
            provider="huggingface", model_name=args.model,
            temperature=settings["llm"]["temperature"], max_tokens=settings["llm"]["max_tokens"],
            use_local=False, use_vllm=True, tensor_parallel_size=1,
            gpu_memory_utilization=0.9, visible_devices=None,
        )
    else:
        llm_client = get_llm_client(
            provider="huggingface", model_name=args.model,
            temperature=settings["llm"]["temperature"], max_tokens=settings["llm"]["max_tokens"],
            use_local=True, use_vllm=False, device="auto",
            load_in_8bit=False, load_in_4bit=False,
        )

    failed_path = os.path.join(args.output_dir, ".failed_indices.json")
    failed = []
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = json.load(f)

    t0 = time.time()
    for n, i in enumerate(todo, 1):
        print(f"\n--- {args.event} h{i} ({n}/{len(todo)}, "
              f"{time.time() - t0:.0f}s elapsed) ---")
        try:
            path = generate_one(args.event, i, settings, agents_config,
                                llm_client, dataset, args.output_dir, verbose)
            print(f"OK: wrote {path}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED h{i}: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append({"index": i, "error": str(e), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            os.makedirs(args.output_dir, exist_ok=True)
            with open(failed_path, "w") as f:
                json.dump(failed, f, indent=1)
            continue  # one bad hypothesis does not kill the batch

    n_done_now = len(completed_indices(args.output_dir, args.event) & set(range(start, end + 1)))
    print(f"\n=== {args.event}: {n_done_now}/{end - start + 1} complete in range "
          f"[{start},{end}], {len(failed)} recorded failures (see {failed_path}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
