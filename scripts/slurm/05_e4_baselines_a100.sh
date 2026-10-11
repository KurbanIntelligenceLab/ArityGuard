#!/bin/bash -l
#SBATCH -J arq_e4_baselines
#SBATCH -o e4_baselines_%j.out
#SBATCH -e e4_baselines_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-00:00:00
#
# REWRITTEN 2 Oct 2026. The prior version of this script called
# `python3 -m arq.run e4 --agent "$MODEL" --match-budget ...` -- arq.run's
# CLI deliberately raises NotImplementedError as soon as --data is reached
# (code/arq/run.py, same stub E1/E3/E5 hit before their own run_*.py
# scripts were written) and that invocation never passed --data at all, so
# this script could never have produced a result. Switched to
# scripts/run_e4.py, the same direct-call pattern every other
# experiment in this repo uses (the development notes C5/D1-D4).
#
# SCOPE NOTE: the development notes's task A4 originally scoped E4 against
# RealtimeQA + NaturalQuestions (the "tau=1 regime, single-fact pools,
# RobustRAG's own benchmark"); those adapters were never written. This
# script instead reuses the SAME relevance-filtered CoPHEME pools E1/E3/E5
# already use real montage-attacked feeds -- the development notes's E4 description
# explicitly asks to "record whether feed_partition's certificate fires on
# montage-attacked feeds", which CoPHEME's pools directly provide and
# RealtimeQA/NQ (plain QA, no adversarial construction) would not. Building
# RealtimeQA/NQ adapters for the tau=1 single-fact regime specifically
# remains a real, separate scope item if that comparison is wanted later --
# flagging this as a deliberate scope decision for this run, not an
# oversight.
#
# code/arq/experiments.py's e4_baselines and scripts/run_e4.py are new
# driver code written 2 Oct 2026 (the development notes: "E2, E4, E6 have no driver
# code at all") -- functionally tested against MockAgent + synthetic pools
# before this script was written (all 6 baselines.py methods ran end to
# end, ARQ's own matched-budget certificate computed alongside them).
#
# llm_judge uses the SAME agent for both analyst and judge roles by default
# (run_e4.py's --judge-model-id would add a genuinely separate model, which
# would need its own GPU headroom on top of the primary agent's -- not
# attempted here; e4_baselines records judge_is_agent=true on every
# llm_judge entry so a same-agent judge is never silently read as a
# stronger two-model comparison).
#
# Requires results/E1_*_seed0.json to exist first (same "read E1, then
# proceed" house rule as E3/E5) -- E4 does not technically depend on E1's
# output, but should not be run before a human has read it.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if ! ls results/E1_*_seed0.json >/dev/null 2>&1; then
    echo "No results/E1_*_seed0.json found. Run and REVIEW 02_e1_calibrate_a100.sh first." >&2
    exit 1
fi

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

mkdir -p results runs/e4_baselines

for MODEL in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"; do
    TAG=$(echo "$MODEL" | tr '/' '_')
    echo "=== E4 baselines (matched budget): $MODEL ==="
    python3 scripts/run_e4.py \
        --model-id "$MODEL" \
        --data-dir data \
        --outdir results \
        --cache "runs/e4_baselines/cache_${TAG}.json" \
        --seed 0 \
        --relevance-filter \
        | tee "runs/e4_baselines/${TAG}_seed0.log"
done

echo "E4 complete. Check results/E4_*_seed0.json's \"summary\" field: if"
echo "ARQ's advantage (arq_accuracy / arq_cert_rate) disappears once"
echo "baselines are given a matched budget (*_matched_accuracy fields),"
echo "that is itself the finding (the development notes Sec. 4 E4)."
