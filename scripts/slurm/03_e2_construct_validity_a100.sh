#!/bin/bash -l
#SBATCH -J arq_e2_construct_validity
#SBATCH -o e2_construct_validity_%j.out
#SBATCH -e e2_construct_validity_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-12:00:00
#
# E2: does tau measure arity? ONE MODEL PER JOB. Submit each model separately
# so they run in parallel:
#   sbatch scripts/slurm/03_e2_construct_validity_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/03_e2_construct_validity_a100.sh Qwen/Qwen2.5-14B-Instruct
#
# REDESIGNED 3 Oct 2026 (see run_e2.py's docstring). The first version (job
# 408674) sampled from each question's full ~37-sentence context and timed
# out at about 5% of the 7B run. Its cache replay showed q(t) of 0.03 to
# 0.18, because a random group almost never holds both supporting facts.
# This version uses controlled pools: the gold supporting items plus d in
# {0, 4, 16} distractor sentences, for 40 questions each from HotpotQA,
# 2Wiki comparison, and 2Wiki bridge_comparison.
#
# Cost: about 117K real model calls per model (draws repeat on small pools
# and hit the cache). About 13h for 7B at the ~9K calls/hour seen in job
# 408674, and longer for 14B. Walltime is 36h. Resume-safe: re-sbatch the same
# command after a timeout and cached calls are reused.
#
# Exit 3 (FALSIFICATION) is raised only if tau fails to recover arity on
# the undiluted d=0 pools. If d>0 pools show tau above arity, that is the
# dilution result being measured, not a failure.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi
mkdir -p results runs/e2_controlled

echo "=== E2 construct validity (controlled pools): $MODEL ==="
set +e
python3 scripts/run_e2.py \
    --model-id "$MODEL" \
    --hotpotqa-dir data/hotpotqa \
    --twowiki-dir data/2wikimultihopqa \
    --outdir results \
    --cache "runs/e2_controlled/cache_${TAG}.json" \
    --seed 0 \
    --groups-per-t 80 \
    --n-examples 40 \
    --distractors 0 4 16 \
    | tee "runs/e2_controlled/${TAG}_seed0.log"
exit_code=${PIPESTATUS[0]}
set -e
if [ "$exit_code" -eq 3 ]; then
    echo "FALSIFICATION: tau did not recover arity on a d=0 dataset for $MODEL." >&2
    echo "Read results/E2_${TAG}_seed0.json before using tau elsewhere." >&2
elif [ "$exit_code" -ne 0 ]; then
    echo "E2 failed for $MODEL (exit $exit_code)." >&2
    exit "$exit_code"
fi
echo "E2 complete for $MODEL. Per dataset, compare 'tau' with"
echo "'ground_truth_arity'; read the d=0 rows as the arity test and the"
echo "d=4/d=16 rows as the dilution curve."
