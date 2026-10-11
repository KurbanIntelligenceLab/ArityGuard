#!/bin/bash -l
#SBATCH -J arq_e4b_feed_vs_pool
#SBATCH -o e4b_feed_vs_pool_%j.out
#SBATCH -e e4b_feed_vs_pool_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 0-10:00:00
#
# E4b: the Figure-1 test E4 could not run. Delivered-feed certificate
# (k=12/24 items, groups of t) vs ARQ pool resampling (G=200), at t = 1, 2, 3,
# on arity-2 questions (the same 40 per dataset as E2), against a worst-case
# selection adversary holding 1, 2 or 4 of ~37-40 items (rho ~ 0.03-0.10).
# ONE MODEL PER JOB:
#   sbatch scripts/slurm/05b_e4b_feed_vs_pool_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/05b_e4b_feed_vs_pool_a100.sh Qwen/Qwen2.5-14B-Instruct
# About 17K real model calls per model (pools are redundant, so draws repeat
# and hit the cache): roughly 2h at the ~9K calls/hour seen in E2. Resume-safe.

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
mkdir -p results runs/e4b_feed_vs_pool

echo "=== E4b feed vs pool: $MODEL ==="
python3 scripts/run_e4b.py \
    --model-id "$MODEL" \
    --outdir results \
    --cache "runs/e4b_feed_vs_pool/cache_${TAG}.json" \
    --seed 0 \
    | tee "runs/e4b_feed_vs_pool/${TAG}_seed0.log"
echo "E4b complete for $MODEL: results/E4b_${TAG}_seed0.json"
