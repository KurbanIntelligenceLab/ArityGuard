#!/bin/bash -l
#SBATCH -J arq_e5b_ablations
#SBATCH -o e5b_ablations_%j.out
#SBATCH -e e5b_ablations_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 0-12:00:00
#
# Table 4 ablations (t, G, fixed order, abstentions discarded, retrieval-
# constrained) on the E4b pools: arity 2, worst-case adversary, rho=0.053,
# 80 questions. See code/arq/experiments.py:e5b_ablations. ONE MODEL PER JOB:
#   sbatch scripts/slurm/08_e5b_ablations_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/08_e5b_ablations_a100.sh Qwen/Qwen2.5-14B-Instruct
# Shares the E4b cache, so many calls are already cached. Resume-safe.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
mkdir -p results runs/e4b_feed_vs_pool
python3 scripts/run_e5b.py --model-id "$MODEL" --outdir results \
    --cache "runs/e4b_feed_vs_pool/cache_${TAG}.json" --seed 0 \
    | tee "runs/e4b_feed_vs_pool/${TAG}_e5b_seed0.log"
