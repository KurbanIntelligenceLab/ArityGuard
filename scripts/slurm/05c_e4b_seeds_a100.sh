#!/bin/bash -l
#SBATCH -J arq_e4b_seeds
#SBATCH -o e4b_seeds_%j.out
#SBATCH -e e4b_seeds_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-00:00:00
#
# E4b on seeds 1-4 (seed 0 already done), for the five seeds the paper's
# Statistics paragraph promises. Each seed redraws the 40 questions per
# dataset, the feeds and the groups. ONE MODEL PER JOB:
#   sbatch scripts/slurm/05c_e4b_seeds_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/05c_e4b_seeds_a100.sh Qwen/Qwen2.5-14B-Instruct
# Skips any seed whose result file already exists, so it is safe to resubmit.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
mkdir -p results runs/e4b_feed_vs_pool
for SEED in 1 2 3 4; do
    if [ -f "results/E4b_${TAG}_seed${SEED}.json" ]; then
        echo "seed $SEED done, skipping"; continue
    fi
    echo "=== E4b seed $SEED: $MODEL ==="
    python3 scripts/run_e4b.py --model-id "$MODEL" --outdir results \
        --cache "runs/e4b_feed_vs_pool/cache_${TAG}.json" --seed "$SEED" \
        | tee "runs/e4b_feed_vs_pool/${TAG}_seed${SEED}.log"
done
