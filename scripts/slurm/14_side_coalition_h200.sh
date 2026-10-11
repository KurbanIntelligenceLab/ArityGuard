#!/bin/bash -l
#SBATCH -J arq_side_coal_h200
#SBATCH -o side_coalition_h200_%j.out
#SBATCH -e side_coalition_h200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-12:00:00
#
# SIDE EXPERIMENT (results/side/coalition/; not in the paper). H200 copy of
# 14_side_coalition_a100.sh: identical job, H200 partition and account. A real-text
# coalition: Sybil accounts repost the CoPHEME montage posts until they hold
# 2.7 / 5.3 / 10% of each relevance-filtered pool, and the agent reads every
# group (no forced votes). ARQ vs feed certificates (share and tail budgets).
# See scripts/run_side_coalition.py. ONE MODEL PER JOB:
#   sbatch scripts/slurm/14_side_coalition_h200.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/14_side_coalition_h200.sh Qwen/Qwen2.5-14B-Instruct
#   sbatch scripts/slurm/14_side_coalition_h200.sh mistralai/Mistral-7B-Instruct-v0.3
# Seeds 0-4; finished seeds are skipped, so resubmitting continues the job.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
source scripts/slurm/side_mode.sh
mkdir -p results/side/coalition runs/side/coalition
CACHE="runs/side/coalition/cache_${CTAG}.json"
# Clean CoPHEME groups were already answered in the relevance-filtered
# calibration run (same prompt, same cache key); start from that cache.
SEEDCACHE="runs/e1_calibrate_relfilt/cache_${TAG}.json"
if [ ! -f "$CACHE" ] && [ -z "$CHAT_FLAG" ] && [ -f "$SEEDCACHE" ]; then
    cp "$SEEDCACHE" "$CACHE"; echo "seeded cache from $SEEDCACHE"
fi
python3 scripts/run_side_coalition.py --model-id "$MODEL" $CHAT_FLAG \
    --cache "$CACHE" | tee "runs/side/coalition/${CTAG}_$(date +%m%d_%H%M).log"
