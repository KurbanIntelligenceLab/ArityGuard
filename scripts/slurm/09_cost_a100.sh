#!/bin/bash -l
#SBATCH -J arq_cost
#SBATCH -o cost_%j.out
#SBATCH -e cost_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 0-10:00:00
#
# Measured cost per method (paper E8): calls, prompt tokens and GPU-seconds
# per query, NO cache, on 4 relevance-filtered CoPHEME pools. The influence
# detector and undefended agent read whole pools, so this is the slow one.
# ONE MODEL PER JOB:
#   sbatch scripts/slurm/09_cost_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/09_cost_a100.sh Qwen/Qwen2.5-14B-Instruct
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
mkdir -p results
python3 scripts/run_cost.py --model-id "$MODEL" --outdir results --n-pools 4
