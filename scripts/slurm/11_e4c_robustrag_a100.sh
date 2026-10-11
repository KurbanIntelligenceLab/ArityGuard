#!/bin/bash -l
#SBATCH -J arq_e4c_robustrag
#SBATCH -o e4c_robustrag_%j.out
#SBATCH -e e4c_robustrag_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-00:00:00
#
# E4c: feed vs pool (the E4b comparison) on RobustRAG's own RealtimeQA and
# Natural Questions data, 40 questions each, 30 retrieved passages per
# question, worst-case adversary holding 1/2/4 items, t in {1,2}. About 33K
# distinct calls (short passages): ~3h for 7B, ~6h for 14B.
#   sbatch scripts/slurm/11_e4c_robustrag_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/11_e4c_robustrag_a100.sh Qwen/Qwen2.5-14B-Instruct
# Needs data/RobustRAG (git clone https://github.com/inspire-group/RobustRAG).
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
mkdir -p results/robustrag runs/e4c_robustrag
if [ -f "results/robustrag/E4b_${TAG}_seed0.json" ]; then echo "done already"; exit 0; fi
python3 scripts/run_e4c_robustrag.py --model-id "$MODEL" \
    --cache "runs/e4c_robustrag/cache_${TAG}.json" --seed 0 \
    | tee "runs/e4c_robustrag/${TAG}_seed0.log"
