#!/bin/bash -l
#SBATCH -J arq_e2_synth
#SBATCH -o e2_synth_%j.out
#SBATCH -e e2_synth_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 0-12:00:00
#
# E2 on synthetic entities: E2's d=0 arity test with every supporting entity
# renamed (data/synthetic_entities.py), seeds 0-4 in one job (a few thousand
# calls per seed). Tests whether 14B's tau=1 on HotpotQA came from memory.
#   sbatch scripts/slurm/10_e2_synth_a100.sh Qwen/Qwen2.5-7B-Instruct
#   sbatch scripts/slurm/10_e2_synth_a100.sh Qwen/Qwen2.5-14B-Instruct
# Writes results/synthetic_entities/E2_<model>_seed<s>.json; finished seeds
# are skipped on resubmission.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:?usage: sbatch $0 <hf-model-id>}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
mkdir -p results/synthetic_entities runs/e2_synth
set +e
python3 scripts/run_e2_synth.py --model-id "$MODEL" \
    --cache "runs/e2_synth/cache_${TAG}.json" --seeds 0 1 2 3 4 \
    | tee -a "runs/e2_synth/${TAG}.log"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -eq 3 ]; then
    echo "tau missed the annotated arity on some synthetic d=0 set; results written, read them." >&2
    exit 0
fi
exit "$rc"
