#!/bin/bash -l
#SBATCH -J arq_montage_putinmissing
#SBATCH -o montage_putinmissing_%j.out
#SBATCH -e montage_putinmissing_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 3-00:00:00
#
# Generates attack plans for all 126 rumour hypotheses of putinmissing.
# See 07b_generate_montage_germanwings.sh for full rationale (resume-safe,
# open Qwen2.5-7B-Instruct model, no paid API). RESUME: if this times out
# or is killed, just `sbatch` it again -- it skips completed hypotheses.
#   ls attack_plan/GPT4.1/putinmissing/result_putinmissing_h*.json | wc -l
# (target: 126)

set -uo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

OUT="$REPO/data/Lying_with_Truth/attack_plan/GPT4.1/putinmissing"
mkdir -p "$OUT"

python3 scripts/generate_montage_batch.py \
    --event putinmissing \
    --output-dir "$OUT" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --start-idx 0 \
    --end-idx 125

n_done=$(ls "$OUT"/result_putinmissing_h*.json 2>/dev/null | wc -l)
echo
echo "putinmissing: ${n_done}/126 attack plans complete."
if [ "$n_done" -lt 126 ]; then
    echo "Not finished -- re-submit this script (sbatch $0) to continue from here."
else
    echo "Complete."
fi
