#!/bin/bash -l
#SBATCH -J arq_montage_germanwings
#SBATCH -o montage_germanwings_%j.out
#SBATCH -e montage_germanwings_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 3-00:00:00
#
# Generates Writer-Editor-Director attack plans for ALL 144 rumour
# hypotheses of the germanwings-crash CoPHEME event, using the open
# Qwen2.5-7B-Instruct model (matching production_name in
# data/Lying_with_Truth/configs/settings.yaml) -- no OpenRouter/paid API
# needed for this stage.
#
# RESUME-SAFE: generate_montage_batch.py skips any hypothesis whose output
# file already exists. gpu-H200's practical ceiling is 3 days (Panther
# guide) and this job may not finish ~144 multi-round debates (est.
# 15-26 LLM calls each, so ~2,000-3,700 calls total) in one submission --
# if it times out or is killed, just `sbatch` this script again; it will
# pick up where it left off with no manual bookkeeping. Check progress with:
#   ls attack_plan/GPT4.1/germanwings-crash/result_germanwings-crash_h*.json | wc -l
# (target: 144)

set -uo pipefail   # NOT -e: a per-hypothesis failure inside the Python
                    # script is caught and logged there, not fatal to this
                    # shell; -e would still be fine since the python script
                    # itself never exits nonzero on a per-hypothesis error,
                    # but omitted deliberately so a future edit that adds a
                    # non-fatal warning elsewhere in this script doesn't
                    # accidentally kill the whole multi-day job.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

OUT="$REPO/data/Lying_with_Truth/attack_plan/GPT4.1/germanwings-crash"
mkdir -p "$OUT"

python3 scripts/generate_montage_batch.py \
    --event germanwings-crash \
    --output-dir "$OUT" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --start-idx 0 \
    --end-idx 143

n_done=$(ls "$OUT"/result_germanwings-crash_h*.json 2>/dev/null | wc -l)
echo
echo "germanwings-crash: ${n_done}/144 attack plans complete."
if [ "$n_done" -lt 144 ]; then
    echo "Not finished -- re-submit this script (sbatch $0) to continue from here."
else
    echo "Complete."
fi
