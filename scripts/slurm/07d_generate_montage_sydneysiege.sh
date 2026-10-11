#!/bin/bash -l
#SBATCH -J arq_montage_sydneysiege
#SBATCH -o montage_sydneysiege_%j.out
#SBATCH -e montage_sydneysiege_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 3-00:00:00
#
# Generates attack plans for all 140 rumour hypotheses of sydneysiege.
# See 07b_generate_montage_germanwings.sh for full rationale (resume-safe,
# open Qwen2.5-7B-Instruct model, no paid API). RESUME: if this times out
# or is killed, just `sbatch` it again -- it skips completed hypotheses.
#   ls attack_plan/GPT4.1/sydneysiege/result_sydneysiege_h*.json | wc -l
# (target: 140)
#
# KNOWN RISK (checked 9 Sep 2026): sydneysiege has 1081 evidence items
# (~128K chars, ~30-40K tokens once formatted into the selection prompt) --
# close to or possibly over Qwen2.5-7B-Instruct's 32768-token context
# window. Writer._llm_select_evidence embeds the FULL pool with no
# truncation on every narrative round. If this overflows context, the
# upstream code catches the exception and falls back to engagement-ranked
# selection (no crash, no extra LLM call) -- so expect this event's plans
# to lean more on that fallback path than germanwings-crash/putinmissing,
# which is worth noting when comparing plan quality across events later.

set -uo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

OUT="$REPO/data/Lying_with_Truth/attack_plan/GPT4.1/sydneysiege"
mkdir -p "$OUT"

python3 scripts/generate_montage_batch.py \
    --event sydneysiege \
    --output-dir "$OUT" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --start-idx 0 \
    --end-idx 139

n_done=$(ls "$OUT"/result_sydneysiege_h*.json 2>/dev/null | wc -l)
echo
echo "sydneysiege: ${n_done}/140 attack plans complete."
if [ "$n_done" -lt 140 ]; then
    echo "Not finished -- re-submit this script (sbatch $0) to continue from here."
else
    echo "Complete."
fi
