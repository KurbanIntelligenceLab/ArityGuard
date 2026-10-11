#!/bin/bash -l
#SBATCH -J arq_side_3rd
#SBATCH -o side_third_%j.out
#SBATCH -e side_third_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-00:00:00
#
# SIDE EXPERIMENT (results/side/third_model/; not in the paper). A third
# model family on the paper's experiments, same prompt and protocol:
#   1. feed against pool (E4b), seeds 0-4      ~1.5 h per seed for a 7B model
#   2. arity validation, real names, d=0, seed 0
#   3. arity validation, invented names, d=0, seeds 0-4
#   sbatch scripts/slurm/13_side_third_model_a100.sh [hf-model-id]
# Default model: mistralai/Mistral-7B-Instruct-v0.3 (ungated; Llama and
# Gemma instruct models are gated and no HF token is configured).
# Every step skips results that exist, so resubmitting continues the job.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
MODEL="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
source scripts/slurm/side_mode.sh
OUT=results/side/third_model
mkdir -p "$OUT" "$OUT/synthetic_entities" runs/side/third_model
CACHE="runs/side/third_model/cache_${CTAG}.json"
ATAG=$(echo "$MODEL${CHAT_FLAG:+::chat}" | tr '/' '_')

for SEED in 0 1 2 3 4; do
    if [ -f "$OUT/E4b_${ATAG}_seed${SEED}.json" ]; then
        echo "E4b seed $SEED done, skipping"; continue
    fi
    echo "=== feed against pool, seed $SEED: $MODEL ==="
    python3 scripts/run_e4b.py --model-id "$MODEL" $CHAT_FLAG --outdir "$OUT" \
        --cache "$CACHE" --seed "$SEED" | tee "runs/side/third_model/e4b_seed${SEED}.log"
done

# The arity drivers exit 3 when their pre-registered falsification flag
# fires; that is a result, not a failure.
if [ ! -f "$OUT/E2_${ATAG}_seed0.json" ]; then
    echo "=== arity validation, real names, d=0: $MODEL ==="
    python3 scripts/run_e2.py --model-id "$MODEL" $CHAT_FLAG --outdir "$OUT" \
        --cache "$CACHE" --distractors 0 || [ $? -eq 3 ]
fi
echo "=== arity validation, invented names, d=0: $MODEL ==="
python3 scripts/run_e2_synth.py --model-id "$MODEL" $CHAT_FLAG \
    --outdir "$OUT/synthetic_entities" --cache "$CACHE" || [ $? -eq 3 ]
echo "done"
