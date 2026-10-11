#!/bin/bash -l
#SBATCH -J arq_seeds
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 1-12:00:00
#SBATCH --array=1-4
#
# Seeds 1-4 for every experiment that has only seed 0, so the paper's
# "five seeds" holds beyond E4b. ONE ARRAY PER (experiment, model); each array
# task is one seed and runs in parallel with the others:
#   sbatch -J seeds_e4_7B 12_seeds_a100.sh e4 Qwen/Qwen2.5-7B-Instruct
# experiments: e1 e2 e3 e4 e5 e5b. Results land beside seed 0's, where
# `make tables` picks them up.
#
# Each task gets its own cache file (no two tasks write the same file),
# started as a copy of the seed-0 cache: groups that recur across seeds
# (every t=1 singleton, for E3/E4/E5) are then free.
# A task whose result already exists exits at once, so resubmitting is safe.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
EXP="${1:?usage: sbatch $0 <e1|e2|e3|e4|e5|e5b> <hf-model-id>}"
MODEL="${2:?usage: sbatch $0 <e1|e2|e3|e4|e5|e5b> <hf-model-id>}"
SEED="${SLURM_ARRAY_TASK_ID:?submit with --array (the script sets 1-4)}"
TAG=$(echo "$MODEL" | tr '/' '_')
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

case "$EXP" in
    e1)  RUNDIR=runs/e1_calibrate_relfilt; DONE="results/relevance_filtered/E1_${TAG}_seed${SEED}.json" ;;
    e2)  RUNDIR=runs/e2_controlled;        DONE="results/E2_${TAG}_seed${SEED}.json" ;;
    e3)  RUNDIR=runs/e3_curve;             DONE="results/E3_${TAG}_t1_seed${SEED}_forced.json" ;;
    e4)  RUNDIR=runs/e4_baselines;         DONE="results/E4_${TAG}_seed${SEED}.json" ;;
    e5)  RUNDIR=runs/e5_stress;            DONE="results/E5_${TAG}_seed${SEED}.json" ;;
    e5b) RUNDIR=runs/e4b_feed_vs_pool;     DONE="results/E5b_${TAG}_seed${SEED}.json" ;;
    *)   echo "unknown experiment $EXP" >&2; exit 2 ;;
esac
if [ -f "$DONE" ]; then echo "$DONE exists, nothing to do"; exit 0; fi
mkdir -p results results/relevance_filtered "$RUNDIR"
CACHE="$RUNDIR/cache_${TAG}_seed${SEED}.json"
if [ ! -f "$CACHE" ] && [ -f "$RUNDIR/cache_${TAG}.json" ]; then
    cp "$RUNDIR/cache_${TAG}.json" "$CACHE"
fi
LOG="$RUNDIR/${TAG}_${EXP}_seed${SEED}.log"
echo "=== $EXP seed $SEED: $MODEL (cache $CACHE) ==="

set +e
case "$EXP" in
    e1)
        python3 scripts/run_e1.py --model-id "$MODEL" --data-dir data \
            --outdir results/relevance_filtered --cache "$CACHE" --seed "$SEED" \
            --t-values 1 2 3 4 5 6 7 8 --groups-per-t 300 --relevance-filter | tee "$LOG"
        rc=${PIPESTATUS[0]} ;;
    e2)
        python3 scripts/run_e2.py --model-id "$MODEL" --hotpotqa-dir data/hotpotqa \
            --twowiki-dir data/2wikimultihopqa --outdir results --cache "$CACHE" \
            --seed "$SEED" --groups-per-t 80 --n-examples 40 --distractors 0 4 16 | tee "$LOG"
        rc=${PIPESTATUS[0]} ;;
    e3)
        rc=0
        for SPLIT_ARGS in "--split calibration" "--split evaluation" \
                          "--split evaluation --forced-worst-case"; do
            python3 scripts/run_e3.py --model-id "$MODEL" --data-dir data \
                --outdir results --cache "$CACHE" --seed "$SEED" --relevance-filter \
                $SPLIT_ARGS | tee -a "$LOG"
            r=${PIPESTATUS[0]}
            if [ "$r" -ne 0 ] && [ "$r" -ne 3 ]; then rc=$r; break; fi
        done ;;
    e4)
        python3 scripts/run_e4.py --model-id "$MODEL" --data-dir data --outdir results \
            --cache "$CACHE" --seed "$SEED" --relevance-filter | tee "$LOG"
        rc=${PIPESTATUS[0]} ;;
    e5)
        python3 scripts/run_e5.py --model-id "$MODEL" --data-dir data --outdir results \
            --cache "$CACHE" --seed "$SEED" --relevance-filter | tee "$LOG"
        rc=${PIPESTATUS[0]} ;;
    e5b)
        python3 scripts/run_e5b.py --model-id "$MODEL" --outdir results \
            --cache "$CACHE" --seed "$SEED" | tee "$LOG"
        rc=${PIPESTATUS[0]} ;;
esac
set -e
# Exit 3 is a FALSIFICATION flag the drivers raise by design (E2's d=0 rule,
# E5's displacing row); the result file is still written and is what we want.
if [ "$rc" -eq 3 ]; then
    echo "FALSIFICATION flag raised ($EXP seed $SEED); result written, read it." >&2
    exit 0
fi
exit "$rc"
