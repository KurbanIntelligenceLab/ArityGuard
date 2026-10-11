#!/bin/bash -l
#SBATCH -J arq_e3_curve
#SBATCH -o e3_curve_%j.out
#SBATCH -e e3_curve_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 2-00:00:00
#
# e3_curve exists (code/arq/experiments.py) but arq.run's CLI stub still
# raises on --data. Uses the same direct-call pattern as run_e1.py, via
# run_e3.py. Requires the CoPHEME adapter (task A1).
#
# WALLTIME (10 Sep 2026): n_pools grew from 3 to 413 after CoPHEME's
# full-parity generation completed (the development notes sec 4b) -- bumped from the
# original 12h to 2 days (gpu-A100's practical ceiling is 3 days). RESUME:
# run_e3.py's cache is periodic-flush/SIGTERM-safe via
# scripts/resume_agent.py, same as run_e1.py -- if this times out,
# `sbatch` it again; cached verdicts are reused, not recomputed.
#
# FIXED 29 Sep 2026: this script predates the relevance-filter fix
# (17 Sep 2026) and never passed --relevance-filter to run_e3.py -- job
# 406176 ran against the unfiltered 413-pool set (206 cal/207 eval) and
# produced a large, confusing predicted-vs-observed gap (MAE 0.58-0.76)
# driven by the same bimodal ABSTAIN-default pattern diagnosed on 17 Sep
# for unfiltered pools, not a real calibration/certificate problem. Added
# --relevance-filter below so this targets the same 16-pool
# construct-valid set where the tau=1 crossing was actually found (job
# 405638) -- see the development notes' 29 Sep entries for the full trace.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if ! ls results/E1_*_seed0.json >/dev/null 2>&1; then
    echo "No results/E1_*_seed0.json found. Run and REVIEW 02_e1_calibrate_a100.sh first." >&2
    exit 1
fi

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

mkdir -p results runs/e3_curve

for MODEL in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"; do
    TAG=$(echo "$MODEL" | tr '/' '_')
    for SPLIT_ARGS in "--split calibration" "--split evaluation" "--split evaluation --forced-worst-case"; do
        echo "=== E3 curve: $MODEL ($SPLIT_ARGS) ==="
        python3 scripts/run_e3.py \
            --model-id "$MODEL" \
            --data-dir data \
            --outdir results \
            --cache "runs/e3_curve/cache_${TAG}.json" \
            --seed 0 \
            --relevance-filter \
            $SPLIT_ARGS \
            | tee -a "runs/e3_curve/${TAG}_seed0.log"
    done
done

echo "E3 complete. Check the FALSIFICATION field in results/E3_*_seed0*.json:"
echo "observed curve below prediction, or forced-worst-case deviating at all,"
echo "indicts the analysis (Theorem 5.2), not the mechanism."
