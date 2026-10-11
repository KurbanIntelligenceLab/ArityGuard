#!/bin/bash -l
#SBATCH -J arq_e1_calibrate_relfilt
#SBATCH -o e1_calibrate_relfilt_%j.out
#SBATCH -e e1_calibrate_relfilt_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 08:00:00
#
# STOP AND READ THE OUTPUT before submitting anything past this script.
#
# Runs E1 with --relevance-filter (see the development notes, 17 Sep 2026 "tau=null"
# diagnosis and data/copheme_adapter.py's RELEVANCE FILTER docstring
# section). The unfiltered 413-pool run (02_e1_calibrate_a100.sh) came back
# tau=null because random draws from a claim's full authentic pool (up to
# 1821 items) usually contain nothing relevant to that specific claim --
# confirmed via cache replay (scripts/analyze_e1_cache.py) to be the
# model correctly abstaining on irrelevant evidence, not a reasoning
# failure. This script restricts each pool to claim-relevant evidence
# before sampling, and deduplicates claims CoPHEME repeats across many
# source tweets.
#
# SCALE: only 16 of 413 pools survive both the relevance filter and the
# duplicate-claim dedup (measured 17 Sep 2026) -- a full run here is
# ~16 x 8 x 300 = ~38,400 calls per model, two orders of magnitude below
# the unfiltered run. No resume infrastructure needed at this scale; an
# 8-hour walltime is generous, not a tight fit. If n_pools measured at
# runtime differs from 16 (e.g. CoPHEME coverage or the filter itself
# changes), that is not itself an error -- just note it when reading the
# result.
#
# THIS DOES NOT REPLACE 02_e1_calibrate_a100.sh's unfiltered result --
# both exist. Whether the relevance-filtered result is the one reported in
# the paper is a decision for the paper's authors (see the development notes's
# task-design discussion, 4 candidate paths) -- this script exists so the
# option can be run once that call is made, not as an endorsement.
#
# Account/QOS confirmed via `sacctmgr show assoc user=$USER format=account,qos`
# on panther-login (8 Sep 2026): YOUR_ACCOUNT / YOUR_QOS.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27
# Deliberately no `module load cuda...` -- the pip torch wheel bundles its
# own CUDA runtime; a system module can break it (libcusparse symbol errors).

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

# IMPORTANT: e1_calibrate names its output file purely from agent.name and
# seed (E1_<agent.name>_seed<seed>.json) -- it has NO awareness of
# relevance-filtering. Writing to the same "results/" dir as
# 02_e1_calibrate_a100.sh would silently OVERWRITE the unfiltered 413-pool
# result with this ~16-pool one (or vice versa, depending on run order).
# Using a separate --outdir keeps both results permanently distinct.
OUTDIR="results/relevance_filtered"
mkdir -p "$OUTDIR" runs/e1_calibrate_relfilt

for MODEL in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"; do
    TAG=$(echo "$MODEL" | tr '/' '_')
    echo "=== E1 calibration (relevance-filtered): $MODEL ==="
    python3 scripts/run_e1.py \
        --model-id "$MODEL" \
        --data-dir data \
        --outdir "$OUTDIR" \
        --cache "runs/e1_calibrate_relfilt/cache_${TAG}.json" \
        --seed 0 \
        --t-values 1 2 3 4 5 6 7 8 \
        --groups-per-t 300 \
        --relevance-filter \
        | tee "runs/e1_calibrate_relfilt/${TAG}_seed0.log"
    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -eq 3 ]; then
        echo "FALSIFICATION condition hit for $MODEL (relevance-filtered run)." >&2
        echo "STOP -- read the result in full before proceeding." >&2
        exit 3
    elif [ "$exit_code" -ne 0 ]; then
        echo "E1 (relevance-filtered) failed for $MODEL (exit $exit_code)." >&2
        exit "$exit_code"
    fi
done

echo
echo "E1 (relevance-filtered) complete for both models."
echo "Results written to ${OUTDIR}/E1_*_seed0.json -- a SEPARATE location"
echo "from the unfiltered run's results/E1_*_seed0.json, so neither"
echo "overwrites the other."
echo "NOTE: this result used --relevance-filter, n_pools ~16 (not 413)."
echo "Compare against the unfiltered results/E1_*.json before drawing"
echo "conclusions -- read both by hand."
