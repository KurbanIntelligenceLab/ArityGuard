#!/bin/bash -l
#SBATCH -J arq_e6_report
#SBATCH -o e6_report_%j.out
#SBATCH -e e6_report_%j.err
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 8000MB
#SBATCH -t 00:30:00
#
# REWRITTEN 2 Oct 2026. The prior version of this script called
# `python3 -m arq.run e6 --agent all ...` -- arq.run's CLI has no e6
# dispatch at all (code/arq/run.py only imports e1_calibrate/e3_curve/
# e5_stress; `e6_cost_transfer` did not exist until today), so that
# invocation could never have worked, the same class of gap run_e1.py/
# run_e5.py/run_e4.py were written to fix for E1/E5/E4.
#
# E6's own scope turned out narrower than this script's old header
# implied: it is an AGGREGATION step over results/E4_*.json (the only
# experiment that already records a real per-method call count) and
# results/E1_*.json (for tau transfer across agents) -- NOT a fresh
# model-calling experiment of its own for the cost/transfer parts. No
# existing experiment tracks tokens or latency per call, and this driver
# does not fabricate either (see code/arq/experiments.py's
# e6_cost_transfer docstring for the full reasoning) -- the paper's own
# cost table must mark those fields \RES until a deliberate
# instrumentation change is made, not silently defaulted here.
#
# This CPU-only pass runs with --skip-examples: finding real verbatim
# examples (a firing certificate, a declined one, a certified-
# insufficient-evidence case, a certified-but-wrong case) needs a real
# agent and real pools, which needs a GPU -- see
# 07b_e6_examples_a100.sh for that separate, optional pass. Running
# aggregation-only here first is cheap and gives the cost/tau tables
# immediately once E1/E4 exist, without waiting on GPU availability.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

mkdir -p results

python3 scripts/run_e6.py \
    --results-dir results \
    --outdir results \
    --seed 0 \
    --skip-examples

echo "=== make tables ==="
make tables

echo
echo "E6 (aggregation-only) complete. Check results/E6_seed0.json:"
echo "cost_per_method is null until results/E4_*_seed*.json exists;"
echo "tau_by_agent is empty until results/E1_*_seed*.json exists."
echo "For verbatim examples, run 07b_e6_examples_a100.sh separately (needs"
echo "a GPU). Then run make tables to rebuild the tables."
