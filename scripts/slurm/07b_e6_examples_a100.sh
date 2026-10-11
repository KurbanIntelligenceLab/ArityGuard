#!/bin/bash -l
#SBATCH -J arq_e6_examples
#SBATCH -o e6_examples_%j.out
#SBATCH -e e6_examples_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 00:30:00
#
# NEW 2 Oct 2026, companion to 07_e6_report_cpu.sh. Finds one real verbatim
# example of each of the 4 categories the development notes's E6 asks for (a firing
# certificate, a declined one, a certified-insufficient-evidence case, a
# certified-but-wrong case -- the last is flagged in the development notes's own
# text as the most important, since it is where the guarantee holds and
# the answer is still wrong). This needs a real agent and real pools,
# which needs a GPU -- the aggregation-only cost/tau parts of E6 do not
# and run separately on cpu-all (07_e6_report_cpu.sh).
#
# Uses the same 16-pool relevance-filtered CoPHEME set E3/E4/E5 already
# use, with a small G (50, not E1's 300 or E5's 1000) -- this pass only
# needs to FIND one instance of each category, not measure a rate, so a
# small scan is sufficient and cheap. Walltime kept short (30 min) since
# this is a small scan over at most 16 pools, not a full experiment sweep.
#
# Overwrites results/E6_seed0.json in place (re-reads whatever
# cost_per_method/tau_by_agent the cpu-only pass already aggregated, via
# e6_cost_transfer's own results_dir read, and adds the examples field) --
# run 07_e6_report_cpu.sh first, or after, in either order; both write to
# the same file and each pass's own aggregation is independent of the
# other's presence.

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi

# Optional model argument (added 3 Oct 2026). With no argument this
# reproduces the 7B pass and writes results/E6_seed0.json. With a model
# argument the output goes to results/e6_examples_<tag>/ so the 7B
# examples are not overwritten. Example: 14B certified 15/16 pools in E4,
# so it can supply the "declines" example 7B could not:
#   sbatch scripts/slurm/07b_e6_examples_a100.sh Qwen/Qwen2.5-14B-Instruct
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
if [ $# -ge 1 ]; then
    OUTDIR="results/e6_examples_$(echo "$MODEL" | tr '/' '_')"
else
    OUTDIR="results"
fi
mkdir -p "$OUTDIR"

python3 scripts/run_e6.py \
    --results-dir results \
    --outdir "$OUTDIR" \
    --seed 0 \
    --example-model-id "$MODEL" \
    --data-dir data \
    --example-G 50

echo
echo "E6 examples pass complete. Check $OUTDIR/E6_seed0.json's \"examples\""
echo "field -- any category still null after this means no instance of"
echo "that category was found in the 16-pool scan; widen --example-G or"
echo "the pool set if a real example is still needed."
