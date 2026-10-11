#!/bin/bash -l
#SBATCH -J arq_verify_smoke
#SBATCH -o verify_smoke_%j.out
#SBATCH -e verify_smoke_%j.err
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH -t 00:20:00

set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate arq_aamas27

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
cd "$REPO"

echo "=== make verify (22-check arithmetic/soundness suite, no model calls) ==="
make verify
echo
echo "=== make smoke (end-to-end pipeline test on MockAgent) ==="
make smoke

echo
echo "verify + smoke both passed."
