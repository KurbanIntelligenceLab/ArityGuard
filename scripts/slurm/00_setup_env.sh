#!/bin/bash
# Run this INTERACTIVELY on a shared server (10.4.64.27 or 10.4.64.28),
# NOT on panther-login (which kills stray long-running processes) and
# NOT via sbatch. One-time environment setup.
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"; REPO="${REPO%/scripts/slurm}"  # submit from the repo root (or scripts/slurm/); or export REPO
ENV_NAME="arq_aamas27"

if [ ! -d "$HOME/miniconda3" ]; then
    echo "miniconda3 not found at \$HOME/miniconda3 -- install it first, then re-run."
    exit 1
fi

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists. Activating and updating requirements only."
    conda activate "${ENV_NAME}"
else
    conda create -y -n "${ENV_NAME}" python=3.11
    conda activate "${ENV_NAME}"
fi

cd "$REPO/code"
pip install -r requirements.txt

python - <<'EOF'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
EOF

echo
echo "Setup done. Environment: ${ENV_NAME}"
echo "Next: confirm your account/QOS with"
echo "  module load shared slurm && sacctmgr show assoc user=\$USER format=account,qos"
echo "and update the placeholders in scripts/*.sh before submitting anything."
