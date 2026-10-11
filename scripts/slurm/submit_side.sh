#!/bin/bash
# Queue all four SIDE jobs (results/side/; nothing here touches the paper).
#   bash scripts/slurm/submit_side.sh
set -euo pipefail
REPO="${REPO:-$PWD}"; REPO="${REPO%/scripts/slurm}"; export REPO  # run from the repo root; jobs inherit REPO
cd "$REPO/scripts/slurm"
THIRD=mistralai/Mistral-7B-Instruct-v0.3
sbatch 13_side_third_model_a100.sh "$THIRD"
sbatch 14_side_coalition_a100.sh Qwen/Qwen2.5-7B-Instruct
sbatch 14_side_coalition_a100.sh Qwen/Qwen2.5-14B-Instruct
# Mistral's mode (raw prompt) was probed on 8 Oct and recorded in
# runs/side/mode_*.txt, so neither Mistral job probes again.
sbatch 14_side_coalition_a100.sh "$THIRD"
squeue -u "$USER"
