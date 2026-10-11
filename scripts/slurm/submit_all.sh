#!/bin/bash
# Queues every remaining experiment (7 Oct 2026). Run from the login node:
#   bash scripts/slurm/submit_all.sh          # gpu-A100 (scripts' defaults)
#   bash scripts/slurm/submit_all.sh h200     # gpu-H200, same jobs
# Order = priority: cheap, high-value jobs first, the long E1/E2 seed arrays
# last. Each job gets a walltime near its expected length (not the scripts'
# generous defaults) so the scheduler can backfill it into gaps. Every job
# skips work whose result file already exists and resumes from its cache,
# so cancelling and resubmitting loses nothing.
set -euo pipefail
# Not $(dirname "$0"): under `sbatch submit_all.sh` SLURM runs a spool copy of
# this file, so dirname pointed at the spool directory and the first sbatch
# failed (job 415089, 7 Oct 2026). Run it from the repo root (or export REPO).
REPO="${REPO:-$PWD}"; REPO="${REPO%/scripts/slurm}"; export REPO  # run from the repo root; jobs inherit REPO
cd "$REPO/scripts/slurm"
M7=Qwen/Qwen2.5-7B-Instruct
M14=Qwen/Qwen2.5-14B-Instruct
short() { case "$1" in *7B*) echo 7B ;; *14B*) echo 14B ;; esac; }

PART=()
if [ "${1:-a100}" = "h200" ]; then
    PART=(-p gpu-H200 -A YOUR_ACCOUNT -q YOUR_QOS --mem 100000MB)
fi
# walltime per (experiment, model): 7B first, 14B second
wt() {
    case "$1" in
        synth)     [ "$2" = 7B ] && echo 04:00:00 || echo 06:00:00 ;;
        robustrag) [ "$2" = 7B ] && echo 08:00:00 || echo 14:00:00 ;;
        e5b|e3|e4|e5) [ "$2" = 7B ] && echo 06:00:00 || echo 10:00:00 ;;
        e1)        [ "$2" = 7B ] && echo 08:00:00 || echo 1-00:00:00 ;;
        e2)        [ "$2" = 7B ] && echo 16:00:00 || echo 1-06:00:00 ;;
    esac
}

for M in $M7 $M14; do
    sbatch "${PART[@]}" -t "$(wt synth "$(short "$M")")" 10_e2_synth_a100.sh "$M"
done
for M in $M7 $M14; do
    sbatch "${PART[@]}" -t "$(wt robustrag "$(short "$M")")" 11_e4c_robustrag_a100.sh "$M"
done
for EXP in e5b e4 e3 e5 e2 e1; do
    for M in $M7 $M14; do
        S=$(short "$M")
        sbatch "${PART[@]}" -t "$(wt "$EXP" "$S")" -J "seeds_${EXP}_$S" \
            12_seeds_a100.sh "$EXP" "$M"
    done
done
echo "queued: 2 synthetic-entity jobs, 2 RobustRAG jobs, 12 seed arrays (48 tasks)"
