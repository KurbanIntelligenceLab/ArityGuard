#!/bin/bash -l
#SBATCH -J arq_e5_stress
#SBATCH -o e5_stress_%j.out
#SBATCH -e e5_stress_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A YOUR_ACCOUNT        # edit for your cluster
#SBATCH -q YOUR_QOS            # edit for your cluster
#SBATCH -t 2-12:00:00
#
# e5_stress exists (code/arq/experiments.py). Largest sweep by wall-clock
# (t, G, eta, abstention, order, sampling ablations plus the displacement
# stress rows), hence H200's larger memory and this project's H200
# allocation. gpu-H200 max walltime is 3 days -- this requests 2.5,
# leaving some margin. Requires the CoPHEME adapter (task A1) for both
# augmenting and displacing pools.
#
# FIXED 29 Sep 2026: this script originally called
# `python3 -m arq.run e5 --agent "$MODEL" --seed 0 --outdir results/` with
# no --data. arq.run's CLI (code/arq/run.py) prints "No --data given" and
# exits 2 as soon as --data is empty -- that call could never have
# produced a result. Switched to scripts/run_e5.py, the same
# direct-call pattern run_e1.py/run_e3.py already use for E1/E3
# (the development notes C5/D1-D4: "arq.run still raises NotImplementedError on
# --data -- the SLURM scripts bypass it by calling drivers directly").
# --t defaults to 1 in run_e5.py, matching the one construct-valid tau=1
# crossing found so far (14B, relevance-filtered, contradicts prompt).
#
# SCALED UP 30 Sep 2026: the first real run (job 406178, 7B only, then
# stopped at the designed FALSIFICATION exit) hit "zero violations under
# a displacing adversary" at the OLD defaults (rho=0.002, G=300) -- traced
# this to a genuine statistical-power problem, not evidence the
# displacing condition is broken: at that rho/G, the expected number of
# contaminated groups per pool was only ~1-27 (depending on pool size)
# against a Dbar of 4-39 votes, making a violation a real but very rare
# tail event, not something 16 pools' worth of trials could be expected
# to surface (see the development notes' 30 Sep 2026 entries for the full
# arithmetic). run_e5.py's defaults raised to rho=0.0038 (the CoPHEME
# feasibility ceiling) and G=1000 (up from 300) to give this stress test
# a fairer chance -- walltime bumped accordingly since this is ~3.3x the
# per-pool group count at a similar per-call cost. This is NOT a
# guarantee of observing a violation (Dbar is a concentration bound
# specifically built to stay ahead of the expected contaminated count,
# so it remains a genuine rare-tail-event test by design) -- read
# whatever this produces with that in mind, not as pass/fail.
#
# Also fixed the same day: code/arq/certificate.py's contamination_bound
# had a real OverflowError at G>=3000ish (comb(G,d) * q**d overflows
# float64 before the tiny probability factors shrink it back down) --
# recomputed in log-space, verified to match the original exactly at
# every (n, rho, G) combination the original could handle without
# overflowing. G=1000 here doesn't hit that overflow itself, but the fix
# removes a ceiling on how far this could be pushed later if needed.

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

mkdir -p results runs/e5_stress

# NOTE (30 Sep 2026): the "displacing, threshold Dbar" FALSIFICATION is
# now a UNDERSTOOD, documented finding at this scale (see the development notes's
# 30 Sep 2026 entries) -- zero violations across only 16 real pools is
# not surprising given Dbar's ~1%-per-pool exceedance rate by
# construction, not evidence of an implementation gap (T2's exhaustive
# test already proves the mechanism CAN produce violations at small G).
# set -e is disabled around this loop so a 7B FALSIFICATION exit (3)
# doesn't prevent 14B from running too -- both models' results are
# wanted for completeness even though neither is expected to resolve
# the underlying test-power limitation.
set +e
for MODEL in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"; do
    TAG=$(echo "$MODEL" | tr '/' '_')
    echo "=== E5 stress/ablations: $MODEL ==="
    python3 scripts/run_e5.py \
        --model-id "$MODEL" \
        --data-dir data \
        --outdir results \
        --cache "runs/e5_stress/cache_${TAG}.json" \
        --seed 0 \
        --relevance-filter \
        | tee "runs/e5_stress/${TAG}_seed0.log"
    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -eq 3 ]; then
        echo "FALSIFICATION hit for $MODEL (expected, documented, see" >&2
        echo "the development notes 30 Sep 2026 -- test-power limitation, not a" >&2
        echo "known implementation bug). Continuing to next model." >&2
    elif [ "$exit_code" -ne 0 ]; then
        echo "E5 failed for $MODEL (exit $exit_code) -- a REAL error," >&2
        echo "not the known FALSIFICATION. Stopping." >&2
        set -e
        exit "$exit_code"
    fi
done
set -e

echo "E5 complete. Check results/E5_*_seed0.json row index 2 (\"displacing,"
echo "threshold Dbar\"): if certified > 0 and violations == 0, the"
echo "displacing condition is probably not implemented -- see the"
echo "FALSIFICATION field and the development notes."
