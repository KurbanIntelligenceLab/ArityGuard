#!/usr/bin/env bash
# Fetch the two upstream repositories the experiments read, at the commits the
# paper used, and put our generated montage attack plans back into place.
#
#   bash scripts/download/fetch_repos.sh        # from the repo root
#
#   data/Lying_with_Truth/  CoPHEME + montage attack (Hu et al., arXiv:2601.01685)
#                           @ 21183d3. It ships attack plans for charliehebdo,
#                           ferguson and ottawshooting only. We generated plans for
#                           germanwings-crash, putinmissing and sydneysiege with an
#                           open model (scripts/generate_montage_batch.py); they
#                           live in data/generated_attack_plans/ and are copied into
#                           attack_plan/GPT4.1/ here, which is where
#                           data/copheme_adapter.py looks for them.
#   data/RobustRAG/         RobustRAG (Xiang et al.) @ 9bc35b2; we read
#                           data/realtimeqa.json and data/open_nq.json.
#
# Safe to rerun: an existing clone is checked out at the pinned commit instead of
# being cloned again.
set -euo pipefail
cd "$(dirname "$0")/../.."

fetch() {  # url dir commit
  local url="$1" dir="$2" commit="$3"
  if [ -d "$dir/.git" ]; then
    echo "present: $dir; checking out $commit"
    git -C "$dir" fetch --quiet origin || true
  else
    echo "cloning $url -> $dir"
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
  echo "  $dir at $(git -C "$dir" rev-parse --short HEAD)"
}

fetch https://github.com/CharlesJW222/Lying_with_Truth.git data/Lying_with_Truth \
      21183d3ce578d7d4bc05c51e29857e9e97237bfa
fetch https://github.com/inspire-group/RobustRAG.git data/RobustRAG \
      9bc35b2fa5fa7d1088383b2789ec19a512d316b1

bash scripts/download/restore_attack_plans.sh
