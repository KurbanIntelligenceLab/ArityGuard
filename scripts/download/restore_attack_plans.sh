#!/usr/bin/env bash
# Copy our generated montage attack plans (data/generated_attack_plans/) into the
# Lying_with_Truth clone, where data/copheme_adapter.py reads them. Run by
# fetch_repos.sh; rerun it by hand if the clone is ever refreshed.
set -euo pipefail
cd "$(dirname "$0")/../.."
dest=data/Lying_with_Truth/attack_plan/GPT4.1
[ -d data/Lying_with_Truth ] || { echo "data/Lying_with_Truth missing: run fetch_repos.sh" >&2; exit 1; }
for ev in germanwings-crash putinmissing sydneysiege; do
  mkdir -p "$dest/$ev"
  cp -p data/generated_attack_plans/"$ev"/*.json "$dest/$ev/"
  echo "  copied $(ls data/generated_attack_plans/"$ev" | wc -l) generated plans -> $dest/$ev"
done
