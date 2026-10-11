#!/usr/bin/env bash
# Fetch every external input the experiments read (`make data` runs this).
# Models are optional and not fetched here: python scripts/download/fetch_models.py
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/download/fetch_repos.sh
python scripts/download/fetch_qa_datasets.py
echo "data ready."
