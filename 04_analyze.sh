#!/usr/bin/env bash
# Step 4 -- cell aggregation, statistics and figures.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.analysis.stats \
  --scores outputs/scores/clip_main.csv \
  --out outputs/tables

python3 -m src.analysis.figures \
  --reference data/reference_results.json \
  --out outputs/figures

echo "step 4 complete: tables in outputs/tables/, figures in outputs/figures/"
