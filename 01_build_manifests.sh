#!/usr/bin/env bash
# Step 1 -- digitise every prompt list into an exhaustive task manifest and
# verify the counts against the paper. Needs no GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tests/test_syntaxes.py
python3 tests/test_metrics.py
python3 scripts/verify_reference.py

python3 -m src.generation.generate_diffusers \
  --config configs/experiment.yaml \
  --prompts prompts \
  --manifests outputs/manifests \
  --dry-run --check

python3 -m src.evaluation.clip_score --self-test
python3 -m src.evaluation.dinov2_distance --self-test

echo "step 1 complete: manifests built and verified (1850 analysis images / 1890 generated)."
