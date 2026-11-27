#!/usr/bin/env bash
# Step 3 -- score every generated image with CLIP-L and DINOv2-large.
set -euo pipefail
cd "$(dirname "$0")/.."

for EXP in main two_word ablation migration generalization nonlinear_mapping calibration; do
  META="outputs/images/${EXP}_metadata.jsonl"
  if [[ -f "$META" ]]; then
    python3 -m src.evaluation.clip_score --metadata "$META" \
      --out "outputs/scores/clip_${EXP}.csv"
    python3 -m src.evaluation.dinov2_distance --metadata "$META" \
      --level-a W1 --level-b W5 --out "outputs/scores/dino_${EXP}.csv"
  else
    echo "skip ${EXP}: $META not found"
  fi
done

echo "step 3 complete: scores written to outputs/scores/"
