#!/usr/bin/env bash
# Step 2 -- generate every image. Needs a CUDA GPU.
#
# The main experiment was produced on one identical SDXL pipeline through the
# controlled parser patch (diffusers backend), plus cross-tool validation runs
# inside Automatic1111 WebUI v1.6.0 and ComfyUI v0.2.2.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS="main two_word ablation migration generalization nonlinear_mapping calibration"

# 2a. primary backend: HF Diffusers + controlled parser patch (all syntaxes)
python3 -m src.generation.generate_diffusers \
  --experiment $EXPERIMENTS \
  --manifests outputs/manifests \
  --out outputs/images

# 2b. cross-tool validation: Automatic1111 WebUI (SD-native + NovelAI syntaxes)
#     start it first:  ./webui.sh --api --nowebui
python3 -m src.generation.generate_a1111 \
  --experiment main \
  --manifests outputs/manifests \
  --out outputs/images

# 2c. cross-tool validation: ComfyUI (ComfyUI numeric syntax)
#     start it first:  python main.py --listen 127.0.0.1 --port 8188
python3 -m src.generation.generate_comfyui \
  --experiment main nonlinear_mapping \
  --manifests outputs/manifests \
  --out outputs/images

echo "step 2 complete: images written to outputs/images/"
