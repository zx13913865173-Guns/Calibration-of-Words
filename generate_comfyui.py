"""Generate images through the ComfyUI API (ComfyUI numeric-weight syntax).

ComfyUI v0.2.2 produced the ComfyUI-numeric conditions and the non-linear
mapping variant of the main experiment. The workflow template lives in
``workflows/comfyui_sdxl_weighted.json``; this module only patches node inputs.

Start the server:
    python main.py --listen 127.0.0.1 --port 8188

Usage:
    python -m src.generation.generate_comfyui --experiment main --manifests outputs/manifests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import uuid
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def patch_workflow(workflow: dict, item: dict) -> dict:
    """Patch a ComfyUI workflow dict with the prompt, seed and sampler settings.

    Node ids follow the shipped template:
      "4"  CheckpointLoaderSimple
      "5"  EmptyLatentImage
      "6"  CLIPTextEncode (positive, weighted prompt)
      "7"  CLIPTextEncode (negative)
      "3"  KSampler
      "9"  SaveImage
    """
    wf = json.loads(json.dumps(workflow))  # deep copy
    wf["6"]["inputs"]["text"] = item["rendered_prompt"]
    wf["7"]["inputs"]["text"] = item["negative_prompt"]
    wf["5"]["inputs"]["width"] = int(item["width"])
    wf["5"]["inputs"]["height"] = int(item["height"])
    wf["3"]["inputs"]["seed"] = int(item["seed"])
    wf["3"]["inputs"]["steps"] = int(item["steps"])
    wf["3"]["inputs"]["cfg"] = float(item["cfg_scale"])
    wf["9"]["inputs"]["filename_prefix"] = item["image_id"]
    return wf


def run(experiments, manifests_dir: Path, out_dir: Path, base_url: str,
        workflow_path: Path, limit=None):
    import requests

    base = base_url.rstrip("/")
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    written = 0
    for name in experiments:
        items = [json.loads(line) for line in
                 (manifests_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        items = [it for it in items if it["syntax"] == "comfyui"]
        if limit:
            items = items[:limit]
        img_dir = out_dir / f"{name}_comfyui"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{name}_comfyui_metadata.jsonl"
        client_id = str(uuid.uuid4())
        with open(meta_path, "w", encoding="utf-8") as meta:
            for item in items:
                payload = {"prompt": patch_workflow(workflow, item), "client_id": client_id}
                response = requests.post(base + "/prompt", json=payload, timeout=60)
                response.raise_for_status()
                prompt_id = response.json()["prompt_id"]

                history = {}
                for _ in range(600):
                    history = requests.get(base + f"/history/{prompt_id}", timeout=30).json()
                    if prompt_id in history:
                        break
                    time.sleep(1)

                outputs = history.get(prompt_id, {}).get("outputs", {})
                for node_output in outputs.values():
                    for image_info in node_output.get("images", []):
                        query = urllib.parse.urlencode({
                            "filename": image_info["filename"],
                            "subfolder": image_info.get("subfolder", ""),
                            "type": image_info.get("type", "output"),
                        })
                        raw = requests.get(base + "/view?" + query, timeout=60).content
                        path = img_dir / f"{item['image_id']}.png"
                        path.write_bytes(raw)
                        meta.write(json.dumps({**item, "image_path": str(path)},
                                              ensure_ascii=False) + "\n")
                        written += 1
    print(f"done: {written} images -> {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", nargs="+", default=["main"])
    ap.add_argument("--manifests", default="outputs/manifests")
    ap.add_argument("--out", default="outputs/images")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workflow", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    base_url = args.base_url or cfg["tools"]["comfyui"]["base_url"]
    workflow_path = Path(args.workflow or (PROJECT_ROOT / cfg["tools"]["comfyui"]["workflow"]))
    run(args.experiment, Path(args.manifests), Path(args.out), base_url, workflow_path, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
