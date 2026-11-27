"""Generate images through the Automatic1111 WebUI API (SD-native and NovelAI syntaxes).

The main experiment rendered the SD-native and NovelAI conditions inside
Automatic1111 WebUI v1.6.0 (NovelAI via extension v1.3). Here the tool parses the
weight syntax itself, so the *rendered* prompt (with brackets) is what we send.

Start the server with the API enabled:
    ./webui.sh --api --nowebui

Usage:
    python -m src.generation.generate_a1111 --experiment main --manifests outputs/manifests
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def txt2img_payload(item: dict, sampler: str = "Euler") -> dict:
    """Translate a manifest item into an A1111 /sdapi/v1/txt2img payload."""
    return {
        "prompt": item["rendered_prompt"],
        "negative_prompt": item["negative_prompt"],
        "seed": int(item["seed"]),
        "steps": int(item["steps"]),
        "cfg_scale": float(item["cfg_scale"]),
        "width": int(item["width"]),
        "height": int(item["height"]),
        "sampler_name": sampler,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": False,
    }


def run(experiments, manifests_dir: Path, out_dir: Path, base_url: str, limit=None):
    import requests
    from PIL import Image

    endpoint = base_url.rstrip("/") + "/sdapi/v1/txt2img"
    written = 0
    for name in experiments:
        items = [json.loads(line) for line in
                 (manifests_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        # SD-native and NovelAI are the two syntaxes handled inside A1111.
        items = [it for it in items if it["syntax"] in ("sd_native", "novelai")]
        if limit:
            items = items[:limit]
        img_dir = out_dir / f"{name}_a1111"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{name}_a1111_metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as meta:
            for item in items:
                response = requests.post(endpoint, json=txt2img_payload(item), timeout=600)
                response.raise_for_status()
                payload = response.json()
                image = Image.open(io.BytesIO(base64.b64decode(payload["images"][0])))
                path = img_dir / f"{item['image_id']}.png"
                image.save(path)
                meta.write(json.dumps({**item, "image_path": str(path)},
                                      ensure_ascii=False) + "\n")
                written += 1
                time.sleep(0.1)
    print(f"done: {written} images -> {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", nargs="+", default=["main"])
    ap.add_argument("--manifests", default="outputs/manifests")
    ap.add_argument("--out", default="outputs/images")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    base_url = args.base_url or cfg["tools"]["a1111"]["base_url"]
    run(args.experiment, Path(args.manifests), Path(args.out), base_url, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
