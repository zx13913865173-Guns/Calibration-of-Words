"""CLIP-L based semantic control score (paper Section 2.4, "CLIP Score").

For every generated image this script computes the cosine similarity between

    (a) the CLIP-L image embedding of the produced image, and
    (b) the CLIP-L text embedding of the *target concept* (the weighted word,
        optionally "a photo of <word>").

That scalar is the "semantic control precision" proxy used throughout the
paper: cell means over the 60 images per (syntax, level) cell give the five
points per syntax that feed ``src/analysis/precision_metrics.py``.

Usage
-----
    # real scoring (needs torch + open_clip / transformers)
    python -m src.evaluation.clip_score --metadata outputs/images/main_metadata.jsonl \
        --out outputs/scores/clip_main.csv

    # schema + aggregation self-test, no GPU required
    python -m src.evaluation.clip_score --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIP_MODEL = "openai/clip-vit-large-patch14"


def load_clip(model_name: str = CLIP_MODEL, device: str | None = None):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def _embed_images(model, processor, device, paths):
    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def _embed_texts(model, processor, device, texts):
    import torch

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def score_manifest(metadata_path: Path, out_csv: Path, model_name: str = CLIP_MODEL,
                   batch_size: int = 32, wrapper: str = ""):
    model, processor, device = load_clip(model_name)
    records = [json.loads(line) for line in
               metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        image_feats = _embed_images(model, processor, device, [r["image_path"] for r in chunk])
        texts = [wrapper + r["target_words"][0] for r in chunk]
        text_feats = _embed_texts(model, processor, device, texts)
        for record, text_feat, image_feat in zip(chunk, text_feats, image_feats):
            rows.append({
                "image_id": record["image_id"],
                "experiment": record["experiment"],
                "syntax": record["syntax"],
                "level": record["level"],
                "weight": record["factor"],
                "target_word": record["target_words"][0],
                "clip_score": float(np.dot(text_feat, image_feat)),
            })

    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} CLIP scores -> {out_csv}")
    return rows


def aggregate_cells(rows):
    """Group per-image CLIP scores into (syntax, level) cells -> mean / sd / n."""
    cells = {}
    for row in rows:
        cells.setdefault((row["syntax"], row["level"]), []).append(row["clip_score"])
    out = {}
    for key, values in sorted(cells.items()):
        values = np.asarray(values, dtype=float)
        out[f"{key[0]}/{key[1]}"] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return out


def self_test() -> int:
    """Verify the aggregation pipeline without torch (seeded synthetic embeddings)."""
    rng = np.random.default_rng(0)
    rows = []
    for syntax in ("sd_native", "comfyui", "novelai"):
        base = {"sd_native": 0.207, "comfyui": 0.181, "novelai": 0.196}[syntax]
        for idx, (level, delta) in enumerate(zip(("W1", "W2", "W3", "W4", "W5"),
                                                 (0.0, 0.028, 0.067, 0.091, 0.104))):
            for image in range(60):
                rows.append({
                    "image_id": f"{syntax}_{level}_{image:03d}",
                    "experiment": "main",
                    "syntax": syntax,
                    "level": level,
                    "weight": 1.0,
                    "target_word": "fox",
                    "clip_score": float(base + delta * {"sd_native": 1.0, "comfyui": 1.55,
                                                        "novelai": 1.27}[syntax]
                                        + rng.normal(0, 0.01)),
                })
    cells = aggregate_cells(rows)
    assert len(cells) == 15, cells.keys()
    for key, stat in cells.items():
        assert stat["n"] == 60, (key, stat)
    w1 = cells["comfyui/W1"]["mean"]
    w5 = cells["comfyui/W5"]["mean"]
    span = w5 - w1
    assert 0.15 < span < 0.17, span  # ~0.161 in the paper
    print(f"clip_score self-test OK: 15 cells x 60 images, comfyui span = {span:.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--out", default="outputs/scores/clip_main.csv")
    ap.add_argument("--model", default=CLIP_MODEL)
    ap.add_argument("--wrapper", default="", help='e.g. "a photo of "')
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.metadata:
        ap.error("--metadata is required unless --self-test is given")
    score_manifest(Path(args.metadata), Path(args.out), args.model, wrapper=args.wrapper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
