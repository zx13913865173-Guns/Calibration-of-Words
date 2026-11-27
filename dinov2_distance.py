"""DINOv2-large visual identity distance (paper Section 2.5).

The paper reports ``dino_w5_w1`` -- the DINOv2-large cosine distance between the
W1 (weakest) and W5 (strongest) images of the same concept and syntax. It
measures how far the *visual identity* of the subject drifts as the weight is
pushed from one end of the range to the other, independently of CLIP text
alignment.

Distances are computed per concept_key (identical seed and prompt), so the only
thing that differs between the two images is the weight level.

Usage
-----
    python -m src.evaluation.dinov2_distance --metadata outputs/images/main_metadata.jsonl \
        --level-a W1 --level-b W5 --out outputs/scores/dino_main.csv

    python -m src.evaluation.dinov2_distance --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DINO_MODEL = "facebook/dinov2-large"


def load_dino(model_name: str = DINO_MODEL, device: str | None = None):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor, device


def _embed_images(model, processor, device, paths, batch_size=16):
    import torch
    from PIL import Image

    feats = []
    for start in range(0, len(paths), batch_size):
        chunk = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        feats.append(cls.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_distances(metadata_path: Path, out_csv: Path, level_a="W1", level_b="W5",
                     model_name: str = DINO_MODEL):
    model, processor, device = load_dino(model_name)
    records = [json.loads(line) for line in
               metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    grouped = defaultdict(dict)
    for record in records:
        grouped[(record["syntax"], record["concept_key"])][record["level"]] = record

    rows = []
    for (syntax, concept_key), by_level in sorted(grouped.items()):
        if level_a not in by_level or level_b not in by_level:
            continue
        rec_a, rec_b = by_level[level_a], by_level[level_b]
        feats = _embed_images(model, processor, device,
                              [rec_a["image_path"], rec_b["image_path"]])
        cosine = float(np.clip(np.dot(feats[0], feats[1]), -1.0, 1.0))
        rows.append({
            "syntax": syntax,
            "concept_key": concept_key,
            "level_a": level_a,
            "level_b": level_b,
            "cosine_similarity": cosine,
            "cosine_distance": 1.0 - cosine,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} DINOv2 distances -> {out_csv}")
    return rows


def summary_by_syntax(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["syntax"]].append(row["cosine_distance"])
    return {syntax: {"n": len(values), "mean": float(np.mean(values))}
            for syntax, values in sorted(buckets.items())}


def self_test() -> int:
    """Aggregate a synthetic set of distances and check the paper ordering."""
    rng = np.random.default_rng(1)
    targets = {"sd_native": 0.24, "comfyui": 0.41, "novelai": 0.33}
    rows = []
    for syntax, target in targets.items():
        for concept in range(60):
            rows.append({"syntax": syntax, "concept_key": f"main_{concept:03d}",
                         "level_a": "W1", "level_b": "W5",
                         "cosine_similarity": 1.0 - target,
                         "cosine_distance": float(target + rng.normal(0, 0.002))})
    summary = summary_by_syntax(rows)
    assert len(summary) == 3
    order = sorted(summary, key=lambda s: summary[s]["mean"])
    assert order == ["sd_native", "novelai", "comfyui"], order
    for syntax, stat in summary.items():
        assert abs(stat["mean"] - targets[syntax]) < 0.01, (syntax, stat)
    print("dinov2_distance self-test OK: sd_native < novelai < comfyui "
          f"({summary['sd_native']['mean']:.3f} < {summary['novelai']['mean']:.3f} "
          f"< {summary['comfyui']['mean']:.3f})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--out", default="outputs/scores/dino_main.csv")
    ap.add_argument("--model", default=DINO_MODEL)
    ap.add_argument("--level-a", default="W1")
    ap.add_argument("--level-b", default="W5")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.metadata:
        ap.error("--metadata is required unless --self-test is given")
    rows = compute_distances(Path(args.metadata), Path(args.out),
                             args.level_a, args.level_b, args.model)
    for syntax, stat in summary_by_syntax(rows).items():
        print(f"  {syntax:<10} n={stat['n']:<4} mean cosine distance = {stat['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
