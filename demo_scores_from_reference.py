"""Generate a DEMO scores table from the paper's published summary values.

Purpose: let a reviewer exercise the statistics and figure pipeline (step 4)
without a GPU or the 1,890 generated images. The output is *simulated*: cell
means and SDs are taken from ``data/reference_results.json`` and per-image
values are drawn around them, so the cell table will match the paper but the
ANOVA F values will not (they depend on the between-cell signal, which the five
published means do not fully determine -- use the real images from step 3 for
that, and see scripts/verify_reference.py for the deterministic checks).

    python scripts/demo_scores_from_reference.py --out outputs/scores/clip_main.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 7
IMAGES_PER_CELL = 60


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference_results.json")
    ap.add_argument("--out", default="outputs/scores/clip_main.csv")
    args = ap.parse_args(argv)

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    rows = []
    for syntax in ("sd_native", "comfyui", "novelai"):
        for level in ("W1", "W2", "W3", "W4", "W5"):
            mean = ref["clip_means"][syntax][level]
            sd = ref["clip_sd"][syntax][level]
            for idx in range(IMAGES_PER_CELL):
                rows.append({
                    "image_id": f"main__{syntax}__{level}__P{idx + 1:02d}",
                    "experiment": "main",
                    "syntax": syntax,
                    "level": level,
                    "weight": ref["native_factors"][syntax][level],
                    "target_word": "target",
                    "clip_score": float(rng.normal(mean, sd)),
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} SIMULATED scores -> {out}")
    print("cell means will match the paper; ANOVA F values will not (see docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
