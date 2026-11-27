"""Render Fig. 2 - Fig. 5 of the paper.

The functions take a results dictionary (by default ``data/reference_results.json``)
so the exact same plotting code can be pointed either at the reference values
(a smoke test that the pipeline reproduces the reported figures) or at freshly
computed results from a real run.

Usage:
    python -m src.analysis.figures --reference data/reference_results.json \
        --out outputs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Bind a CJK-capable font so that any Chinese labels render correctly.
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

SYNTAX_ORDER = ["sd_native", "comfyui", "novelai"]
COLORS = {"sd_native": "#4C72B0", "comfyui": "#DD8452", "novelai": "#55A868"}
LEVELS = ["W1", "W2", "W3", "W4", "W5"]


def fig2(ref: Dict, out: Path) -> Path:
    """Perceptual calibration and main objective results (Fig. 2A-D)."""
    cal = ref["reported"]["calibration"]
    means = ref["clip_means"]
    sds = ref["clip_sd"]
    rep = ref["reported"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    keys = ["sd_native_W3", "sd_native_W4", "sd_native_W5", "novelai_W5"]
    labels = ["SD 1-layer\n(1.10)", "SD 2-layer\n(1.21)", "SD 3-layer\n(1.33)", "NovelAI 3l\n(1.16)"]
    vals = [cal[k]["matched_comfyui"] for k in keys]
    err = [cal[k]["ci95"][1] - cal[k]["matched_comfyui"] for k in keys]
    axes[0, 0].bar(labels, vals, yerr=err, capsize=5, color="#8172B3")
    axes[0, 0].axhline(2.0, ls="--", c="grey", lw=1)
    axes[0, 0].text(0.02, 2.02, "native ComfyUI W5 = 2.0 (not matched)", fontsize=8, color="grey")
    axes[0, 0].set_title("A. Perceptual calibration (12 raters, 95% CI)")
    axes[0, 0].set_ylabel("Matched ComfyUI value")

    for s in SYNTAX_ORDER:
        y = [means[s][l] for l in LEVELS]
        e = [sds[s][l] for l in LEVELS]
        axes[0, 1].errorbar(LEVELS, y, yerr=e, marker="o", capsize=4,
                            label=s, color=COLORS[s])
    axes[0, 1].set_title("B. CLIP Score vs weight level")
    axes[0, 1].set_ylabel("CLIP Score")
    axes[0, 1].legend(fontsize=8)

    x = np.arange(len(SYNTAX_ORDER))
    w = 0.38
    r2 = [rep["r2"][s] for s in SYNTAX_ORDER]
    cr = [rep["compression_rate"][s] for s in SYNTAX_ORDER]
    axes[1, 0].bar(x - w / 2, r2, w, label="R^2", color="#4C72B0")
    axes[1, 0].bar(x + w / 2, cr, w, label="compression (1-R^2)", color="#C44E52")
    axes[1, 0].set_xticks(x, SYNTAX_ORDER)
    axes[1, 0].set_title("C. Native R^2 and compression")
    axes[1, 0].legend(fontsize=8)

    d = [rep["dino_w5_w1"][s] for s in SYNTAX_ORDER]
    axes[1, 1].bar(SYNTAX_ORDER, d, color=[COLORS[s] for s in SYNTAX_ORDER])
    axes[1, 1].set_title("D. DINOv2 distance (W5 vs W1)")
    axes[1, 1].set_ylabel("cosine distance")

    fig.tight_layout()
    path = out / "fig2_calibration_and_main.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig3(ref: Dict, out: Path) -> Path:
    """Four-level progressive ablation (Fig. 3)."""
    abl = ref["reported"]["ablation"]
    levels = ["L0", "L1", "L2", "L3", "L4"]
    vals = [abl[l] for l in levels]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(levels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_title("Fig. 3. CLIP range across ablation levels L0-L4")
    ax.set_ylabel("CLIP range")
    fig.tight_layout()
    path = out / "fig3_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4(ref: Dict, out: Path) -> Path:
    """Two-word weighting and cross-tool migration (Fig. 4A-B)."""
    mig = ref["reported"]["migration"]
    keys = ["a1111_to_comfyui", "comfyui_to_a1111", "novelai_to_comfyui", "comfyui_to_novelai"]
    labels = [k.replace("_to_", " -> ") for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(keys))
    w = 0.38
    before = [mig[k]["retention_before"] for k in keys]
    after = [mig[k]["retention_after"] for k in keys]
    axes[0].bar(x - w / 2, before, w, label="uncalibrated", color="#C44E52")
    axes[0].bar(x + w / 2, after, w, label="calibrated", color="#55A868")
    axes[0].set_xticks(x, labels, rotation=15, fontsize=8)
    axes[0].set_title("A. Intent retention")
    axes[0].legend(fontsize=8)

    db = [mig[k]["dclip_before"] for k in keys]
    da = [mig[k]["dclip_after"] for k in keys]
    axes[1].bar(x - w / 2, db, w, label="uncalibrated", color="#C44E52")
    axes[1].bar(x + w / 2, da, w, label="calibrated", color="#55A868")
    axes[1].set_xticks(x, labels, rotation=15, fontsize=8)
    axes[1].set_title("B. Delta CLIP before / after calibration")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = out / "fig4_migration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig5(ref: Dict, out: Path) -> Path:
    """Generalization and non-linear mapping validation (Fig. 5A-C)."""
    rep = ref["reported"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    cfgs = ["3.0", "7.5", "15.0"]
    for s in SYNTAX_ORDER:
        axes[0].plot(cfgs, [rep["cfg"][c][s] for c in cfgs], marker="o",
                     label=s, color=COLORS[s])
    axes[0].set_title("A. R^2 across CFG scales")
    axes[0].set_xlabel("CFG")
    axes[0].set_ylabel("R^2")
    axes[0].legend(fontsize=8)

    models = ["sd35", "flux1"]
    x = np.arange(len(models))
    w = 0.38
    axes[1].bar(x - w / 2, [rep["extended_generalization"][m]["comfyui"] for m in models],
                w, label="comfyui", color=COLORS["comfyui"])
    axes[1].bar(x + w / 2, [rep["extended_generalization"][m]["sd_native"] for m in models],
                w, label="sd_native", color=COLORS["sd_native"])
    axes[1].set_xticks(x, ["SD3.5", "Flux.1"])
    axes[1].set_title("B. R^2 on extended models")
    axes[1].legend(fontsize=8)

    maps = ["linear", "squared", "sqrt"]
    axes[2].bar(maps, [rep["nonlinear_mapping"][m] for m in maps],
                color=["#4C72B0", "#DD8452", "#55A868"])
    axes[2].set_title("C. Mapping variants on ComfyUI")
    axes[2].set_ylabel("R^2")

    fig.tight_layout()
    path = out / "fig5_generalization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_all(ref: Dict, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig2(ref, out_dir), fig3(ref, out_dir), fig4(ref, out_dir), fig5(ref, out_dir)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the paper's figures.")
    ap.add_argument("--reference", default="data/reference_results.json")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args(argv)

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    paths = render_all(ref, Path(args.out))
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
