"""Build the full experiment manifest of the paper.

Running this module enumerates every condition of Fig. 1 and writes one JSONL
file per experiment. It is pure Python -- no torch, no GPU -- so it doubles as
the end-to-end self-check of the pipeline orchestration:

    python -m src.generation.manifest --config configs/experiment.yaml \
        --prompts prompts --out outputs/manifests --check

Expected image counts (paper Section 2.1):
    main 900 | two_word 225 | ablation 375 | migration 80 |
    generalization 120 | nonlinear 150  ->  main-analysis total 1850
    calibration 40                      ->  total generation  1890
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_weighting import syntaxes as S  # noqa: E402

# Perceptual equivalence conversion table (paper Section 3.1, Fig. 2A).
# source_syntax -> sorted list of (source_factor, comfyui_factor) anchors.
CALIBRATION_ANCHORS: Dict[str, List[tuple]] = {
    "sd_native": [(1.10, 1.16), (1.21, 1.34), (1.33, 1.57)],
    "novelai": [(1.16, 1.22)],
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _interp(x: float, points: List[tuple]) -> float:
    """Piecewise-linear interpolation / extrapolation through anchor points."""
    pts = sorted(points)
    if len(pts) == 1:
        xs, ys = pts[0]
        return ys * (x / xs)
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i][0] <= x <= pts[i + 1][0]:
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                break
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def calibrate_factor(source_syntax: str, target_syntax: str, factor: float) -> float:
    """Convert a native weight into its perceptually calibrated target value."""
    if source_syntax == target_syntax:
        return factor
    if target_syntax == "comfyui" and source_syntax in CALIBRATION_ANCHORS:
        return round(_interp(factor, CALIBRATION_ANCHORS[source_syntax]), 4)
    if source_syntax == "comfyui":
        # invert: comfyui factor -> (source factor, comfyui factor) anchors
        if target_syntax in CALIBRATION_ANCHORS:
            inverted = [(c, s) for (s, c) in CALIBRATION_ANCHORS[target_syntax]]
            return round(_interp(factor, inverted), 4)
        # no calibration table for this pair -> fall back to the identity
        return factor
    return factor


def _item(cfg, experiment, image_id, prompt_id, prompt, target_words, syntax,
          level, factor, syntax_string, rendered_prompt, model_key="main",
          backend=None, extra=None) -> Dict:
    model = cfg["models"]["main"]["name"]
    backend = backend or cfg["models"]["main"]["backend"]
    item = {
        "image_id": image_id,
        "experiment": experiment,
        "prompt_id": prompt_id,
        # Stable identity of the depicted concept: same experiment, prompt and
        # seed -> same subject. Used to pair W1/W5 images for DINOv2 distances.
        "concept_key": f"{experiment}__{prompt_id}__s{cfg['sampling']['seed']}",
        "prompt": prompt,
        "target_words": target_words,
        "syntax": syntax,
        "level": level,
        "factor": factor,
        "syntax_string": syntax_string,
        "rendered_prompt": rendered_prompt,
        "negative_prompt": cfg["sampling"]["negative_prompt"],
        "model_key": model_key,
        "model": model,
        "backend": backend,
        "seed": cfg["sampling"]["seed"],
        "cfg_scale": cfg["sampling"]["cfg_scale"],
        "steps": cfg["sampling"]["steps"],
        "width": cfg["models"]["main"]["resolution"][0],
        "height": cfg["models"]["main"]["resolution"][1],
    }
    if extra:
        item.update(extra)
    return item


def _load(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# experiment builders
# ---------------------------------------------------------------------------
def build_main(cfg, prompts) -> List[Dict]:
    items = []
    for p in prompts["prompts"]:
        for syntax in cfg["experiments"]["main"]["syntaxes"]:
            for level in cfg["experiments"]["main"]["levels"]:
                word = p["target_word"]
                items.append(_item(
                    cfg, "main", f"main__{syntax}__{level}__{p['id']}",
                    p["id"], p["text"], [word], syntax, level,
                    S.native_factor(syntax, level), S.render(syntax, level, word),
                    S.render_prompt(p["text"], word, syntax, level),
                ))
    return items


def build_two_word(cfg, prompts) -> List[Dict]:
    items = []
    for p in prompts["prompts"]:
        for syntax in cfg["experiments"]["two_word"]["syntaxes"]:
            for level in cfg["experiments"]["two_word"]["levels"]:
                items.append(_item(
                    cfg, "two_word", f"two_word__{syntax}__{level}__{p['id']}",
                    p["id"], p["text"], list(p["target_words"]), syntax, level,
                    S.native_factor(syntax, level),
                    " | ".join(S.render(syntax, level, w) for w in p["target_words"]),
                    S.render_prompt_multi(p["text"], p["target_words"], syntax, level),
                ))
    return items


def build_ablation(cfg, prompts) -> List[Dict]:
    """Four-level progressive ablation (paper Section 2.8).

    L0 native SD syntax | L1 brackets kept, linear mapping |
    L2 wrapper removed, direct numeric weights (native SD values) |
    L3 fixed neutral template, factor 1.0 | L4 original prompt, no weighting.

    DESIGN NOTE: the paper fixes the semantics of L0-L4 but does not publish
    the literal strings; the strings below are the minimal implementation of
    those semantics.
    """
    items = []
    subset = prompts["prompts"][: cfg["experiments"]["ablation"]["prompts"]]
    for p in subset:
        word = p["target_word"]
        for level in cfg["experiments"]["ablation"]["levels"]:
            idx = S.LEVELS.index(level)
            for cond in cfg["experiments"]["ablation"]["conditions"]:
                if cond == "L0":
                    factor = S.native_factor("sd_native", level)
                    sstr = S.render("sd_native", level, word)
                    rendered = S.render_prompt(p["text"], word, "sd_native", level)
                    mapping = "native"
                elif cond == "L1":
                    factor = S.LINEAR_FACTORS[idx]
                    sstr = S.render("sd_native", level, word)
                    rendered = S.render_prompt(p["text"], word, "sd_native", level)
                    mapping = "linear"
                elif cond == "L2":
                    factor = S.native_factor("sd_native", level)
                    sstr = f"({word}:{factor})"
                    rendered = S.render_prompt(p["text"], word, "comfyui", level)
                    rendered = S.replace_word(p["text"], word, sstr)
                    mapping = "native"
                elif cond == "L3":
                    factor, sstr, mapping = 1.0, word, "neutral"
                    rendered = f"a photo of {word}"
                else:  # L4
                    factor, sstr, mapping = 1.0, word, "blank"
                    rendered = p["text"]
                items.append(_item(
                    cfg, "ablation", f"ablation__{cond}__{level}__{p['id']}",
                    p["id"], p["text"], [word], "sd_native", level, factor,
                    sstr, rendered, extra={"condition": cond, "mapping": mapping},
                ))
    return items


def build_migration(cfg, prompts) -> List[Dict]:
    """Cross-tool migration blind evaluation (paper Sections 2.9 / 3.4)."""
    items = []
    subset = prompts["prompts"][: cfg["experiments"]["migration"]["prompts"]]
    level = "W4"  # originals reused from the main experiment at W4
    for p in subset:
        word = p["target_word"]
        for src, dst in cfg["experiments"]["migration"]["directions"]:
            src_factor = S.native_factor(src, level) if src in S.SYNTAXES else 1.5
            for condition in ("uncalibrated", "calibrated"):
                if condition == "uncalibrated":
                    factor = src_factor                      # numeric substitution
                else:
                    factor = calibrate_factor(src, dst, src_factor)
                items.append(_item(
                    cfg, "migration",
                    f"migration__{src}_to_{dst}__{condition}__{p['id']}",
                    p["id"], p["text"], [word], dst, level, factor,
                    S.render(dst, level, word) if dst in S.SYNTAXES else f"({word}:{factor})",
                    S.render_prompt(p["text"], word, dst, level) if dst in S.SYNTAXES
                    else S.replace_word(p["text"], word, f"({word}:{factor})"),
                    extra={"source_tool": src, "target_tool": dst,
                           "condition": condition, "source_factor": src_factor},
                ))
    return items


def build_generalization(cfg, prompts) -> List[Dict]:
    spec = cfg["experiments"]["generalization"]
    items = []
    for p in prompts["prompts"][: spec["prompts"]]:
        word = p["target_word"]
        for model_key in spec["models"]:
            model_name = cfg["models"]["extended"][model_key]["name"]
            for syntax in spec["syntaxes"]:
                for level in spec["levels"]:
                    item = _item(
                        cfg, "generalization",
                        f"generalization__{model_key}__{syntax}__{level}__{p['id']}",
                        p["id"], p["text"], [word], syntax, level,
                        S.native_factor(syntax, level), S.render(syntax, level, word),
                        S.render_prompt(p["text"], word, syntax, level),
                        model_key=model_key,
                    )
                    item["model"] = model_name
                    items.append(item)
    return items


def build_nonlinear(cfg, prompts) -> List[Dict]:
    """Non-linear mapping validation on ComfyUI (paper Sections 2.9 / 3.5)."""
    spec = cfg["experiments"]["nonlinear_mapping"]
    items = []
    mapping_fn = {
        "linear": lambda f: f,
        "squared": lambda f: f ** 2,
        "sqrt": lambda f: f ** 0.5,
    }
    for p in prompts["prompts"][: spec["prompts"]]:
        word = p["target_word"]
        for mapping in spec["mappings"]:
            for level in spec["levels"]:
                base = S.native_factor("comfyui", level)
                factor = mapping_fn[mapping](base)
                items.append(_item(
                    cfg, "nonlinear_mapping",
                    f"nonlinear__{mapping}__{level}__{p['id']}",
                    p["id"], p["text"], [word], "comfyui", level, round(factor, 6),
                    f"({word}:{factor})", S.replace_word(p["text"], word, f"({word}:{factor})"),
                    backend="comfyui", extra={"mapping": mapping},
                ))
    return items


def build_calibration(cfg, prompts) -> List[Dict]:
    """Perceptual calibration stimuli (paper Section 2.3); excluded from analysis."""
    items = []
    refs = prompts["reference_operations"]
    for p in prompts["prompts"]:
        word = p["target_word"]
        for ref in refs:
            syntax, level = ref["syntax"], ref["level"]
            items.append(_item(
                cfg, "calibration",
                f"calibration__{ref['key']}__{p['id']}",
                p["id"], p["text"], [word], syntax, level,
                S.native_factor(syntax, level), S.render(syntax, level, word),
                S.render_prompt(p["text"], word, syntax, level),
                extra={"reference_op": ref["key"], "matched_comfyui": ref["matched_comfyui"]},
            ))
    return items


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------
def build_all(cfg: Dict, prompts_dir: Path) -> Dict[str, List[Dict]]:
    main = _load(prompts_dir / "main_prompts.json")
    two = _load(prompts_dir / "two_word_prompts.json")
    gen = _load(prompts_dir / "generalization_prompts.json")
    cal = _load(prompts_dir / "calibration_prompts.json")
    return {
        "main": build_main(cfg, main),
        "two_word": build_two_word(cfg, two),
        "ablation": build_ablation(cfg, main),
        "migration": build_migration(cfg, main),
        "generalization": build_generalization(cfg, gen),
        "nonlinear_mapping": build_nonlinear(cfg, gen),
        "calibration": build_calibration(cfg, cal),
    }


def check_counts(cfg: Dict, manifests: Dict[str, List[Dict]]) -> bool:
    exp = cfg["expected_counts"]
    ok = True
    for key, expect in exp.items():
        if key in ("main_analysis_total", "total_generation"):
            continue
        got = len(manifests[key])
        mark = "OK " if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"  [{mark}] {key:<18} expected {expect:>4}  got {got:>4}")
    analysis_total = sum(len(manifests[k]) for k in
                         ("main", "two_word", "ablation", "migration",
                          "generalization", "nonlinear_mapping"))
    generation_total = analysis_total + len(manifests["calibration"])
    for label, got, expect in (
        ("main_analysis_total", analysis_total, exp["main_analysis_total"]),
        ("total_generation", generation_total, exp["total_generation"]),
    ):
        mark = "OK " if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"  [{mark}] {label:<18} expected {expect:>4}  got {got:>4}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the experiment manifest.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--out", default="outputs/manifests")
    ap.add_argument("--check", action="store_true",
                    help="compare image counts against the paper's expected totals")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifests = build_all(cfg, Path(args.prompts))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in manifests.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(manifests)} manifest files to {out_dir}")

    if args.check:
        print("Count check against paper Section 2.1:")
        return 0 if check_counts(cfg, manifests) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
