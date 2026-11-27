"""Generate images through HF Diffusers with the controlled parser patch.

This backend runs all three syntaxes and the ablation levels on ONE identical
pipeline, which is what makes the syntax comparison confound-free (paper
Section 2.2: "the syntaxes and ablation levels were implemented through a
controlled local parser patch on the same diffusion backend").

Key point about how weighting is applied
----------------------------------------
Automatic1111 / ComfyUI / NovelAI first *parse* the syntax out of the prompt and
then tokenise the remaining plain text. This script does the same:

    plain prompt  : "a red fox standing in fresh snow"
    syntax+level  : comfyui / W5  ->  factor 2.0
    what is encoded: the plain prompt, with the 'fox' token embedding x2.0

So the weighting never touches the tokenizer input -- it is applied to the
token embedding inside the text encoder (see embedding_patch.py). Therefore
``item['prompt']`` (plain) is passed to the pipeline, and ``item['factor']`` is
the only thing taken from the syntax.

Usage
-----
    # 1) dry run: build every manifest and print/verify the counts (no GPU)
    python -m src.generation.generate_diffusers --dry-run --check

    # 2) real generation
    python -m src.generation.generate_diffusers --experiment main \
        --manifests outputs/manifests --out outputs/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import manifest as M  # noqa: E402


def _load_dense_sigmas():
    try:
        from diffusers import EulerDiscreteScheduler  # noqa: F401
    except Exception:  # pragma: no cover
        pass


def build_pipeline(model_name: str, dtype=None):
    """Load the right pipeline class for the requested model."""
    import torch
    from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

    dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    if "stable-diffusion-xl" in model_name:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=dtype,
                                                         use_safetensors=True)
    elif "stable-diffusion-3.5" in model_name or "flux" in model_name.lower():
        # SD3.5 / FLUX use their own scheduler/transformer stacks.
        pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=dtype)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def generate_one(pipe, item, device: str):
    import torch
    from src.prompt_weighting.embedding_patch import generate_weighted

    weights = {w: float(item["factor"]) for w in item["target_words"]}
    generator = torch.Generator(device=device).manual_seed(int(item["seed"]))
    common = dict(
        negative_prompt=item["negative_prompt"],
        num_inference_steps=int(item["steps"]),
        guidance_scale=float(item["cfg_scale"]),
        width=int(item["width"]),
        height=int(item["height"]),
        generator=generator,
    )
    if not weights or all(abs(v - 1.0) < 1e-9 for v in weights.values()):
        result = pipe(prompt=item["prompt"], **common)
    else:
        result = generate_weighted(pipe, item["prompt"], weights, **common)
    return result.images[0]


def run(experiments, manifests_dir: Path, out_dir: Path, limit=None, device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))

    pipe = None
    model_cache = {}
    written = 0
    for name in experiments:
        items = [json.loads(line) for line in
                 (manifests_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit:
            items = items[:limit]
        img_dir = out_dir / name
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{name}_metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as meta:
            for item in items:
                model_name = item["model"]
                if model_name not in model_cache:
                    model_cache[model_name] = build_pipeline(model_name)
                pipe = model_cache[model_name]
                image = generate_one(pipe, item, device)
                path = img_dir / f"{item['image_id']}.png"
                image.save(path)
                meta.write(json.dumps({**item, "image_path": str(path)},
                                      ensure_ascii=False) + "\n")
                written += 1
                if written % 25 == 0:
                    print(f"  {written} images written")
    print(f"done: {written} images -> {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--manifests", default="outputs/manifests")
    ap.add_argument("--out", default="outputs/images")
    ap.add_argument("--experiment", nargs="+", default=None,
                    help="one or more of main two_word ablation migration "
                         "generalization nonlinear_mapping calibration")
    ap.add_argument("--dry-run", action="store_true",
                    help="only build and verify the manifests; requires no GPU")
    ap.add_argument("--check", action="store_true",
                    help="verify image counts against the paper")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.dry_run:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        manifests = M.build_all(cfg, Path(args.prompts))
        out = Path(args.manifests)
        out.mkdir(parents=True, exist_ok=True)
        for name, items in manifests.items():
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for item in items:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[dry-run] wrote {len(manifests)} manifests to {out}")
        for name, items in manifests.items():
            print(f"  {name:<20} {len(items):>5} tasks")
        if args.check:
            ok = M.check_counts(cfg, manifests)
            return 0 if ok else 1
        return 0

    experiments = args.experiment or ["main"]
    run(experiments, Path(args.manifests), Path(args.out), limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
