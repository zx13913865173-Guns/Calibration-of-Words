# The Calibration of Words — Reproduction Package

Code, prompts and workflows for

> **The Calibration of Words: A Comparative Study of Three Prompt Weight Syntaxes
> on Semantic Control Precision in Diffusion Models**
> Xin Zhang, Dmitry Galkin (Tomsk State University)

This repository is the artifact referenced by the paper's *Data and Code
Availability* statement. It regenerates every image the paper analyses,
recomputes every reported statistic, and redraws every figure.

---

## 1. What the paper compares

Three ways of writing a prompt weight, all of which resolve to "multiply the
token embedding of the target word by a factor *f*":

| Syntax      | Written as          | Level mapping (W1→W5)                            | Parsed by        |
|-------------|---------------------|--------------------------------------------------|------------------|
| SD-native   | `(word)`, `[word]`  | 0.81, 0.90, 1.10, 1.21, 1.33                     | Automatic1111    |
| ComfyUI     | `(word:f)`          | 0.2, 0.5, 1.0, 1.5, 2.0                          | ComfyUI          |
| NovelAI     | `{word}`, `[word]`  | 0.9025, 0.95, 1.05, 1.1025, 1.157625             | NovelAI ext.     |

Because the three tools tokenise and weight differently, the *nominal* factors
are not the same set of numbers. The comparison is therefore run on **one
identical pipeline** (SDXL, HF Diffusers v0.25.0) through a *controlled local
parser patch*: the syntax only decides the factor, and the factor is applied to
the token embedding inside the text encoder. That removes the backend as a
confound. Cross-tool runs inside the real Automatic1111 and ComfyUI servers are
kept as a validation condition (`migration`).

## 2. Repository layout

```
├── configs/
│   ├── experiment.yaml        # models, sampling, experiments, analysis seeds
│   └── weight_mapping.yaml    # <-- the W1..W5 -> factor tables (paper Sec. 2.2)
├── prompts/                   # every prompt list, digitised from the appendix
│   ├── main_prompts.json           (60 prompts, 6 categories x 10)
│   ├── two_word_prompts.json       (15 prompts, 2 weighted words each)
│   ├── generalization_prompts.json (10 held-out prompts)
│   └── calibration_prompts.json    (10-word x 4-operation reference grid)
├── src/
│   ├── prompt_weighting/
│   │   ├── syntaxes.py         # render/parse for the three syntaxes
│   │   └── embedding_patch.py  # token-span lookup + prompt_embeds injection
│   ├── generation/
│   │   ├── manifest.py         # prompt lists -> exhaustive task manifests
│   │   ├── generate_diffusers.py   # primary backend (all three syntaxes)
│   │   ├── generate_a1111.py       # Automatic1111 /sdapi/v1/txt2img
│   │   └── generate_comfyui.py     # ComfyUI /prompt
│   ├── evaluation/
│   │   ├── clip_score.py       # CLIP-L semantic control score
│   │   └── dinov2_distance.py  # DINOv2-large visual identity distance
│   └── analysis/
│       ├── precision_metrics.py  # dynamic range, R^2, compression, CTME
│       ├── stats.py              # two-way ANOVA, CI, kappa, paired tests
│       └── figures.py            # rebuilds the paper figures
├── workflows/
│   ├── comfyui_sdxl_weighted.json  # API-format ComfyUI graph
│   └── a1111_txt2img_payload.json  # exact A1111 request shape
├── scripts/                   # 01 build -> 02 generate -> 03 score -> 04 analyse
├── tests/                     # unit tests for the syntax layer and the metrics
├── data/reference_results.json  # every number printed in the paper
└── docs/reproduction_workflow_zh.md  # step-by-step guide (Chinese)
```

## 3. Install

```bash
git clone <this-repo>.git && cd Calibration-of-Words-Repro
conda env create -f environment.yml && conda activate calibration-of-words
# or: pip install -r requirements.txt
```

CPU-only analysis (steps 1, 3-aggregation and 4) runs on numpy/pandas/scipy and
needs no GPU; only image generation (step 2) requires CUDA.

## 4. Quick start

```bash
# Step 1 — build & verify every manifest, run the tests (no GPU needed)
bash scripts/01_build_manifests.sh

# Step 2 — generate the 1,890 images (CUDA)
bash scripts/02_generate.sh

# Step 3 — CLIP-L + DINOv2-large scoring
bash scripts/03_score.sh

# Step 4 — cell tables, ANOVA, figures
bash scripts/04_analyze.sh

# No GPU? Exercise step 4 on a simulated scores table first:
python scripts/demo_scores_from_reference.py   # writes outputs/scores/clip_main.csv
```

Step 1 alone is a full self-check: it recomputes the reported statistics from
the published summary values and asserts the image counts.

```
test_syntaxes : 9/9 passed
test_metrics  : 16/16 passed
RESULT: all reference checks passed.   (scripts/verify_reference.py)
main 900 | two_word 225 | ablation 375 | migration 80
generalization 120 | nonlinear_mapping 150 | calibration 40
main_analysis_total 1850 | total_generation 1890
```

## 5. Experiment → paper-section map

| Experiment          | Paper                  | Images | What it isolates                                    |
|---------------------|------------------------|--------|-----------------------------------------------------|
| `main`              | Sec. 3.1–3.2, Fig. 2   | 900    | the three-way syntax comparison, 5 levels x 3 syntaxes |
| `two_word`          | Sec. 3.3               | 225    | coupling when two words carry a weight              |
| `ablation`          | Sec. 3.4, Fig. 3       | 375    | which component of the syntax carries the effect (L0–L4) |
| `migration`         | Sec. 3.5, Fig. 4       | 80     | moving a prompt between real tools, before/after calibration |
| `generalization`    | Sec. 3.6, Fig. 5       | 120    | 10 prompts held out of the calibration fit          |
| `nonlinear_mapping` | Sec. 4                 | 150    | linear vs squared vs sqrt weight-to-factor mapping  |
| `calibration`       | Sec. 2.6 / 3.1         | 40     | the 12-rater perceptual calibration grid            |

Total analysed = 1,850 images; 1,890 generated (40 calibration images are
reused inside `main`).

## 6. How weighting is applied (important)

Automatic1111, ComfyUI and NovelAI all parse the brackets *out* of the prompt
and then tokenise the remaining plain text. This package does the same:

```
plain prompt   : "a red fox standing in fresh snow"
syntax + level : comfyui / W5          -> factor 2.0
encoded        : the plain prompt, with the 'fox' token embedding x 2.0
```

`embedding_patch.generate_weighted()` locates the target word's token span in
each text encoder (`CLIPTextModel` and `CLIPTextModelWithProjection` for SDXL)
and scales only that span in the `prompt_embeds` tensor returned by
`encode_prompt`. No syntax characters ever reach the tokenizer on this path —
which is exactly why the three syntaxes form a clean within-backend
comparison.

## 7. Outputs

```
outputs/
├── manifests/     <experiment>.jsonl        one row per image, fully specified
├── images/        <experiment>/<id>.png     + <experiment>_metadata.jsonl
├── scores/        clip_<experiment>.csv, dino_<experiment>.csv
├── tables/        cell_means_<exp>.csv, anova_<exp>.csv
└── figures/       fig2 … fig5 .png
```

Each `images/<experiment>_metadata.jsonl` row carries `image_id`, `concept_key`
(pairs W1/W5 images of the same subject for the DINOv2 distance), `syntax`,
`level`, `factor`, `rendered_prompt`, `seed`, `cfg_scale`, `steps`, `model`.

## 8. Reproducibility notes

* Seeds are fixed in `configs/experiment.yaml`; SDXL at 1024x1024, 30 steps,
  CFG 7.5, Euler sampler.
* The reported partial eta-squared values are reproduced exactly by
  `src/analysis/stats.py` (weight 0.771, interaction 0.108, syntax < 0.001).
* The five mean points per syntax are *not* sufficient to recover the
  image-level R^2 (0.92 / 0.61 / 0.58) and the "linear mapping wins" ordering;
  those are image-level statistics. `scripts/verify_reference.py` therefore
  treats the mapping comparison there as informational and the image-level
  checks as tolerance-based. Run step 3 to recompute them from real images.
* Exact numerical equality with the paper requires the same GPU/software stack;
  small deviations in the image-level statistics are expected.

## 9. License & citation

Code is released under the MIT License (see `LICENSE`). See `CITATION.cff` for
the citation entry.
