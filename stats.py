"""Statistical routines of the paper (Sections 2.7 and 2.10).

All routines are torch-free and operate on pandas DataFrames / numpy arrays:

* balanced two-way ANOVA (syntax x weight) with eta-squared
* mean +/- 95% CI with a t critical value (used for the 12-rater calibration)
* Fleiss' kappa for multi-rater agreement
* paired t-tests with Bonferroni correction (cross-tool migration)
* Pearson correlation between CLIP means and human intensity ratings
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
def two_way_anova(df: pd.DataFrame, dv: str = "clip",
                  factor_a: str = "syntax", factor_b: str = "level") -> Dict[str, Dict[str, float]]:
    """Balanced two-way ANOVA with interaction; returns F, p and eta^2 per effect.

    ``df`` must have one row per observation with the dependent variable in
    ``dv`` and the two factors in ``factor_a`` / ``factor_b``.
    """

    y = df[dv].to_numpy(dtype=float)
    a_levels = sorted(df[factor_a].unique())
    b_levels = sorted(df[factor_b].unique())
    A, B = len(a_levels), len(b_levels)
    n = len(y) // (A * B)
    grand = y.mean()

    cell = df.groupby([factor_a, factor_b])[dv].mean()
    mean_a = df.groupby(factor_a)[dv].mean()
    mean_b = df.groupby(factor_b)[dv].mean()

    ss_cells = float(n * sum((cell.loc[(a, b)] - grand) ** 2
                             for a in a_levels for b in b_levels))
    ss_a = float(B * n * sum((mean_a.loc[a] - grand) ** 2 for a in a_levels))
    ss_b = float(A * n * sum((mean_b.loc[b] - grand) ** 2 for b in b_levels))
    ss_ab = ss_cells - ss_a - ss_b

    within = df.copy()
    within["_cellmean"] = within.groupby([factor_a, factor_b])[dv].transform("mean")
    ss_within = float(((within[dv] - within["_cellmean"]) ** 2).sum())
    ss_total = ss_cells + ss_within

    df_a, df_b, df_ab = A - 1, B - 1, (A - 1) * (B - 1)
    df_within = A * B * (n - 1)

    def _effect(ss, dfe, name):
        # The paper reports PARTIAL eta^2: SS_effect / (SS_effect + SS_within).
        # Verified against the reported values (weight 0.771, interaction 0.108,
        # syntax < 0.001): 2.223/(2.223+0.660) = 0.771, 0.0791/0.7392 = 0.107.
        ms = ss / dfe
        ms_within = ss_within / df_within
        f = ms / ms_within if ms_within > 0 else float("nan")
        p = float(stats.f.sf(f, dfe, df_within)) if np.isfinite(f) else float("nan")
        eta2_partial = float(ss / (ss + ss_within)) if (ss + ss_within) else float("nan")
        return name, {"F": float(f), "df1": dfe, "df2": df_within, "p": p,
                      "eta2": eta2_partial,
                      "eta2_partial": eta2_partial,
                      "eta2_total": float(ss / ss_total) if ss_total else float("nan"),
                      "ss": float(ss)}

    out = {}
    for name, payload in (_effect(ss_a, df_a, factor_a),
                          _effect(ss_b, df_b, factor_b),
                          _effect(ss_ab, df_ab, f"{factor_a}x{factor_b}")):
        out[name] = payload
    out["_within"] = {"ss": ss_within, "df": df_within,
                      "ss_total": ss_total, "n": int(len(y)), "n_per_cell": int(n)}
    return out


# ---------------------------------------------------------------------------
# confidence intervals (12-rater calibration)
# ---------------------------------------------------------------------------
def mean_ci(values: Sequence[float], t_crit: float | None = None) -> Dict[str, float]:
    """Mean and two-sided 95% CI; ``t_crit`` defaults to t(11) = 2.201."""

    v = np.asarray(list(values), dtype=float)
    n = v.size
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if t_crit is None:
        t_crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    half = t_crit * sd / np.sqrt(n) if n else 0.0
    return {"mean": mean, "sd": sd, "n": int(n),
            "t_crit": float(t_crit), "ci_low": mean - half, "ci_high": mean + half}


def calibration_range(fit_slope: float, fit_intercept: float,
                      low_comfyui: float, high_comfyui: float,
                      slope_uncertainty: float = 0.0) -> Dict[str, float]:
    """Calibrated ComfyUI dynamic range (paper Section 3.1).

    Evaluates the ComfyUI five-point regression at the calibrated W5 anchor and
    at the native W1 anchor. ``slope_uncertainty`` reproduces the reported
    rounding convention: the value from ``slope + uncertainty`` is retained so
    the calibrated range is not understated.
    """

    def _eval(slope):
        return slope * high_comfyui + fit_intercept, slope * low_comfyui + fit_intercept

    hi, lo = _eval(fit_slope)
    hi_u, lo_u = _eval(fit_slope + slope_uncertainty)
    return {"clip_at_calibrated_W5": hi, "clip_at_native_W1": lo,
            "range_point": hi - lo,
            "range_upper": hi_u - lo_u,
            "range_reported": round(hi_u - lo_u, 3)}


def calibrated_r2(r2_native: float, dr_native: float, dr_calibrated: float) -> float:
    """Range-ratio approximation of the post-calibration R^2 (Section 4.4)."""

    return float(1.0 - (1.0 - r2_native) * (dr_native / dr_calibrated))


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------
def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for a ``(n_items, n_categories)`` count table."""

    table = np.asarray(table, dtype=float)
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa requires a constant number of raters per item")
    n = n_raters[0]
    p_j = table.sum(axis=0) / (n_items * n)
    P_i = (np.sum(table ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e))


# ---------------------------------------------------------------------------
# paired tests
# ---------------------------------------------------------------------------
def paired_ttest_bonferroni(before: Sequence[float], after: Sequence[float],
                            n_comparisons: int = 4) -> Dict[str, float]:
    """One-sided paired t-test (after > before) with Bonferroni correction."""

    b = np.asarray(list(before), dtype=float)
    a = np.asarray(list(after), dtype=float)
    t, p = stats.ttest_rel(a, b)
    p_one = p / 2 if t > 0 else 1 - p / 2
    return {"t": float(t), "p": float(p), "p_one_sided": float(p_one),
            "p_bonferroni": float(min(1.0, p_one * n_comparisons)),
            "n_comparisons": n_comparisons,
            "mean_gain": float((a - b).mean())}


def pearson(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    r, p = stats.pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"r": float(r), "p": float(p), "n": int(len(x))}


# ---------------------------------------------------------------------------
# CLI: score csv -> cell table + two-way ANOVA
# ---------------------------------------------------------------------------
def analyze_scores(scores_csv, out_dir) -> int:
    """Read a per-image score CSV and emit the paper's cell/ANOVA tables."""
    from pathlib import Path

    scores_csv = Path(scores_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    df = df.rename(columns={"clip_score": "clip"})
    experiment = str(df["experiment"].iloc[0]) if "experiment" in df.columns else "main"

    # cell means / sd, one row per (syntax, level) -- the five points per syntax
    cell = (df.groupby(["syntax", "level"])["clip"]
              .agg(n="count", mean="mean", sd="std")
              .reset_index())
    cell_path = out_dir / f"cell_means_{experiment}.csv"
    cell.to_csv(cell_path, index=False)
    print(f"wrote {cell_path}  ({len(cell)} cells)")

    anova = two_way_anova(df, dv="clip", factor_a="syntax", factor_b="level")
    rows = []
    for effect, payload in anova.items():
        if effect == "_within":
            continue
        rows.append({"effect": effect,
                     "F": payload["F"], "df1": payload["df1"], "df2": payload["df2"],
                     "p": payload["p"], "partial_eta2": payload["eta2_partial"]})
    anova_df = pd.DataFrame(rows)
    anova_path = out_dir / f"anova_{experiment}.csv"
    anova_df.to_csv(anova_path, index=False)
    print(f"wrote {anova_path}")
    print(anova_df.to_string(index=False))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="cell aggregation and two-way ANOVA")
    ap.add_argument("--scores", default="outputs/scores/clip_main.csv")
    ap.add_argument("--out", default="outputs/tables")
    args = ap.parse_args(argv)
    return analyze_scores(args.scores, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
