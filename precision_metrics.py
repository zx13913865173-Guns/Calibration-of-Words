"""Precision metrics of the paper (Section 2.6) and the local-slope analysis.

Definitions reproduced verbatim from the paper:

* Dynamic range (DR)   = max(CLIP) - min(CLIP)
* Linearity (R^2)      = coefficient of determination of CLIP regressed on weight
* Compression rate     = 1 - R^2
* Saturation point (SP)= first weight level whose adjacent CLIP increment < 0.015
* CTME                 = |CLIP_B - CLIP_A| + (1 - expert intent retention)

The module is deliberately torch-free so it can be unit-tested on CPU.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats

SATURATION_THRESHOLD = 0.015
LEVELS: Sequence[str] = ("W1", "W2", "W3", "W4", "W5")


# ---------------------------------------------------------------------------
# core metrics
# ---------------------------------------------------------------------------
def dynamic_range(clip_scores: Iterable[float]) -> float:
    """DR = max(CLIP) - min(CLIP)."""

    arr = np.asarray(list(clip_scores), dtype=float)
    return float(arr.max() - arr.min())


def compression_rate(r2: float) -> float:
    """Compression rate = 1 - R^2 (proportion of CLIP variance left unexplained)."""

    return float(1.0 - r2)


def fit_linear(weights: Sequence[float], clip_scores: Sequence[float]) -> Dict[str, float]:
    """Least-squares fit of CLIP on weight; returns slope, intercept, r, r2."""

    w = np.asarray(weights, dtype=float)
    c = np.asarray(clip_scores, dtype=float)
    if w.size < 2:
        raise ValueError("need at least two observations")
    slope, intercept, r, _p, _se = stats.linregress(w, c)
    return {"slope": float(slope), "intercept": float(intercept),
            "r": float(r), "r2": float(r ** 2)}


def linearity_r2(weights: Sequence[float], clip_scores: Sequence[float]) -> float:
    """R^2 of CLIP regressed on the weight values.

    Note (paper Section 2.6): R^2 is invariant under affine rescaling of the
    predictor, so min-max normalising the weights cannot change it.
    """

    return fit_linear(weights, clip_scores)["r2"]


def saturation_point(clip_means: Sequence[float],
                     levels: Sequence[str] = LEVELS,
                     threshold: float = SATURATION_THRESHOLD) -> Optional[int]:
    """First level at which the forward CLIP increment drops below ``threshold``.

    ``clip_means`` is ordered from the weakest (W1) to the strongest (W5)
    weight. Returns the 1-based level index of the plateau onset, or ``None``
    when no increment falls below the threshold.
    """

    c = np.asarray(clip_means, dtype=float)
    increments = np.diff(c)
    for i, inc in enumerate(increments):
        if inc < threshold:
            return i + 1
    return None


def local_slopes(weights: Sequence[float], clip_scores: Sequence[float],
                 window: float = 0.2) -> List[Optional[float]]:
    """Local slope of CLIP around each weight value, within a +/- ``window``."""

    w = np.asarray(weights, dtype=float)
    c = np.asarray(clip_scores, dtype=float)
    out: List[Optional[float]] = []
    for x0 in w:
        mask = (w >= x0 - window) & (w <= x0 + window)
        if mask.sum() < 2 or np.ptp(w[mask]) == 0:
            out.append(None)
        else:
            out.append(float(stats.linregress(w[mask], c[mask]).slope))
    return out


# ---------------------------------------------------------------------------
# non-linear model comparison (AIC / BIC), paper Section 2.9
# ---------------------------------------------------------------------------
def _aic_bic(rss: float, n: int, k: int) -> Dict[str, float]:
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return {"aic": float(aic), "bic": float(bic)}


def compare_nonlinear_models(weights: Sequence[float], clip_scores: Sequence[float]) -> Dict[str, Dict[str, float]]:
    """Fit linear / quadratic / logarithmic / exponential models and report AIC-BIC."""

    w = np.asarray(weights, dtype=float)
    c = np.asarray(clip_scores, dtype=float)
    n = w.size
    out: Dict[str, Dict[str, float]] = {}

    # linear: c = a + b w          (k = 2, +1 for residual variance)
    lin = np.polyfit(w, c, 1)
    rss = float(np.sum((c - np.polyval(lin, w)) ** 2))
    out["linear"] = {**_aic_bic(rss, n, 3), "rss": rss, "coef": lin.tolist()}

    # quadratic: c = a + b w + d w^2
    quad = np.polyfit(w, c, 2)
    rss = float(np.sum((c - np.polyval(quad, w)) ** 2))
    out["quadratic"] = {**_aic_bic(rss, n, 4), "rss": rss, "coef": quad.tolist()}

    # logarithmic: c = a + b ln(w)
    logx = np.log(w)
    slope, intercept, *_ = stats.linregress(logx, c)
    rss = float(np.sum((c - (intercept + slope * logx)) ** 2))
    out["logarithmic"] = {**_aic_bic(rss, n, 3), "rss": rss,
                          "coef": [intercept, slope]}

    # exponential: c = a exp(b w)  ->  ln(c) linear in w
    lc = np.log(np.clip(c, 1e-9, None))
    slope, intercept, *_ = stats.linregress(w, lc)
    pred = np.exp(intercept + slope * w)
    rss = float(np.sum((c - pred) ** 2))
    out["exponential"] = {**_aic_bic(rss, n, 3), "rss": rss,
                          "coef": [float(np.exp(intercept)), float(slope)]}
    return out


def compare_predictor_transforms(weights: Sequence[float],
                                 clip_scores: Sequence[float]) -> Dict[str, float]:
    """R^2 of CLIP regressed on the raw, squared and square-root weight axis.

    This is the comparison reported in the paper (Section 3.3): the identity
    (linear) mapping 0.92 vs squared 0.58 vs square-root 0.61.
    """

    w = np.asarray(weights, dtype=float)
    out = {"linear": linearity_r2(w, clip_scores)}
    if np.all(w > 0):
        out["squared"] = linearity_r2(w ** 2, clip_scores)
        out["sqrt"] = linearity_r2(np.sqrt(w), clip_scores)
    return out


# ---------------------------------------------------------------------------
# cross-tool migration error (paper Section 2.6)
# ---------------------------------------------------------------------------
def ctme(clip_a: float, clip_b: float, intent_retention: float) -> float:
    """CTME = |CLIP_B - CLIP_A| + (1 - expert intent retention).

    The two components are equally weighted; the paper primarily reports the
    two components separately (Delta CLIP and blind-rated retention).
    """

    return float(abs(clip_b - clip_a) + (1.0 - intent_retention))


# ---------------------------------------------------------------------------
# the min-max invariance demonstration (paper Section 2.6, limitation 8)
# ---------------------------------------------------------------------------
def minmax_invariance_demo(weights: Sequence[float], clip_scores: Sequence[float]) -> Dict[str, float]:
    """Show that min-max normalising the predictor leaves R^2 unchanged."""

    w = np.asarray(weights, dtype=float)
    r2_raw = linearity_r2(w, clip_scores)
    w_norm = (w - w.min()) / (w.max() - w.min())
    r2_norm = linearity_r2(w_norm, clip_scores)
    return {"r2_raw": r2_raw, "r2_minmax": r2_norm,
            "abs_diff": abs(r2_raw - r2_norm)}
