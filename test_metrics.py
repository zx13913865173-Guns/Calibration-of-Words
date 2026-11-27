"""Unit tests for the precision metrics and the statistics routines.

    python tests/test_metrics.py
    python -m pytest tests/test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import precision_metrics as pm  # noqa: E402
from src.analysis import stats as st  # noqa: E402

SD_MEANS = [0.207, 0.235, 0.274, 0.298, 0.311]
SD_W = [0.81, 0.90, 1.10, 1.21, 1.33]


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_dynamic_range() -> None:
    assert approx(pm.dynamic_range(SD_MEANS), 0.104, 1e-9)


def test_compression_rate() -> None:
    assert approx(pm.compression_rate(0.92), 0.08)


def test_saturation_point_sd_native() -> None:
    # increments 0.028, 0.039, 0.024, 0.013 -> plateau starts at W4
    assert pm.saturation_point(SD_MEANS) == 4


def test_saturation_point_none_when_still_rising() -> None:
    assert pm.saturation_point([0.181, 0.211, 0.275, 0.318, 0.342]) is None


def test_linearity_r2_perfect_line() -> None:
    x = [1, 2, 3, 4, 5]
    y = [2 * v + 1 for v in x]
    assert approx(pm.linearity_r2(x, y), 1.0, 1e-12)


def test_linearity_r2_is_scale_invariant() -> None:
    demo = pm.minmax_invariance_demo(SD_W, SD_MEANS)
    assert approx(demo["abs_diff"], 0.0, 1e-12)


def test_comfyui_five_point_fit() -> None:
    fit = pm.fit_linear([0.2, 0.5, 1.0, 1.5, 2.0],
                        [0.181, 0.211, 0.275, 0.318, 0.342])
    assert approx(round(fit["slope"], 4), 0.0927, 1e-4)
    assert approx(round(fit["intercept"], 4), 0.1690, 1e-4)


def test_calibrated_range_reproduces_paper() -> None:
    cr = st.calibration_range(0.0927, 0.1690, 0.2, 1.57, slope_uncertainty=0.0005)
    assert approx(round(cr["range_point"], 3), 0.127, 1e-3)
    assert approx(round(cr["range_upper"], 3), 0.128, 1e-3)


def test_post_calibration_r2() -> None:
    assert approx(round(st.calibrated_r2(0.92, 0.161, 0.128), 2), 0.90, 0.005)


def test_ctme_components() -> None:
    assert approx(pm.ctme(0.30, 0.22, 1.0), 0.08, 1e-9)
    assert approx(pm.ctme(0.30, 0.30, 0.79), 0.21, 1e-9)


def test_predictor_transforms_prefer_linear_on_a_linear_response() -> None:
    # Under an exactly linear response the identity mapping must score highest;
    # this unit-tests compare_predictor_transforms itself (the paper's
    # 0.92 vs 0.58 vs 0.61 figures are image-level values).
    w = [0.2, 0.5, 1.0, 1.5, 2.0]
    c = [0.0927 * v + 0.1690 for v in w]
    t = pm.compare_predictor_transforms(w, c)
    assert approx(t["linear"], 1.0, 1e-9), t
    assert t["linear"] > t["sqrt"] > t["squared"], t


def test_two_way_anova_reproduces_marginal_f() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for s, sm in (("sd_native", SD_MEANS), ("comfyui", [0.181, 0.211, 0.275, 0.318, 0.342]),
                  ("novelai", [0.196, 0.227, 0.274, 0.307, 0.328])):
        for i, mu in enumerate(sm):
            for _ in range(60):
                rows.append({"syntax": s, "level": f"W{i+1}", "clip": float(rng.normal(mu, 0.03))})
    out = st.two_way_anova(pd.DataFrame(rows))
    # syntax main effect must be non-significant and tiny
    assert out["syntax"]["p"] > 0.05
    assert out["syntax"]["eta2"] < 0.01
    # weight main effect must dominate (effect key is the factor name 'level')
    assert out["level"]["eta2"] > 0.6


def test_two_way_anova_df() -> None:
    rng = np.random.default_rng(11)
    rows = [{"syntax": s, "level": f"W{i+1}", "clip": float(rng.normal(0.27, 0.03))}
            for s in ("a", "b", "c") for i in range(5) for _ in range(60)]
    out = st.two_way_anova(pd.DataFrame(rows))
    assert out["_within"]["df"] == 885
    assert out["_within"]["n"] == 900


def test_mean_ci_t11() -> None:
    ci = st.mean_ci([1.57] * 12, t_crit=2.201)
    assert approx(ci["mean"], 1.57)
    ci2 = st.mean_ci(np.linspace(1.4, 1.7, 12))
    assert approx(ci2["t_crit"], 2.201, 1e-3)


def test_fleiss_kappa_perfect_agreement() -> None:
    table = np.array([[5, 0], [5, 0], [0, 5]])
    assert approx(st.fleiss_kappa(table), 1.0, 1e-9)


def test_paired_ttest_bonferroni_direction() -> None:
    before = [0.66, 0.63, 0.69, 0.65]
    after = [0.79, 0.76, 0.78, 0.74]
    out = st.paired_ttest_bonferroni(before, after, n_comparisons=4)
    assert out["mean_gain"] > 0
    assert out["p_bonferroni"] < 0.05


def _collect() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_collect())
