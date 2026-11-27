"""Unit tests for the three prompt weight syntaxes and the manifest builder.

    python tests/test_syntaxes.py          # runs without pytest
    python -m pytest tests/test_syntaxes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompt_weighting import syntaxes as S  # noqa: E402
from src.generation import manifest as M  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_native_mapping_table() -> None:
    assert S.NATIVE_MAPPING["sd_native"]["W5"][0] == 1.33
    assert S.NATIVE_MAPPING["sd_native"]["W1"][0] == 0.81
    assert S.NATIVE_MAPPING["comfyui"]["W5"][0] == 2.0
    assert S.NATIVE_MAPPING["comfyui"]["W1"][0] == 0.2
    assert S.NATIVE_MAPPING["novelai"]["W5"][0] == 1.157625


def test_roundtrip_all_cells() -> None:
    bad = [row for row in S.roundtrip_report() if not row[-1]]
    assert not bad, f"render/parse mismatch: {bad}"


def test_sd_native_parser() -> None:
    assert approx(S.parse_sd_native("fox"), 1.0)
    assert approx(S.parse_sd_native("(fox)"), 1.1)
    assert approx(S.parse_sd_native("((fox))"), 1.21)
    assert approx(S.parse_sd_native("(((fox)))"), 1.1 ** 3, 1e-9)
    assert approx(S.parse_sd_native("[fox]"), 0.9)
    assert approx(S.parse_sd_native("[[fox]]"), 0.81)


def test_novelai_parser() -> None:
    assert approx(S.parse_novelai("{fox}"), 1.05)
    assert approx(S.parse_novelai("{{{fox}}}"), 1.05 ** 3, 1e-9)
    assert approx(S.parse_novelai("[fox]"), 0.95)
    assert approx(S.parse_novelai("[[fox]]"), 0.95 ** 2, 1e-9)


def test_comfyui_parser() -> None:
    assert approx(S.parse_comfyui("(fox:2.0)"), 2.0)
    assert approx(S.parse_comfyui("(fox:0.2)"), 0.2)
    assert approx(S.parse_comfyui("((fox))"), 1.21)   # A1111-compatible fallback


def test_render_prompt_replaces_only_the_target_word() -> None:
    text = "a cat sitting on a wooden chair near a window"
    out = S.render_prompt(text, "cat", "comfyui", "W5")
    assert "(cat:2.0)" in out and "chair" in out and "(chair" not in out


def test_render_prompt_multi_two_words() -> None:
    text = "a woman holding a red umbrella on a rainy street"
    out = S.render_prompt_multi(text, ["woman", "umbrella"], "sd_native", "W4")
    assert "((woman))" in out and "((umbrella))" in out


def test_malformed_fragment_raises() -> None:
    for bad in ("(fox", "fox)", "((fox)"):
        try:
            S.parse_sd_native(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_manifest_counts_match_the_paper() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    manifests = M.build_all(cfg, ROOT / "prompts")
    exp = cfg["expected_counts"]
    for key in ("main", "two_word", "ablation", "migration",
                "generalization", "nonlinear_mapping", "calibration"):
        assert len(manifests[key]) == exp[key], (key, len(manifests[key]), exp[key])
    analysis = sum(len(manifests[k]) for k in
                   ("main", "two_word", "ablation", "migration",
                    "generalization", "nonlinear_mapping"))
    assert analysis == exp["main_analysis_total"] == 1850
    assert analysis + len(manifests["calibration"]) == exp["total_generation"] == 1890


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
