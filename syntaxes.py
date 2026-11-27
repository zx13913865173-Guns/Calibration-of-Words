"""Prompt weight syntax parsers and native weight-level mapping.

This module implements the three prompt weight syntaxes compared in the paper
(Section 2.2):

    sd_native : Automatic1111 WebUI --  ( ) x1.1 per layer, [ ] x0.9 per layer
    comfyui   : ComfyUI v0.2.2      --  (word:n)  numeric weight n
    novelai   : NovelAI (A1111 + extension v1.3) -- { } x1.05, [ ] x0.95

Every function here is pure Python (no torch) so the whole module can be
exercised on CPU, which is what makes the weighting logic independently
verifiable without a GPU.

The parser's job is to turn a human-facing prompt fragment such as
``(((fox)))`` or ``(fox:2.0)`` into a single float -- the *effective
multiplier* applied to the target token embedding.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# --- per-layer constants (paper Section 2.2) --------------------------------
SD_EMPHASIS = 1.1
SD_SUPPRESSION = 0.9
NOVELAI_EMPHASIS = 1.05
NOVELAI_SUPPRESSION = 0.95

SYNTAXES: Tuple[str, ...] = ("sd_native", "comfyui", "novelai")
LEVELS: Tuple[str, ...] = ("W1", "W2", "W3", "W4", "W5")

# Placeholder token used inside the literal fragments below. It deliberately
# contains no bracket characters, so brace/parenthesis counting is unambiguous.
PLACEHOLDER = "@W@"

# --- native weight-level mapping (paper Section 2.2, verbatim) --------------
# syntax -> level -> (factor, literal fragment written on the tool side)
NATIVE_MAPPING: Dict[str, Dict[str, Tuple[float, str]]] = {
    "sd_native": {
        # W1/W2 use square brackets (suppression); W3/W4/W5 nested parentheses.
        "W1": (0.81, "[[@W@]]"),          # 0.9 ** 2
        "W2": (0.90, "[@W@]"),            # 0.9 ** 1
        "W3": (1.10, "(@W@)"),            # 1.1 ** 1
        "W4": (1.21, "((@W@))"),          # 1.1 ** 2
        "W5": (1.33, "(((@W@)))"),        # 1.1 ** 3
    },
    "comfyui": {
        "W1": (0.2, "(@W@:0.2)"),
        "W2": (0.5, "(@W@:0.5)"),
        "W3": (1.0, "(@W@:1.0)"),
        "W4": (1.5, "(@W@:1.5)"),
        "W5": (2.0, "(@W@:2.0)"),
    },
    "novelai": {
        "W1": (0.9025, "[[@W@]]"),         # 0.95 ** 2
        "W2": (0.95, "[@W@]"),             # 0.95 ** 1
        "W3": (1.05, "{@W@}"),             # 1.05 ** 1
        "W4": (1.1025, "{{@W@}}"),         # 1.05 ** 2
        "W5": (1.157625, "{{{@W@}}}"),     # 1.05 ** 3
    },
}

# Linear mapping used by ablation level L1 ("keep brackets, replace the native
# geometric mapping with a linear one, spanning the wide ComfyUI-like range").
LINEAR_FACTORS: List[float] = [0.2, 0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _leading(text: str, ch: str) -> int:
    n = 0
    while n < len(text) and text[n] == ch:
        n += 1
    return n


def _trailing(text: str, ch: str) -> int:
    n = 0
    while n < len(text) and text[len(text) - 1 - n] == ch:
        n += 1
    return n


def render(syntax: str, level: str, word: str) -> str:
    """Return the literal prompt fragment for ``word`` at ``level``."""
    _factor, template = NATIVE_MAPPING[syntax][level]
    return template.replace(PLACEHOLDER, word)


def native_factor(syntax: str, level: str) -> float:
    """Return the effective multiplier for ``syntax`` at ``level``."""
    return NATIVE_MAPPING[syntax][level][0]


def replace_word(text: str, word: str, replacement: str, count: int = 1) -> str:
    """Replace whole-word occurrences of ``word`` (case-insensitive)."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.IGNORECASE)
    return pattern.sub(lambda _m: replacement, text, count=count)


def render_prompt(text: str, target_word: str, syntax: str, level: str) -> str:
    """Wrap the single target word of ``text`` in the given syntax."""
    return replace_word(text, target_word, render(syntax, level, target_word))


def render_prompt_multi(text: str, target_words: List[str], syntax: str, level: str) -> str:
    """Wrap every word in ``target_words`` at the same level (two-word set)."""
    out = text
    for word in target_words:
        out = replace_word(out, word, render(syntax, level, word))
    return out


# ---------------------------------------------------------------------------
# parsers: fragment -> effective multiplier
# ---------------------------------------------------------------------------
def parse_sd_native(expr: str) -> float:
    """Parse an SD-native (Automatic1111) fragment.

    >>> parse_sd_native("(((fox)))")
    1.331
    >>> round(parse_sd_native("[[fox]]"), 6)
    0.81
    """
    e = expr.strip()
    lp, lb = _leading(e, "("), _leading(e, "[")
    rp, rb = _trailing(e, ")"), _trailing(e, "]")
    if lp and lb:
        raise ValueError(f"mixed bracket direction in {expr!r}")
    if lp != rp or lb != rb:
        raise ValueError(f"unbalanced brackets in {expr!r}")
    if lp:
        return SD_EMPHASIS ** lp
    if lb:
        return SD_SUPPRESSION ** lb
    return 1.0


def parse_novelai(expr: str) -> float:
    """Parse a NovelAI bracket fragment.

    >>> round(parse_novelai("{{{fox}}}"), 6)
    1.157625
    """
    e = expr.strip()
    lc, lb = _leading(e, "{"), _leading(e, "[")
    rc, rb = _trailing(e, "}"), _trailing(e, "]")
    if lc and lb:
        raise ValueError(f"mixed bracket direction in {expr!r}")
    if lc != rc or lb != rb:
        raise ValueError(f"unbalanced brackets in {expr!r}")
    if lc:
        return NOVELAI_EMPHASIS ** lc
    if lb:
        return NOVELAI_SUPPRESSION ** lb
    return 1.0


_COMFY_NUMERIC = re.compile(r"^\(+\s*(?P<w>.+?)\s*:\s*(?P<v>[0-9]*\.?[0-9]+)\s*\)+$")


def parse_comfyui(expr: str) -> float:
    """Parse a ComfyUI fragment.

    ``(word:1.5)`` -> 1.5.  Bare parentheses fall back to the A1111-compatible
    1.1-per-layer rule, square brackets to 0.9-per-layer.

    >>> parse_comfyui("(fox:1.5)")
    1.5
    >>> round(parse_comfyui("((fox))"), 6)
    1.21
    """
    e = expr.strip()
    m = _COMFY_NUMERIC.match(e)
    if m:
        return float(m.group("v"))
    lp, lb = _leading(e, "("), _leading(e, "[")
    rp, rb = _trailing(e, ")"), _trailing(e, "]")
    if lp and lb:
        raise ValueError(f"mixed bracket direction in {expr!r}")
    if lp != rp or lb != rb:
        raise ValueError(f"unbalanced brackets in {expr!r}")
    if lp:
        return SD_EMPHASIS ** lp
    if lb:
        return SD_SUPPRESSION ** lb
    return 1.0


_PARSERS = {
    "sd_native": parse_sd_native,
    "comfyui": parse_comfyui,
    "novelai": parse_novelai,
}


def parse(syntax: str, expr: str) -> float:
    """Dispatch to the parser of the given syntax."""
    try:
        return _PARSERS[syntax](expr)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown syntax {syntax!r}; expected one of {SYNTAXES}") from exc


# ---------------------------------------------------------------------------
# round-trip self-check (used by tests/test_syntaxes.py)
# ---------------------------------------------------------------------------
def roundtrip_report() -> List[Tuple[str, str, str, float, float, bool]]:
    """Verify render() -> parse() reproduces the native factor for every cell.

    Tolerance is 5e-3 because the paper reports SD three-layer parentheses as
    1.33 (exact value 1.1**3 = 1.331) and NovelAI three braces as 1.16
    (exact value 1.05**3 = 1.157625).
    """
    rows = []
    for syntax in SYNTAXES:
        for level in LEVELS:
            fragment = render(syntax, level, "fox")
            expected = native_factor(syntax, level)
            got = parse(syntax, fragment)
            rows.append((syntax, level, fragment, expected, got, abs(expected - got) < 5e-3))
    return rows
