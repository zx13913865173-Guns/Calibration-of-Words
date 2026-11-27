"""Apply a numeric weight to a target word's token embedding.

Why this file exists
--------------------
Automatic1111, ComfyUI and NovelAI all reach the same effect through the same
mechanism: the embedding vector of the weighted token is multiplied by a
scalar before it enters the text encoder / cross-attention stack (see paper
Section 2.2 and the references therein). Diffusers deliberately exposes the
``prompt_embeds`` argument so that a caller can inject an already-scaled
embedding ("Can be used to easily tweak text inputs (prompt weighting)").

That is exactly the hook the paper calls the *controlled local parser patch on
the same diffusion backend*: by scaling token embeddings ourselves we can run
all three syntaxes, and the ablation levels L0-L2, on one identical pipeline.
This removes the confound of comparing two different desktop applications.

The module imports torch lazily so that importing it on a CPU-only machine
still works.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# locating the target word inside a tokenized prompt
# ---------------------------------------------------------------------------
def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return the start index of ``needle`` inside ``haystack`` or -1."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if list(haystack[i : i + m]) == list(needle):
            return i
    return -1


def token_span(tokenizer, prompt: str, word: str) -> List[int]:
    """Indices of the tokens that make up ``word`` in ``prompt``.

    Tries the two encodings that a word can take inside a sentence (with and
    without a leading space) and returns the first one that matches, so that
    single-token and multi-token words are both handled.
    """
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    candidates: List[List[int]] = []
    for probe in (word, " " + word):
        enc = tokenizer(probe, add_special_tokens=False)["input_ids"]
        if enc:
            candidates.append(list(enc))
    for enc in candidates:
        start = _find_subsequence(ids, enc)
        if start >= 0:
            return list(range(start, start + len(enc)))
    # Fall back to a single token id match (first hit) rather than failing hard.
    fallback = tokenizer(word, add_special_tokens=False)["input_ids"]
    if fallback:
        start = _find_subsequence(ids, [fallback[0]])
        if start >= 0:
            return [start]
    raise ValueError(f"could not locate {word!r} in the tokenized prompt")


# ---------------------------------------------------------------------------
# forward hook that scales the target rows of the token embedding output
# ---------------------------------------------------------------------------
class ScaleEmbeddingHook:
    """Multiply selected rows of a token-embedding output by a factor.

    Registered on the *token embedding* module of a text encoder. Inside the
    hook ``inputs[0]`` is the ``input_ids`` tensor and ``output`` is the
    ``(batch, seq, dim)`` embedding tensor, so the hook can read the token ids,
    locate the target span, and scale it in place.
    """

    def __init__(self, tokenizer, prompt: str, weights: Dict[str, float]):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.weights = weights
        self._spans = {
            word: token_span(tokenizer, prompt, word) for word in weights
        }

    def __call__(self, module, inputs, output):  # noqa: D401 - torch hook
        ids = inputs[0]
        # Only the first sequence in the batch carries the weighted prompt.
        for word, idxs in self._spans.items():
            factor = self.weights[word]
            for idx in idxs:
                output[0, idx, :] = output[0, idx, :] * factor
        return output

    def handles(self, ids) -> bool:  # pragma: no cover - kept for clarity
        return True


@contextmanager
def weighted_prompt(pipe, tokenizer_encoder_pairs: Iterable[Tuple[object, object]],
                    prompt: str, weights: Dict[str, float]):
    """Context manager that scales ``weights`` on every encoder of a pipeline.

    ``tokenizer_encoder_pairs`` is a list of ``(tokenizer, text_encoder)``
    tuples -- one for SD1.x/SD2.x, two for SDXL. The hooks are always removed
    again, even if generation raises.
    """
    handles = []
    for tokenizer, encoder in tokenizer_encoder_pairs:
        embedding_module = encoder.get_input_embeddings()
        hook = ScaleEmbeddingHook(tokenizer, prompt, weights)
        handle = embedding_module.register_forward_hook(hook)
        handles.append(handle)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def encoder_pairs(pipe):
    """Return the ``(tokenizer, text_encoder)`` pairs of a diffusers pipeline."""
    pairs = [(pipe.tokenizer, pipe.text_encoder)]
    if getattr(pipe, "text_encoder_2", None) is not None:
        pairs.append((pipe.tokenizer_2, pipe.text_encoder_2))
    return pairs


# ---------------------------------------------------------------------------
# one-call generation helper used by src/generation/generate_diffusers.py
# ---------------------------------------------------------------------------
def generate_weighted(pipe, prompt: str, weights: Dict[str, float], **kwargs):
    """Run ``pipe`` while scaling ``weights`` on the target tokens of ``prompt``.

    Every remaining keyword argument is forwarded to the pipeline call
    (seed through ``generator``, ``cfg_scale``, ``num_inference_steps``, ...).
    """
    pairs = encoder_pairs(pipe)
    with weighted_prompt(pipe, pairs, prompt, weights):
        return pipe(prompt=prompt, **kwargs)
