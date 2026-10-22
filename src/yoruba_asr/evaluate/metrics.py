"""ASR metrics. All public functions return PERCENTAGES (0-100).

* ``wer``/``cer``: jiwer, on tone-preserved normalised text (corpus level).
* ``wer_no_tone``: WER after removing tone marks on both sides (sub-dots kept).
* ``tone_error_rate`` (TER): edit distance between tone-label sequences / reference TBUs x 100.
* ``ter_matched_words``: tone errors only inside word pairs whose tone-stripped forms are equal.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..text import TONE_LABELS, strip_tones, tone_sequence, word_tone_pairs


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance (substitutions + deletions + insertions) between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _clean(refs: Sequence[str], hyps: Sequence[str]) -> tuple[list[str], list[str]]:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    pairs = [(str(r).strip(), str(h).strip()) for r, h in zip(refs, hyps) if str(r).strip()]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    refs, hyps = _clean(refs, hyps)
    if not refs:
        return float("nan")
    try:
        import jiwer

        return 100.0 * float(jiwer.wer(refs, hyps))
    except Exception:  # jiwer edge cases (e.g. empty hypotheses in old versions)
        e = sum(levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps))
        return 100.0 * e / sum(len(r.split()) for r in refs)


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    refs, hyps = _clean(refs, hyps)
    if not refs:
        return float("nan")
    try:
        import jiwer

        return 100.0 * float(jiwer.cer(refs, hyps))
    except Exception:
        e = sum(levenshtein(list(r), list(h)) for r, h in zip(refs, hyps))
        return 100.0 * e / sum(len(r) for r in refs)


def wer_no_tone(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return wer([strip_tones(r) for r in refs], [strip_tones(h) for h in hyps])


def tone_edits(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, reference TBUs) for one utterance."""
    r, h = tone_sequence(ref), tone_sequence(hyp)
    return levenshtein(r, h), len(r)


def tone_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float:
    refs, hyps = _clean(refs, hyps)
    edits = total = 0
    for r, h in zip(refs, hyps):
        e, n = tone_edits(r, h)
        edits += e
        total += n
    return 100.0 * edits / total if total else float("nan")


def word_alignment(ref: str, hyp: str) -> list[tuple[str, str | None, str | None]]:
    """Word alignment as ``(op, ref_word, hyp_word)`` with op in equal/substitute/delete/insert."""
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("equal" if r[i - 1] == h[j - 1] else "substitute", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("delete", r[i - 1], None))
            i -= 1
        else:
            ops.append(("insert", None, h[j - 1]))
            j -= 1
    return ops[::-1]


def ter_matched_words(refs: Sequence[str], hyps: Sequence[str]) -> tuple[float, list[list[int]]]:
    """Pure tone error rate on aligned word pairs with equal tone-stripped forms, plus the 3x3
    H/M/L confusion matrix (rows = reference, columns = hypothesis)."""
    refs, hyps = _clean(refs, hyps)
    idx = {lab: i for i, lab in enumerate(TONE_LABELS)}
    conf = np.zeros((3, 3), dtype=int)
    errors = total = 0
    for r, h in zip(refs, hyps):
        for op, rw, hw in word_alignment(r, h):
            if op in ("equal", "substitute") and strip_tones(rw) == strip_tones(hw):
                for rl, hl in word_tone_pairs(rw, hw):
                    conf[idx[rl], idx[hl]] += 1
                    total += 1
                    errors += rl != hl
    rate = 100.0 * errors / total if total else float("nan")
    return rate, conf.tolist()


def all_metrics(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    ter_mw, confusion = ter_matched_words(refs, hyps)
    return {
        "wer": wer(refs, hyps), "cer": cer(refs, hyps), "wer_no_tone": wer_no_tone(refs, hyps),
        "ter": tone_error_rate(refs, hyps), "ter_matched_words": ter_mw, "tone_confusion": confusion,
    }


def utterance_stats(ref: str, hyp: str) -> dict:
    """Per-utterance error counts used by the bootstrap and the error analysis."""
    rw, hw = ref.split(), hyp.split()
    e_ter, n_tbu = tone_edits(ref, hyp)
    return {
        "word_errors": levenshtein(rw, hw), "ref_words": len(rw),
        "char_errors": levenshtein(list(ref), list(hyp)), "ref_chars": len(ref),
        "word_errors_notone": levenshtein(strip_tones(ref).split(), strip_tones(hyp).split()),
        "tone_errors": e_ter, "ref_tbus": n_tbu,
    }


def make_compute_metrics(tokenizer):
    """``compute_metrics`` for the HF Trainer: dev ``wer``, ``cer``, ``wer_no_tone`` as FRACTIONS
    (as in the original notebook). Predictions are argmax ids (see ``preprocess_logits``)."""

    def compute_metrics(pred):
        ids = np.asarray(pred.predictions)
        if ids.ndim == 3:  # logits were not pre-reduced
            ids = ids.argmax(-1)
        ids = np.where(ids < 0, tokenizer.pad_token_id, ids)
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        hyp = tokenizer.batch_decode(ids)
        ref = tokenizer.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer(ref, hyp) / 100, "cer": cer(ref, hyp) / 100, "wer_no_tone": wer_no_tone(ref, hyp) / 100}

    return compute_metrics


def preprocess_logits(logits, labels):
    """Reduce logits to argmax ids on the GPU (keeps evaluation memory small)."""
    import torch

    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)
