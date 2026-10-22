"""Blockwise (speaker) bootstrap significance testing (Liu & Peng, 2020).

Speakers are resampled with replacement; for each resample the corpus-level error rate of each
system is recomputed over all utterances of the sampled speakers, and the difference is taken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import utterance_stats

METRIC_COLUMNS = {
    "wer": ("word_errors", "ref_words"),
    "cer": ("char_errors", "ref_chars"),
    "wer_no_tone": ("word_errors_notone", "ref_words"),
    "ter": ("tone_errors", "ref_tbus"),
}


def per_utterance(preds: pd.DataFrame) -> pd.DataFrame:
    stats = [utterance_stats(str(r), str(h) if isinstance(h, str) else "")
             for r, h in zip(preds["reference"], preds["hypothesis"])]
    out = pd.DataFrame(stats)
    out["speaker"] = preds["speaker"].astype(str).values
    out["path"] = preds["path"].astype(str).values
    return out


def _speaker_sums(u: pd.DataFrame, num: str, den: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    g = u.groupby("speaker")[[num, den]].sum()
    return g[num].to_numpy(float), g[den].to_numpy(float), list(g.index)


def system_ci(preds: pd.DataFrame, metric: str = "wer", n: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Point estimate and percentile CI (percent) of one system's metric."""
    u = per_utterance(preds)
    num, den = METRIC_COLUMNS[metric]
    e, d, _ = _speaker_sums(u, num, den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n, len(e)))
    rates = 100 * e[idx].sum(1) / np.maximum(d[idx].sum(1), 1e-9)
    lo, hi = np.percentile(rates, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"metric": metric, "point": 100 * e.sum() / max(d.sum(), 1e-9), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap(preds_a: pd.DataFrame, preds_b: pd.DataFrame, metric: str = "wer", n: int = 1000,
                     ci: float = 0.95, seed: int = 42) -> dict:
    """Delta = metric(A) - metric(B) in percentage points with a speaker-block bootstrap CI."""
    a = per_utterance(preds_a).set_index("path")
    b = per_utterance(preds_b).set_index("path")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("The two prediction sets share no utterances")
    a, b = a.loc[common].reset_index(), b.loc[common].reset_index()
    num, den = METRIC_COLUMNS[metric]
    ea, d, speakers = _speaker_sums(a, num, den)
    b = b.assign(speaker=a["speaker"].values)
    eb, _, _ = _speaker_sums(b, num, den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(speakers), size=(n, len(speakers)))
    den_s = np.maximum(d[idx].sum(1), 1e-9)
    deltas = 100 * ea[idx].sum(1) / den_s - 100 * eb[idx].sum(1) / den_s
    lo, hi = np.percentile(deltas, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    point_a, point_b = 100 * ea.sum() / max(d.sum(), 1e-9), 100 * eb.sum() / max(d.sum(), 1e-9)
    return {
        "metric": metric, "value_a": point_a, "value_b": point_b, "delta": point_a - point_b,
        "delta_mean": float(deltas.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0), "n_speakers": len(speakers), "n_utts": int(len(common)),
        "resamples": n,
    }


def significance_table(predictions: dict[str, pd.DataFrame], pairs: list[tuple[str, str]],
                       n: int = 1000, ci: float = 0.95, seed: int = 42,
                       metrics: tuple[str, ...] = ("wer", "ter")) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        if a not in predictions or b not in predictions:
            continue
        for m in metrics:
            r = paired_bootstrap(predictions[a], predictions[b], m, n, ci, seed)
            rows.append({"system_a": a, "system_b": b, "comparison": f"{a}-{b}", **r})
    return pd.DataFrame(rows)
