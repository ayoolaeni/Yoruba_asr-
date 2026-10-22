"""Leakage control: keep Common Voice dev/test out of every training and unlabelled pool."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..text import normalize_text, strip_tones


class LeakageError(AssertionError):
    pass


def leakage_key(sentence: str) -> str:
    """Normalised, tone-stripped form used to compare sentences across corpora."""
    return strip_tones(normalize_text(sentence, keep_tones=True))


def forbidden_sentences(cv_dev: pd.DataFrame, cv_test: pd.DataFrame) -> set[str]:
    return {leakage_key(s) for s in pd.concat([cv_dev["sentence"], cv_test["sentence"]])}


def remove_forbidden(df: pd.DataFrame, forbidden: set[str], column: str = "sentence") -> tuple[pd.DataFrame, int]:
    """Drop rows whose tone-stripped normalised sentence is in ``forbidden``."""
    if df.empty:
        return df, 0
    mask = df[column].map(leakage_key).isin(forbidden)
    return df[~mask].reset_index(drop=True), int(mask.sum())


def _norm_path(p: str) -> str:
    return str(Path(p).name).lower() if p else ""


def assert_no_eval_audio(pools: dict[str, pd.DataFrame], cv_dev: pd.DataFrame, cv_test: pd.DataFrame) -> dict:
    """Raise :class:`LeakageError` if a CV dev/test clip appears in a training/unlabelled pool."""
    eval_paths = set(map(str, cv_dev["path"])) | set(map(str, cv_test["path"]))
    eval_names = {_norm_path(p) for p in eval_paths}
    report = {}
    for name, df in pools.items():
        if df is None or df.empty:
            report[name] = 0
            continue
        paths = df["path"].astype(str)
        hits = paths.isin(eval_paths) | (df.get("corpus", "") == "common_voice") & paths.map(_norm_path).isin(eval_names)
        n = int(hits.sum())
        report[name] = n
        if n:
            raise LeakageError(f"{n} CV dev/test clips found in pool '{name}', e.g. {paths[hits].iloc[0]}")
    return report


def speaker_overlap(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr, dv, te = set(train["speaker"]), set(dev["speaker"]), set(test["speaker"])
    tr.discard("unknown"); dv.discard("unknown"); te.discard("unknown")
    return {"train_dev": len(tr & dv), "train_test": len(tr & te), "dev_test": len(dv & te)}
