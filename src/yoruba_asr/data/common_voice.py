"""Common Voice Scripted Speech 27.0 - Yoruba (Mozilla Data Collective release)."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from . import ensure_columns

# Official reference statistics for release 27.0 (2026-09-11), used only for the audit comparison.
REFERENCE_COUNTS = {"train": 1468, "dev": 983, "test": 1096, "other": 1152, "invalidated": 269}
SPLIT_FILES = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv", "other": "other.tsv",
               "validated": "validated.tsv", "invalidated": "invalidated.tsv"}


def _gender(g: str) -> str:
    g = (g or "").lower()
    if g.startswith("female"):
        return "female"
    if g.startswith("male"):
        return "male"
    return g


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                       encoding="utf-8")


def load_split(cv_dir: str | Path, split: str) -> pd.DataFrame:
    """Load one Common Voice split. ``split`` is train/dev/test/other/validated/invalidated."""
    cv_dir = Path(cv_dir)
    tsv = cv_dir / SPLIT_FILES[split]
    if not tsv.exists():
        return ensure_columns(pd.DataFrame(columns=["path", "sentence"]))
    df = read_tsv(tsv)
    clips = cv_dir / "clips"
    out = pd.DataFrame({
        "path": [str(clips / p) for p in df["path"]],
        "sentence": df.get("sentence", ""),
        "speaker": df.get("client_id", "unknown"),
        "corpus": "common_voice",
        "split": split,
        "gender": [_gender(g) for g in df.get("gender", [""] * len(df))],
        "age": df.get("age", ""),
    })
    out = out[out["sentence"].astype(str).str.strip() != ""]
    return ensure_columns(out.reset_index(drop=True))


def index_audio_files(roots, exts=(".mp3", ".wav", ".flac", ".ogg")) -> dict[str, str]:
    """File name -> full path for every audio file under ``roots`` (first occurrence wins)."""
    import os

    index: dict[str, str] = {}
    for root in roots:
        if not root or not Path(root).exists():
            continue
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for f in files:
                if f.lower().endswith(exts) and f not in index:
                    index[f] = os.path.join(dirpath, f)
    return index


def resolve_clip_paths(df: pd.DataFrame, search_roots) -> tuple[pd.DataFrame, dict]:
    """Point ``path`` at the real clip when it is not in ``<cv>/clips/`` (Kaggle may unpack uploads into
    other folders). Returns the updated frame and a report {found_in_place, remapped, missing}."""
    exists = df["path"].map(lambda p: Path(p).exists())
    report = {"found_in_place": int(exists.sum()), "remapped": 0, "missing": 0}
    if exists.all():
        return df, report
    index = index_audio_files(search_roots)
    df = df.copy()
    names = df.loc[~exists, "path"].map(lambda p: Path(p).name)
    hits = names.map(index.get)
    df.loc[hits.index[hits.notna()], "path"] = hits[hits.notna()]
    report["remapped"] = int(hits.notna().sum())
    report["missing"] = int(hits.isna().sum())
    report["audio_files_found_under_input"] = len(index)
    return df, report


def dataset_tree_report(root, max_lines: int = 40) -> str:
    """Short listing of folders under ``root`` with file counts (for error messages)."""
    import os

    lines = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        if not files:
            continue
        audio = sum(f.lower().endswith((".mp3", ".wav", ".flac")) for f in files)
        other = [f for f in files if not f.lower().endswith((".mp3", ".wav", ".flac"))][:6]
        lines.append(f"  {dirpath}  | files: {len(files)} | audio: {audio} | e.g. {other}")
        if len(lines) >= max_lines:
            lines.append("  ...")
            break
    return "\n".join(lines) or f"  (nothing found under {root})"


def load_common_voice(cv_dir: str | Path, splits=("train", "dev", "test", "other")) -> pd.DataFrame:
    return pd.concat([load_split(cv_dir, s) for s in splits], ignore_index=True)


def load_sentence_pool(cv_dir: str | Path, search_roots=None) -> pd.DataFrame:
    """Text-only sentences for TTS: ``validated_sentences.tsv`` from the CV folder, or found anywhere
    under ``search_roots`` (Kaggle may unpack files into other folders). Falls back to the sentences of
    ``validated.tsv``. The ``pool_source`` attribute of the result records which file was used."""
    import os

    candidates = [Path(cv_dir) / "validated_sentences.tsv"]
    for root in search_roots or []:
        if root and Path(root).exists():
            for dirpath, _d, files in os.walk(root, followlinks=True):
                if "validated_sentences.tsv" in files:
                    candidates.append(Path(dirpath) / "validated_sentences.tsv")
    for p in candidates:
        if p.exists():
            df = read_tsv(p)
            if "sentence" in df.columns and len(df):
                keep = [c for c in ("sentence", "is_used", "clips_count", "source") if c in df.columns]
                out = df[keep].copy()
                out.attrs["pool_source"] = str(p)
                return out
    fallback = Path(cv_dir) / "validated.tsv"
    if fallback.exists():
        df = read_tsv(fallback)
        if "sentence" in df.columns:
            out = df[["sentence"]].drop_duplicates().copy()
            out.attrs["pool_source"] = f"{fallback} (fallback: validated_sentences.tsv not found)"
            return out
    out = pd.DataFrame(columns=["sentence", "is_used", "clips_count"])
    out.attrs["pool_source"] = "none found"
    return out


def clip_durations(cv_dir: str | Path) -> dict[str, float]:
    """Durations (s) from ``clip_durations.tsv`` when present (file name -> seconds)."""
    p = Path(cv_dir) / "clip_durations.tsv"
    if not p.exists():
        return {}
    df = read_tsv(p)
    col = [c for c in df.columns if c.startswith("duration")]
    if not col:
        return {}
    return {r["clip"]: float(r[col[0]]) / 1000.0 for _, r in df.iterrows()}
