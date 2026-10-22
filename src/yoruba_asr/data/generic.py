"""Generic manifest loader: a CSV/TSV with an audio-path column and a transcript column.

Accepted column names (case-insensitive):
  * audio:      path, audio, audio_path, file, filename, wav, audio_filepath
  * transcript: sentence, text, transcript, transcription, normalized_text
  * speaker:    speaker, speaker_id, client_id, spk (optional)
Relative paths are resolved against the manifest's folder (then searched recursively by name).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from . import ensure_columns

AUDIO_COLS = ("path", "audio", "audio_path", "file", "filename", "wav", "audio_filepath", "file_name")
TEXT_COLS = ("sentence", "text", "transcript", "transcription", "normalized_text")
SPEAKER_COLS = ("speaker", "speaker_id", "client_id", "spk", "user_id")


class ManifestError(ValueError):
    pass


MANIFEST_HELP = (
    "Create a manifest CSV with the header 'path,sentence,speaker' (speaker optional), one row per "
    "utterance, where 'path' is the audio file path (absolute or relative to the CSV) and 'sentence' "
    "is the Yoruba transcript with tone marks. Upload it with the audio and set "
    "data.<corpus>.manifest to its path (or place it in the corpus folder)."
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows).astype(str)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8",
                           quoting=csv.QUOTE_NONE)


def _pick(columns, options) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for o in options:
        if o in lower:
            return lower[o]
    return None


def looks_like_manifest(path: Path) -> bool:
    try:
        df = _read_table(path).head(2)
    except Exception:
        return False
    return _pick(df.columns, AUDIO_COLS) is not None and _pick(df.columns, TEXT_COLS) is not None


def load_manifest(manifest: str | Path, corpus: str, split: str = "ood_train",
                  audio_root: str | Path | None = None) -> pd.DataFrame:
    manifest = Path(manifest)
    if not manifest.exists():
        raise ManifestError(f"Manifest not found: {manifest}. {MANIFEST_HELP}")
    df = _read_table(manifest)
    a_col, t_col = _pick(df.columns, AUDIO_COLS), _pick(df.columns, TEXT_COLS)
    if a_col is None or t_col is None:
        raise ManifestError(f"{manifest} has columns {list(df.columns)}; need an audio column {AUDIO_COLS} "
                            f"and a transcript column {TEXT_COLS}. {MANIFEST_HELP}")
    s_col = _pick(df.columns, SPEAKER_COLS)
    root = Path(audio_root) if audio_root else manifest.parent
    by_name: dict[str, Path] | None = None

    def resolve(p: str) -> str:
        nonlocal by_name
        cand = Path(p)
        if cand.is_absolute() and cand.exists():
            return str(cand)
        for base in (root, manifest.parent):
            if (base / p).exists():
                return str(base / p)
        if by_name is None:
            by_name = {}
            for f in root.rglob("*"):
                if f.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"):
                    by_name.setdefault(f.name, f)
        hit = by_name.get(Path(p).name)
        return str(hit) if hit else str(root / p)

    out = pd.DataFrame({
        "path": [resolve(p) for p in df[a_col]],
        "sentence": df[t_col],
        "speaker": df[s_col] if s_col else "unknown",
        "corpus": corpus,
        "split": split,
    })
    out = out[out["sentence"].astype(str).str.strip() != ""]
    return ensure_columns(out.reset_index(drop=True))


def find_manifest(folder: str | Path) -> Path | None:
    """Auto-detect a manifest table inside ``folder`` (largest matching CSV/TSV/JSONL)."""
    folder = Path(folder)
    cands = [p for ext in ("*.csv", "*.tsv", "*.jsonl") for p in folder.rglob(ext) if looks_like_manifest(p)]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_size)


def list_audio(folder: str | Path) -> list[str]:
    folder = Path(folder)
    return sorted(str(p) for p in folder.rglob("*")
                  if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"))
