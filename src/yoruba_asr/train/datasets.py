"""Training datasets.

* Common Voice (small) is preprocessed once into a Hugging Face ``Dataset`` with
  ``input_values``, ``input_length`` and ``labels`` and saved under ``work_dir/cache/``.
* Large or generated corpora (OOD, augmented FLAC, TTS) are read from manifests and loaded
  on the fly, so no float arrays are written to disk for them.

Every dataset exposes ``lengths`` (samples per item) so the Trainer can group by length.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..audio import SR, duration_reason, probe, safe_load_trimmed
from ..utils import get_logger

log = get_logger(__name__)

CACHE_META_COLUMNS = ["path", "speaker", "gender", "age", "text_tone", "text_notone", "duration", "corpus"]


# --------------------------------------------------------------------------------------
# Common Voice cache
# --------------------------------------------------------------------------------------
def build_cached_split(rows: pd.DataFrame, processor, out_dir: str | Path, tmp_dir: str | Path,
                       min_s: float, max_s: float, top_db: float, text_column: str = "text_tone"):
    """Decode, trim, duration-filter and feature-extract ``rows``; save a HF dataset to ``out_dir``.

    Returns ``(dataset, drop_log)`` where ``drop_log`` maps path -> drop reason.
    """
    from datasets import Dataset, Features, Sequence, Value

    drop_log: dict[str, str] = {}
    fe, tok = processor.feature_extractor, processor.tokenizer
    records = rows.to_dict("records")

    def gen():
        for r in records:
            audio = safe_load_trimmed(r["path"], SR, top_db)
            if audio is None:
                drop_log[r["path"]] = "corrupt"
                continue
            dur = len(audio) / SR
            reason = duration_reason(dur, min_s, max_s)
            if reason:
                drop_log[r["path"]] = reason
                continue
            iv = fe(audio, sampling_rate=SR).input_values[0].astype(np.float32)
            yield {
                "input_values": iv,
                "input_length": int(len(iv)),
                "labels": tok(r[text_column]).input_ids,
                "path": str(r["path"]), "speaker": str(r.get("speaker", "unknown")),
                "gender": str(r.get("gender", "")), "age": str(r.get("age", "")),
                "text_tone": str(r["text_tone"]), "text_notone": str(r["text_notone"]),
                "duration": float(dur), "corpus": str(r.get("corpus", "")),
            }

    features = Features({
        "input_values": Sequence(Value("float32")), "input_length": Value("int64"),
        "labels": Sequence(Value("int32")), "path": Value("string"), "speaker": Value("string"),
        "gender": Value("string"), "age": Value("string"), "text_tone": Value("string"),
        "text_notone": Value("string"), "duration": Value("float32"), "corpus": Value("string"),
    })
    gen_cache = Path(tmp_dir) / "hf_gen" / uuid.uuid4().hex
    try:
        ds = Dataset.from_generator(gen, features=features, cache_dir=str(gen_cache))
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        reasons = {r: list(drop_log.values()).count(r) for r in set(drop_log.values())}
        raise RuntimeError(f"No usable audio clip in this split ({len(records)} listed, dropped: {reasons}). "
                           "Check that the Common Voice 'clips' folder was uploaded completely.") from exc
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    return ds, drop_log


def load_cached_split(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def arrow_row(hf_dataset, i: int, columns=("input_values", "labels")) -> dict:
    """Read one row straight from the Arrow table.

    Deliberately avoids ``with_format("numpy")``: in ``datasets`` that formatter imports
    ``torchvision.io.VideoReader``, which newer torchvision builds (e.g. on Kaggle) no longer have.
    """
    table = hf_dataset.with_format("arrow")[int(i): int(i) + 1]
    out = {}
    if "input_values" in columns:
        out["input_values"] = table.column("input_values")[0].values.to_numpy(zero_copy_only=False).astype(np.float32)
    if "labels" in columns:
        out["labels"] = table.column("labels")[0].values.to_numpy(zero_copy_only=False).astype(np.int64).tolist()
    return out


class CachedASRDataset(torch.utils.data.Dataset):
    """Wraps a cached HF dataset; returns ``input_values`` (np.float32) and ``labels``."""

    def __init__(self, hf_dataset, limit: int | None = None):
        if limit is not None and limit < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(limit))
        self.hf = hf_dataset
        self.lengths = list(hf_dataset["input_length"])

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, i: int) -> dict:
        return arrow_row(self.hf, int(i))

    def metadata(self) -> pd.DataFrame:
        cols = [c for c in CACHE_META_COLUMNS if c in self.hf.column_names]
        return self.hf.select_columns(cols).to_pandas()


class ManifestASRDataset(torch.utils.data.Dataset):
    """Loads audio on the fly from a manifest with ``path, text_tone, duration`` columns."""

    def __init__(self, manifest: pd.DataFrame, processor, top_db: float = 30, text_column: str = "text_tone",
                 trim: bool = True):
        self.df = manifest.reset_index(drop=True)
        self.processor = processor
        self.top_db = top_db
        self.trim = trim
        self.text_column = text_column
        self.lengths = [int(float(d) * SR) for d in self.df["duration"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[int(i)]
        audio = safe_load_trimmed(r["path"], SR, self.top_db if self.trim else 200)
        if audio is None or len(audio) < int(0.1 * SR):
            audio = np.zeros(int(0.5 * SR), dtype=np.float32)  # corrupt file: silent stand-in, label kept short
            labels = []
        else:
            labels = self.processor.tokenizer(r[self.text_column]).input_ids
        iv = self.processor.feature_extractor(audio, sampling_rate=SR).input_values[0].astype(np.float32)
        return {"input_values": iv, "labels": labels}


class ConcatASRDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Iterable[torch.utils.data.Dataset]):
        datasets = [d for d in datasets if d is not None and len(d) > 0]
        super().__init__(datasets)
        self.lengths = [n for d in datasets for n in d.lengths]


def probe_manifest(df: pd.DataFrame, min_s: float, max_s: float) -> pd.DataFrame:
    """Add ``duration``, ``sample_rate`` and ``drop_reason`` columns using file headers only."""
    durs, srs, reasons = [], [], []
    for p in df["path"]:
        info = probe(p)
        if info is None:
            durs.append(np.nan); srs.append(np.nan); reasons.append("corrupt")
            continue
        d, sr = info
        durs.append(d); srs.append(sr); reasons.append(duration_reason(d, min_s, max_s))
    out = df.copy()
    out["duration"], out["sample_rate"], out["drop_reason"] = durs, srs, reasons
    return out
