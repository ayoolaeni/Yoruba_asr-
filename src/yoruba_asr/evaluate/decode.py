"""Batched greedy CTC decoding of a test set and writing ``predictions.csv`` + ``metrics.json``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..text import nfc, strip_tones
from ..utils import get_logger, save_json
from .metrics import all_metrics

log = get_logger(__name__)


def load_model_and_processor(model_dir: str | Path, device: str | None = None):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device).eval()
    return model, processor, device


@torch.no_grad()
def greedy_decode_arrays(model, processor, arrays: list[np.ndarray], device: str, batch_size: int = 8,
                         normalized: bool = True) -> list[str]:
    """Decode a list of 16 kHz arrays. ``normalized=True`` means they are already feature-normalised."""
    order = np.argsort([-len(a) for a in arrays])
    out: list[str] = [""] * len(arrays)
    for start in range(0, len(arrays), batch_size):
        idx = order[start: start + batch_size]
        batch = [arrays[i] for i in idx]
        if normalized:
            feats = processor.feature_extractor.pad([{"input_values": a} for a in batch], padding=True,
                                                    return_tensors="pt")
        else:
            feats = processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = feats["input_values"].to(device)
        mask = feats.get("attention_mask")
        logits = model(inputs, attention_mask=mask.to(device) if mask is not None else None).logits
        ids = torch.argmax(logits, dim=-1).cpu().numpy()
        texts = processor.batch_decode(ids)
        for i, t in zip(idx, texts):
            out[i] = nfc(t).strip()
    return out


def decode_cached(model, processor, hf_dataset, device: str, batch_size: int = 8) -> pd.DataFrame:
    """Decode a cached HF split (input_values + metadata) -> predictions DataFrame."""
    from ..train.datasets import arrow_row

    arrays = [arrow_row(hf_dataset, i, ("input_values",))["input_values"] for i in range(len(hf_dataset))]
    hyps = greedy_decode_arrays(model, processor, arrays, device, batch_size, normalized=True)
    meta = hf_dataset.select_columns([c for c in ("path", "speaker", "gender", "age", "text_tone", "duration")
                                      if c in hf_dataset.column_names]).to_pandas()
    return pd.DataFrame({
        "path": meta["path"], "speaker": meta["speaker"],
        "gender": meta.get("gender", ""), "age": meta.get("age", ""), "duration": meta.get("duration", np.nan),
        "reference": meta["text_tone"], "hypothesis": hyps,
        "reference_notone": [strip_tones(r) for r in meta["text_tone"]],
        "hypothesis_notone": [strip_tones(h) for h in hyps],
    })


def evaluate_predictions(preds: pd.DataFrame, variant: str, out_dir: str | Path, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    m = all_metrics(preds["reference"].tolist(), preds["hypothesis"].fillna("").tolist())
    metrics = {"variant": variant, "n_utts": int(len(preds)), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                               for k, v in m.items()}, **(extra or {})}
    save_json(metrics, out_dir / "metrics.json")
    log.info("[%s] test: WER=%.2f CER=%.2f WER(no tone)=%.2f TER=%.2f (n=%d)", variant, m["wer"], m["cer"],
             m["wer_no_tone"], m["ter"], len(preds))
    return metrics
