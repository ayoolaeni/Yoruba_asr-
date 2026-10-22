"""Inference: load a fine-tuned model and transcribe audio files or arrays (greedy CTC, no LM)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .audio import SR, load_audio, resample, split_on_silence, trim_silence
from .text import nfc, strip_tones

_SPECIAL = ("[UNK]", "[PAD]", "<s>", "</s>")


def clean_output(text: str) -> str:
    """Remove special tokens a CTC model can emit and tidy spaces (display only)."""
    for tok in _SPECIAL:
        text = text.replace(tok, "")
    return " ".join(nfc(text).split())


class Transcriber:
    def __init__(self, model, processor, device: str, name: str = "model"):
        self.model, self.processor, self.device, self.name = model, processor, device, name

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None) -> "Transcriber":
        return cls._load(str(path), device, name=Path(path).name)

    @classmethod
    def from_hub(cls, repo_id: str, device: str | None = None) -> "Transcriber":
        return cls._load(repo_id, device, name=repo_id)

    @classmethod
    def _load(cls, source: str, device: str | None, name: str) -> "Transcriber":
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = Wav2Vec2Processor.from_pretrained(source)
        model = Wav2Vec2ForCTC.from_pretrained(source).to(device).eval()
        return cls(model, processor, device, name)

    def _decode_chunks(self, chunks: list[np.ndarray]) -> list[str]:
        import torch

        texts = []
        for c in chunks:
            inputs = self.processor(c, sampling_rate=SR, return_tensors="pt", padding=True)
            mask = inputs.get("attention_mask")
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device),
                                    attention_mask=mask.to(self.device) if mask is not None else None).logits
            ids = torch.argmax(logits, dim=-1).cpu().numpy()
            texts.append(clean_output(self.processor.batch_decode(ids)[0]))
        return texts

    def transcribe_array(self, audio: np.ndarray, sr: int) -> dict:
        """Transcribe a waveform. Audio longer than 15 s is split on silence into <= 15 s chunks."""
        t0 = time.time()
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
        if np.issubdtype(np.asarray(audio).dtype, np.integer):
            x = x / float(np.iinfo(np.asarray(audio).dtype).max)
        x = resample(x, int(sr), SR)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak
        x = trim_silence(x, 30)
        if len(x) < int(0.1 * SR):
            return {"text": "", "text_notone": "", "seconds": round(time.time() - t0, 3), "model": self.name}
        chunks = split_on_silence(x, SR, max_seconds=15.0)
        text = " ".join(t for t in self._decode_chunks(chunks) if t)
        return {"text": text, "text_notone": strip_tones(text), "seconds": round(time.time() - t0, 3),
                "model": self.name, "audio_seconds": round(len(x) / SR, 2), "chunks": len(chunks)}

    def transcribe_file(self, path: str | Path) -> dict:
        return self.transcribe_array(load_audio(path, SR), SR)

    def transcribe_batch(self, paths: list[str], batch_size: int = 8) -> list[dict]:
        """Batched transcription of short (<= 15 s) files; longer files fall back to chunking."""
        import torch

        out: list[dict | None] = [None] * len(paths)
        short: list[tuple[int, np.ndarray]] = []
        for i, p in enumerate(paths):
            x = trim_silence(load_audio(p, SR), 30)
            if len(x) > 15 * SR:
                out[i] = self.transcribe_array(x, SR)
            else:
                short.append((i, x))
        short.sort(key=lambda t: -len(t[1]))
        for s in range(0, len(short), batch_size):
            part = short[s: s + batch_size]
            t0 = time.time()
            inputs = self.processor([x for _, x in part], sampling_rate=SR, return_tensors="pt", padding=True)
            mask = inputs.get("attention_mask")
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device),
                                    attention_mask=mask.to(self.device) if mask is not None else None).logits
            texts = self.processor.batch_decode(torch.argmax(logits, dim=-1).cpu().numpy())
            dt = (time.time() - t0) / len(part)
            for (i, _), t in zip(part, texts):
                t = clean_output(t)
                out[i] = {"text": t, "text_notone": strip_tones(t), "seconds": round(dt, 3), "model": self.name}
        return out  # type: ignore[return-value]
