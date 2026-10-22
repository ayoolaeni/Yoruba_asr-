"""Synthetic Yoruba speech with MMS-TTS (``facebook/mms-tts-yor``, CC BY-NC 4.0, research use only).

Before synthesising, a coverage check verifies that the TTS tokenizer can represent the tone marks
and sub-dots of the normalised text. If tone marks cannot be represented, TTS is skipped
(``enabled_effective = false``): toneless audio must never be labelled as toned text.
"""
from __future__ import annotations

import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..audio import SR, duration_reason, resample, save_flac, trim_silence
from ..text import ACUTE, GRAVE, MACRON, SUBDOT, TONE_MARKS, nfc, strip_tones, tone_sequence
from ..utils import get_logger, save_json

log = get_logger(__name__)


class DummyTTS:
    """Offline stand-in for tests/smoke runs: renders the tone sequence as short sine syllables."""

    name = "dummy"
    sampling_rate = SR

    def __init__(self):
        self.vocab = None  # represents every character

    def synthesize(self, text: str, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        f0 = {"H": 240.0, "M": 200.0, "L": 160.0}
        parts = []
        for lab in tone_sequence(text) or ["M"]:
            t = np.arange(int(0.12 * SR)) / SR
            parts.append(0.3 * np.sin(2 * np.pi * f0[lab] * t) * np.hanning(len(t)))
            parts.append(np.zeros(int(0.02 * SR)))
        y = np.concatenate([np.zeros(int(0.1 * SR))] + parts + [np.zeros(int(0.1 * SR))])
        return (y + 0.003 * rng.standard_normal(len(y))).astype(np.float32)


class MMSTTS:
    def __init__(self, model_name: str = "facebook/mms-tts-yor", device: str | None = None):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device).eval()
        self.sampling_rate = int(self.model.config.sampling_rate)
        self.vocab = set(self.tokenizer.get_vocab().keys())
        self.normal_form = "NFC"

    def synthesize(self, text: str, seed: int) -> np.ndarray:
        import torch

        torch.manual_seed(seed)
        text = unicodedata.normalize(self.normal_form, text)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = self.model(**inputs).waveform[0].float().cpu().numpy()
        return wav.astype(np.float32)


def load_tts(model_name: str):
    if model_name == "dummy":
        return DummyTTS()
    return MMSTTS(model_name)


def coverage_report(sentences: list[str], vocab: set[str] | None) -> dict:
    """Which characters (NFC and NFD forms) the TTS tokenizer cannot represent."""
    text = " ".join(sentences)
    if vocab is None:
        return {"normal_form": "NFC", "missing_nfc": {}, "missing_nfd": {}, "tone_marks_representable": True,
                "subdots_representable": True, "enabled_effective": True}

    def missing(form: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for ch in unicodedata.normalize(form, text):
            if ch != " " and ch not in vocab and ch.lower() not in vocab:
                out[ch] = out.get(ch, 0) + 1
        return out

    def tones_ok(form: str, miss: dict[str, int]) -> tuple[bool, bool]:
        # High (acute) and low (grave) tones carry the contrast in standard orthography and must be
        # representable. The mid-tone macron is rare (mid tone is normally unmarked); sentences that
        # contain an unrepresentable macron are skipped one by one instead of disabling TTS.
        decomposed = {ch: unicodedata.normalize("NFD", ch) for ch in miss}
        toned = [ch for ch, d in decomposed.items() if (GRAVE in d or ACUTE in d)]
        dotted = [ch for ch, d in decomposed.items() if SUBDOT in d]
        return not toned, not dotted

    m_nfc, m_nfd = missing("NFC"), missing("NFD")
    t_nfc, d_nfc = tones_ok("NFC", m_nfc)
    t_nfd, d_nfd = tones_ok("NFD", m_nfd)
    share = {f: float(np.mean([sentence_representable(s, vocab, f) for s in sentences])) if sentences else 0.0
             for f in ("NFC", "NFD")}
    if t_nfc and d_nfc:
        form, t_ok, d_ok = "NFC", True, True
    elif t_nfd and d_nfd:
        form, t_ok, d_ok = "NFD", True, True
    else:
        # A few rare characters (e.g. ḿ, an m with a tone mark) may be missing. Tones still count as
        # representable if the core toned vowels can be written in this form; the sentences that contain
        # an unrepresentable character are then skipped one by one (never stripped).
        form = max(("NFC", "NFD"), key=lambda f: share[f])
        core = [unicodedata.normalize(form, v + m) for v in "aeiou" for m in (ACUTE, GRAVE)]
        core_ok = all(all(ch in vocab or ch.lower() in vocab for ch in c) for c in core)
        dot_ok = all(ch in vocab for ch in unicodedata.normalize(form, "ẹọṣ"))
        t_ok, d_ok = core_ok and share[form] >= 0.5, dot_ok
    named = lambda d: {f"{k} ({unicodedata.name(k, hex(ord(k)))})": v for k, v in d.items()}  # noqa: E731
    return {
        "normal_form": form, "missing_nfc": named(m_nfc), "missing_nfd": named(m_nfd),
        "tone_marks_representable": t_ok, "subdots_representable": d_ok,
        "enabled_effective": bool(t_ok and d_ok),
        "share_of_sentences_representable": round(share[form], 4),
        "macron_representable": not any(MACRON in unicodedata.normalize("NFD", ch)
                                        for ch in (m_nfc if form == "NFC" else m_nfd)),
        "policy": "sentences containing any unrepresentable character are skipped (never silently stripped)",
        "combining_marks_in_vocab": {unicodedata.name(c): c in vocab for c in (GRAVE, ACUTE, MACRON, SUBDOT)},
    }


def sentence_representable(sentence: str, vocab: set[str] | None, form: str) -> bool:
    if vocab is None:
        return True
    return all(ch == " " or ch in vocab or ch.lower() in vocab for ch in unicodedata.normalize(form, sentence))


def synthesize_set(sentences: pd.DataFrame, out_dir: str | Path, results_dir: str | Path, model_name: str,
                   seed: int, max_items: int, min_s: float, max_s: float, top_db: float,
                   listening_sample: int = 50) -> tuple[pd.DataFrame, dict]:
    """Synthesise up to ``max_items`` sentences (column ``text_tone``); returns (manifest, coverage)."""
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tts = load_tts(model_name)
    except Exception as exc:
        cov = {"enabled_effective": False, "error": f"could not load {model_name}: {exc}"}
        save_json(cov, results_dir / "tts_coverage.json")
        log.warning("TTS disabled: %s", cov["error"])
        return pd.DataFrame(), cov

    texts = sentences["text_tone"].tolist()
    cov = coverage_report(texts, tts.vocab)
    cov["model"] = model_name
    save_json(cov, results_dir / "tts_coverage.json")
    if not cov["enabled_effective"]:
        log.warning("=" * 80 + "\nTTS SKIPPED: the TTS tokenizer cannot represent Yoruba tone marks/sub-dots "
                    "(see results/V2/tts_coverage.json). V2/V4 will train without synthetic speech.\n" + "=" * 80)
        return pd.DataFrame(), cov
    if hasattr(tts, "normal_form"):
        tts.normal_form = cov["normal_form"]

    rows, skipped = [], {"unrepresentable": 0, "too_short": 0, "too_long": 0, "error": 0}
    for i, r in enumerate(sentences.to_dict("records")):
        if len(rows) >= max_items:
            break
        text = r["text_tone"]
        if not sentence_representable(text, tts.vocab, cov["normal_form"]):
            skipped["unrepresentable"] += 1
            continue
        path = out_dir / f"tts_{i:06d}.flac"
        try:
            if path.exists():
                import soundfile as sf

                y, _ = sf.read(str(path), dtype="float32")
            else:
                y = tts.synthesize(text, seed=seed * 100003 + i)
                y = trim_silence(resample(y, tts.sampling_rate, SR), top_db)
            dur = len(y) / SR
            reason = duration_reason(dur, min_s, max_s)
            if reason:
                skipped[reason] += 1
                continue
            if not path.exists():
                save_flac(path, y)
        except Exception as exc:
            log.warning("TTS failed for %r: %s", text[:40], exc)
            skipped["error"] += 1
            continue
        rows.append({"path": str(path), "sentence": r.get("sentence", text), "text_tone": text,
                     "text_notone": strip_tones(text), "speaker": "tts", "corpus": "tts", "split": "train",
                     "duration": dur, "synthetic": True, "chain": "tts"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest_tts.csv", index=False, encoding="utf-8")
    cov["synthesised"] = len(rows)
    cov["skipped"] = skipped
    save_json(cov, results_dir / "tts_coverage.json")

    # Listening sample for manual quality control.
    sample_dir = results_dir / "tts_listening_sample"
    if len(manifest):
        sample_dir.mkdir(parents=True, exist_ok=True)
        pick = manifest.sample(n=min(listening_sample, len(manifest)), random_state=seed)
        for p in pick["path"]:
            shutil.copy2(p, sample_dir / Path(p).name)
        pick.assign(file=[Path(p).name for p in pick["path"]])[["file", "text_tone"]].to_csv(
            sample_dir / "transcripts.csv", index=False, encoding="utf-8")
    log.info("TTS: %d clips synthesised (%s)", len(rows), skipped)
    return manifest, cov


def nfc_all(xs):
    return [nfc(x) for x in xs]
