"""Synthesise 3 demo clips for the web app with MMS-TTS (no Common Voice audio is ever shipped).

Usage: python scripts/make_demo_assets.py
Output: app/assets/demo_{1,2,3}.wav + app/assets/demo_transcripts.csv
Licence of the generated audio: facebook/mms-tts-yor is CC BY-NC 4.0 (research/demo use only).
"""
from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf

from yoruba_asr.audio import SR, resample
from yoruba_asr.augment.tts import MMSTTS, coverage_report
from yoruba_asr.text import normalize_text

SENTENCES = [
    "Ẹ kú àárọ̀, ṣé dáadáa ni ẹ jí?",
    "Ọmọ náà ń lọ sí ilé ìwé ní òwúrọ̀ yìí.",
    "Àwọn àgbẹ̀ ń gbin àgbàdo sí oko wọn.",
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "app" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    tts = MMSTTS("facebook/mms-tts-yor", device="cpu")
    texts = [normalize_text(s) for s in SENTENCES]
    cov = coverage_report(texts, tts.vocab)
    print("coverage:", {k: cov[k] for k in ("normal_form", "tone_marks_representable", "subdots_representable",
                                            "enabled_effective", "missing_nfc")})
    rows = []
    for i, (raw, text) in enumerate(zip(SENTENCES, texts), 1):
        y = resample(tts.synthesize(text, seed=42 + i), tts.sampling_rate, SR)
        path = out / f"demo_{i}.wav"
        sf.write(path, y, SR, subtype="PCM_16")
        rows.append({"file": path.name, "sentence": raw, "normalised": text, "seconds": round(len(y) / SR, 2)})
        print(path, rows[-1]["seconds"], "s")
    with open(out / "demo_transcripts.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
