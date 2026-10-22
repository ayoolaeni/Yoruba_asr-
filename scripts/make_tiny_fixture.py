"""Create a tiny synthetic dataset for tests and the smoke pipeline.

* ``common_voice/`` : Common Voice-style TSVs + 40 short WAV clips (tone-coded sine "syllables" + noise)
* ``slr86_female/`` : OpenSLR 86-style ``line_index.tsv`` + WAVs
* ``iroyin_speech/``: a manifest corpus (``manifest.csv`` with path,sentence,speaker)

All sentences were written for this fixture; none come from Common Voice.
Usage: ``python scripts/make_tiny_fixture.py [out_dir]`` (default: tests/fixtures/data)
"""
from __future__ import annotations

import csv
import hashlib
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

TRAIN = [
    "Ọmọ náà ń lọ sí ilé ìwé.", "Bàbá mi ra ọkọ̀ tuntun.", "Ìyá wa ń se oúnjẹ ní ilé.",
    "Ojú ọjọ́ dára púpọ̀ lónìí.", "Mo fẹ́ jẹ ẹ̀wà àti dòdò.", "Àwọn ọmọdé ń ṣeré ní àgbàlá.",
    "Olùkọ́ wa kọ́ wa ní èdè Yorùbá.", "Ṣé o ti jẹun láàárọ̀ yìí?", "Ọjà Ọ̀yọ́ kún fún àwọn èèyàn.",
    "Ẹ kú iṣẹ́ o, ẹ̀yin ará!",
]
DEV = ["Òjò ń rọ̀ ní ìlú Ìbàdàn.", "Mo ní ìwé mẹ́ta nínú àpò mi.", "Ajá náà ń gbó ní òru.",
       "Ilé wa wà ní ẹ̀gbẹ́ odò."]
TEST = ["Àgbẹ̀ náà gbin àgbàdo sí oko rẹ̀.", "Kí ni orúkọ rẹ, ọ̀rẹ́ mi?", "Ọba ìlú náà jẹ́ ènìyàn rere.",
        "A máa lọ sí ṣọ́ọ̀ṣì lọ́la."]
OTHER = ["Ẹyẹ náà fò lọ sí orí igi.", "Omi tútù dára fún ara.", "Ọkùnrin náà ń kọrin.", "Ìwé yìí wúlò gan-an."]
POOL_EXTRA = ["Ìrìn àjò wa lọ sí Èkó dùn mọ́ mi.", "Ọ̀gá wa ti dé láti ọjà.", "Ẹ jọ̀wọ́ ẹ fún mi ní omi.",
              "Àwọn àgbẹ̀ ń kórè iṣu.", "Mo rí ọ̀rẹ́ mi ní ọjà.", "Ọmọ ọba ń jó ní àgbàlá.",
              "Ilé ìwé wa tóbi gan-an.", "Ìyá àgbà ń sọ ìtàn.", "Ọ̀nà náà jìn púpọ̀.", "Ọdún tuntun ti dé."]
SLR86 = ["Ẹ̀kọ́ ṣe pàtàkì fún gbogbo ọmọ.", "Àṣà Yorùbá lẹ́wà gan-an.", "Ọjà ń ta dáradára lónìí.",
         "Ìlú wa ní àlàáfíà.", "Ọ̀rọ̀ àgbà kì í tàsé.", "Ọkọ̀ ojú irin ti dé.",
         "Ẹ̀gbọ́n mi ń kàwé.", "Owó kò tó nǹkan."]
IROYIN = ["Ìjọba ti kéde ètò tuntun fún àwọn àgbẹ̀.", "Àwọn akẹ́kọ̀ọ́ ti padà sí ilé ìwé.",
          "Ọjà epo ti gbówó lórí.", "Òjò ń rọ̀ ní ìlú Ìbàdàn.",  # duplicates a DEV sentence -> leakage filter
          "Ilé ẹjọ́ ti dá ẹjọ́ náà.", "Ìpàdé náà wáyé ní ọjọ́ 12 oṣù kẹta."]  # contains digits -> dropped

F0 = {"H": 230.0, "M": 190.0, "L": 150.0}


def _tones(text: str) -> list[str]:
    s = unicodedata.normalize("NFD", text.lower())
    out = []
    for i, ch in enumerate(s):
        if ch in "aeiou":
            j, lab = i + 1, "M"
            while j < len(s) and unicodedata.combining(s[j]):
                lab = {"́": "H", "̀": "L"}.get(s[j], lab)
                j += 1
            out.append(lab)
    return out or ["M"]


def synth(text: str, sr: int, seed: int, speaker_shift: float = 1.0, max_syll: int = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(0.15 * sr))]
    for lab in _tones(text)[:max_syll]:
        n = int(rng.uniform(0.09, 0.14) * sr)
        t = np.arange(n) / sr
        f = F0[lab] * speaker_shift
        tone = 0.3 * np.sin(2 * np.pi * f * t) + 0.1 * np.sin(2 * np.pi * 2 * f * t)
        parts += [tone * np.hanning(n), np.zeros(int(0.02 * sr))]
    parts.append(np.zeros(int(0.15 * sr)))
    y = np.concatenate(parts)
    return (y + 0.004 * rng.standard_normal(len(y))).astype(np.float32)


def _cid(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def make_fixture(root: str | Path) -> Path:
    root = Path(root)
    cv = root / "common_voice" / "cv-corpus-27.0-test" / "yo"
    clips = cv / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    header = ["client_id", "path", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes", "age",
              "gender", "accents", "variant", "locale", "segment"]
    # speakers are disjoint across splits (as in Common Voice)
    plan = {
        "train": [(s, spk) for spk in ("spk_a", "spk_b") for s in TRAIN],
        "dev": [(s, spk) for spk in ("spk_c", "spk_d") for s in DEV],
        "test": [(s, spk) for spk in ("spk_e", "spk_f") for s in TEST],
        "other": [(s, "spk_g") for s in OTHER],
    }
    meta = {"spk_a": ("twenties", "male_masculine", 0.9), "spk_b": ("thirties", "female_feminine", 1.25),
            "spk_c": ("twenties", "female_feminine", 1.2), "spk_d": ("forties", "male_masculine", 0.85),
            "spk_e": ("twenties", "female_feminine", 1.15), "spk_f": ("", "", 1.0),
            "spk_g": ("fifties", "male_masculine", 0.95)}
    durations, n = [], 0
    validated_rows = []
    for split, items in plan.items():
        rows = []
        for i, (sent, spk) in enumerate(items):
            n += 1
            name = f"fixture_yo_{n:04d}.wav"
            age, gender, shift = meta[spk]
            y = synth(sent, 48000, seed=n, speaker_shift=shift)
            if split == "train" and i == 0:
                y = y[: int(0.3 * 48000)]  # too short after trimming -> filtered
            sf.write(clips / name, y, 48000, subtype="PCM_16")
            durations.append((name, int(1000 * len(y) / 48000)))
            row = [_cid(spk), name, _cid(sent), sent, "", "2", "0", age, gender, "", "", "yo", ""]
            rows.append(row)
            if split != "other":
                validated_rows.append(row)
        if split == "other":  # one corrupt (empty) file in the unlabelled pool
            (clips / "fixture_yo_corrupt.wav").write_bytes(b"")
            rows.append([_cid("spk_g"), "fixture_yo_corrupt.wav", _cid("x"), OTHER[0], "", "0", "0", "", "", "",
                         "", "yo", ""])
        with open(cv / f"{split}.tsv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(header)
            w.writerows(rows)
    for name in ("validated.tsv", "invalidated.tsv"):
        with open(cv / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(header)
            w.writerows(validated_rows if name == "validated.tsv" else [])
    with open(cv / "clip_durations.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("clip\tduration[ms]\n")
        for name, ms in durations:
            fh.write(f"{name}\t{ms}\n")
    with open(cv / "validated_sentences.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("sentence_id\tsentence\tvariant\tsentence_domain\tsource\tis_used\tclips_count\n")
        for s in TRAIN + DEV + TEST + POOL_EXTRA:
            fh.write(f"{_cid(s)}\t{s}\t\t\tfixture\t1\t1\n")

    slr = root / "slr86_female"
    slr.mkdir(parents=True, exist_ok=True)
    with open(slr / "line_index.tsv", "w", encoding="utf-8") as fh:
        for i, s in enumerate(SLR86):
            fid = f"yof_{i % 2:05d}_{1000 + i:011d}"
            sf.write(slr / f"{fid}.wav", synth(s, 48000, seed=500 + i, speaker_shift=1.1 + 0.05 * (i % 2)), 48000,
                     subtype="PCM_16")
            fh.write(f"{fid}\t{s}\n")

    ir = root / "iroyin_speech"
    (ir / "audio").mkdir(parents=True, exist_ok=True)
    with open(ir / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "sentence", "speaker"])
        for i, s in enumerate(IROYIN):
            name = f"audio/iroyin_{i:03d}.flac"
            sf.write(ir / name, synth(s, 16000, seed=900 + i, speaker_shift=0.95), 16000, subtype="PCM_16",
                     format="FLAC")
            w.writerow([name, s, f"news_{i % 3}"])
    return root


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "data"
    print(make_fixture(out))
