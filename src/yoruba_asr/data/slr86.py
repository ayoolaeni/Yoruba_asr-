"""Google crowdsourced Yoruba speech corpus (OpenSLR 86, CC BY-SA 4.0).

Each archive (``yo_ng_female`` / ``yo_ng_male``) has a ``line_index.tsv`` mapping a file id to a
transcript, plus ``<id>.wav`` files. Lines are ``<id>\\t<transcript>``; some releases use a
space instead of a tab, so both are accepted. The speaker is the id prefix (``yof_00295``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import ensure_columns


_TAG = re.compile(r"\[[^\]]*\]")
# Recordings whose audio stops mid-sentence: the transcript does not match the audio, so they are dropped.
DROP_TAGS = {"[abrupt]"}


def clean_annotations(sentence: str) -> tuple[str, list[str]]:
    """Remove OpenSLR annotation tags ([breath], [snap], [external], [hesitation], [abrupt])."""
    tags = [t.lower() for t in _TAG.findall(sentence)]
    return " ".join(_TAG.sub(" ", sentence).split()), tags


def _index_audio(folder: Path) -> dict[str, Path]:
    return {p.stem: p for ext in ("*.wav", "*.flac") for p in folder.rglob(ext)}


def speaker_split(df: pd.DataFrame, dev_share: float = 0.1, test_share: float = 0.1, seed: int = 42,
                  key=None) -> dict[str, pd.DataFrame]:
    """Speaker-disjoint train/dev/test split (done per gender so both are represented in every split).

    Speakers are shuffled with ``seed`` and assigned to test, then dev, until each holds about its share of
    utterances. Training utterances whose sentence (``key``) also occurs in dev/test are removed, so the
    evaluation sentences are unseen as well.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    parts = {"train": [], "dev": [], "test": []}
    for _gender, g in df.groupby(df["gender"].fillna(""), sort=True):
        counts = g.groupby("speaker").size()
        speakers = list(counts.index[rng.permutation(len(counts))])
        total = len(g)
        assigned = {"test": [], "dev": []}
        for split, share in (("test", test_share), ("dev", dev_share)):
            n = 0
            while len(speakers) > 1 and n < share * total:  # always keep >= 1 speaker for training
                s = speakers.pop(0)
                assigned[split].append(s)
                n += counts[s]
        parts["test"].append(g[g["speaker"].isin(assigned["test"])])
        parts["dev"].append(g[g["speaker"].isin(assigned["dev"])])
        parts["train"].append(g[g["speaker"].isin(speakers)])
    out = {k: pd.concat(v, ignore_index=True) if v else df.iloc[:0] for k, v in parts.items()}
    if key is not None:
        held_out = set(out["dev"]["sentence"].map(key)) | set(out["test"]["sentence"].map(key))
        out["train"] = out["train"][~out["train"]["sentence"].map(key).isin(held_out)].reset_index(drop=True)
    for k in out:
        out[k] = out[k].assign(split=k).reset_index(drop=True)
    return out


def load_slr86(folders: list[str | Path]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        folder = Path(folder)
        index = folder / "line_index.tsv"
        if not index.exists():
            continue
        audio = _index_audio(folder)
        gender = "female" if "female" in folder.name.lower() else ("male" if "male" in folder.name.lower() else "")
        with open(index, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1) if "\t" in line else re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    continue
                fid, sentence = parts[0].strip().lstrip("﻿"), parts[1].strip()
                sentence, tags = clean_annotations(sentence)
                if not sentence or DROP_TAGS & set(tags):
                    continue
                path = audio.get(fid)
                if path is None:
                    continue
                m = re.match(r"^(yo[fm]_\d+)", fid)
                if gender == "" and fid.startswith(("yof", "yom")):
                    g = "female" if fid.startswith("yof") else "male"
                else:
                    g = gender
                rows.append({"path": str(path), "sentence": sentence,
                             "speaker": m.group(1) if m else "unknown", "corpus": "slr86",
                             "split": "ood_train", "gender": g, "age": ""})
    return ensure_columns(pd.DataFrame(rows, columns=["path", "sentence", "speaker", "corpus", "split",
                                                      "gender", "age"]))
