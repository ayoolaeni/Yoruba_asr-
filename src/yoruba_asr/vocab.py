"""Shared character vocabulary and Wav2Vec2 processor.

The vocabulary is built ONCE (before any variant is trained) and every variant loads the same
``shared/vocab.json``; its SHA-256 is stored in ``results/vocab_info.json`` and checked at load.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

from .text import ACUTE, GRAVE, MACRON, SUBDOT, nfc
from .utils import load_json, save_json, sha256_file

BASE_LETTERS = list("abdefghijklmnoprstuwy") + ["ẹ", "ọ", "ṣ"]
TONE_BASES = ["a", "e", "i", "o", "u", "ẹ", "ọ", "n", "m"]
COMBINING = [GRAVE, ACUTE, MACRON, SUBDOT]
WORD_DELIM, UNK, PAD = "|", "[UNK]", "[PAD]"


class VocabMismatchError(RuntimeError):
    pass


def canonical_charset() -> set[str]:
    """Yoruba letters, every precomposed toned vowel/nasal and the standalone combining marks."""
    chars = set(BASE_LETTERS)
    for base in TONE_BASES:
        for mark in (GRAVE, ACUTE, MACRON):
            s = nfc(base + mark)
            if len(s) == 1:
                chars.add(s)
            else:  # e.g. ẹ́ has no precomposed form: base + combining mark
                chars.update(s)
    chars.update(COMBINING)
    return chars


def build_vocab(texts: Iterable[str]) -> dict[str, int]:
    """Union of characters in ``texts`` (normalised NFC) and the canonical set; space -> ``|``."""
    chars = set(canonical_charset())
    for t in texts:
        chars.update(nfc(t))
    chars.add(" ")
    chars.discard(WORD_DELIM)
    ordered = sorted(chars)
    vocab = {c: i for i, c in enumerate(ordered)}
    vocab[WORD_DELIM] = vocab.pop(" ")
    vocab = {c: i for i, c in enumerate(sorted(vocab, key=lambda c: vocab[c]))}
    vocab[UNK] = len(vocab)
    vocab[PAD] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], shared_dir: str | Path, results_dir: str | Path,
               sources: dict | None = None) -> dict:
    shared_dir, results_dir = Path(shared_dir), Path(results_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    path = shared_dir / "vocab.json"
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    info = {
        "vocab_path": str(path),
        "sha256": sha256_file(path),
        "size": len(vocab),
        "characters": [c for c in vocab if c not in (UNK, PAD)],
        "character_names": {c: (unicodedata.name(c, f"U+{ord(c):04X}") if len(c) == 1 else c) for c in vocab},
        "sources": sources or {},
    }
    save_json(info, results_dir / "vocab_info.json")
    return info


def unknown_chars(texts: Iterable[str], vocab: dict[str, int]) -> dict[str, int]:
    """Characters (other than space) missing from the vocabulary, with counts."""
    out: dict[str, int] = {}
    for t in texts:
        for ch in nfc(t):
            if ch != " " and ch not in vocab:
                out[ch] = out.get(ch, 0) + 1
    return out


def make_feature_extractor():
    from transformers import Wav2Vec2FeatureExtractor

    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                    do_normalize=True, return_attention_mask=True)


def make_processor(vocab_path: str | Path):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token=UNK, pad_token=PAD, word_delimiter_token=WORD_DELIM)
    return Wav2Vec2Processor(feature_extractor=make_feature_extractor(), tokenizer=tokenizer)


def load_shared_processor(shared_dir: str | Path, results_dir: str | Path, check_hash: bool = True):
    """Load the processor from the shared vocabulary, failing if the hash differs."""
    path = Path(shared_dir) / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(f"Shared vocabulary not found at {path}. Run the 'prepare' stage first.")
    if check_hash:
        info = load_json(Path(results_dir) / "vocab_info.json")
        if info is None:
            raise VocabMismatchError("results/vocab_info.json missing: cannot verify the shared vocabulary hash.")
        actual = sha256_file(path)
        if actual != info["sha256"]:
            raise VocabMismatchError(f"Shared vocabulary hash mismatch: {actual} != {info['sha256']}. "
                                     "All variants must use the same vocab.json.")
    return make_processor(path)


def vocab_sha256(shared_dir: str | Path) -> str:
    return sha256_file(Path(shared_dir) / "vocab.json")
