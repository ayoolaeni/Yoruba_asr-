"""Yoruba text normalisation and tone utilities.

Normalisation reproduces the original Colab baseline exactly (NFC, lower-case, remove the
same punctuation set, collapse spaces). Extra helpers support tone-aware evaluation.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

GRAVE, ACUTE, MACRON, SUBDOT = "̀", "́", "̄", "̣"
TONE_MARKS = {GRAVE, ACUTE, MACRON}  # grave (low), acute (high), macron (mid)
# Same character class as the original notebook (hyphens and en/em dashes are removed).
PUNCT = r"[\,\?\.\!\;\:\"\“\”\‘\’\…\(\)\[\]\{\}\–\—\«\»\-]"
_PUNCT_RE = re.compile(PUNCT)
_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")

VOWELS = set("aeiou")
NASALS = set("nm")
TONE_LABELS = ("H", "M", "L")
_MARK_TO_LABEL = {ACUTE: "H", GRAVE: "L", MACRON: "M"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_tones(s: str) -> str:
    """Remove tone marks (grave/acute/macron) but keep the sub-dot of ẹ, ọ, ṣ."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if ch not in TONE_MARKS)
    return unicodedata.normalize("NFC", s)


def normalize_text(s: str, keep_tones: bool = True) -> str:
    s = nfc(s).lower()
    s = _PUNCT_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return nfc(s if keep_tones else strip_tones(s))


def has_digits(s: str) -> bool:
    return bool(_DIGIT_RE.search(s))


def has_tone_mark(s: str) -> bool:
    return any(ch in TONE_MARKS for ch in unicodedata.normalize("NFD", s))


def has_subdot(s: str) -> bool:
    return SUBDOT in unicodedata.normalize("NFD", s)


def count_intraword(s: str) -> dict[str, int]:
    """Count apostrophes and hyphens/dashes that sit inside words (logged for the thesis)."""
    return {
        "apostrophes": len(re.findall(r"\w['’‘]\w", s)),
        "hyphens": len(re.findall(r"\w[-–—]\w", s)),
    }


def _tbu_units(text: str) -> list[tuple[str, str | None, bool]]:
    """Walk NFD text and return ``(base, label, is_tbu)`` for every vowel/nasal letter.

    ``label`` is H/L/M from the following combining marks (None for a nasal without a mark).
    """
    s = unicodedata.normalize("NFD", text.lower())
    units: list[tuple[str, str | None, bool]] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        j = i + 1
        marks = []
        while j < n and unicodedata.combining(s[j]):
            marks.append(s[j])
            j += 1
        if ch in VOWELS or ch in NASALS:
            label = None
            for m in marks:
                if m in _MARK_TO_LABEL:
                    label = _MARK_TO_LABEL[m]
            if ch in VOWELS:
                units.append((ch, label or "M", True))
            else:
                units.append((ch, label, label is not None))
        i = j
    return units


def tone_sequence(text: str) -> list[str]:
    """One tone label (H/M/L) per tone-bearing unit (TBU).

    TBUs are the vowels a e i o u (ẹ/ọ count as e/o) plus any n/m carrying a tone mark
    (syllabic nasal). Acute -> H, grave -> L, macron or no mark -> M.
    Examples: ọkọ́ -> [M, H]; ń lọ -> [H, M]; àwọn -> [L, M].
    """
    return [label for _, label, is_tbu in _tbu_units(text) if is_tbu]


def word_tone_pairs(ref_word: str, hyp_word: str) -> list[tuple[str, str]]:
    """Position-wise (ref, hyp) tone labels for two words with equal tone-stripped forms.

    Units are vowels and nasals; a nasal counts if the reference marks it as a TBU.
    A nasal unmarked in the hypothesis is read as M.
    """
    r_units, h_units = _tbu_units(ref_word), _tbu_units(hyp_word)
    pairs = []
    for (rb, rl, r_tbu), (_hb, hl, _h_tbu) in zip(r_units, h_units):
        if r_tbu:
            pairs.append((rl or "M", hl or "M"))
    return pairs


def char_inventory(texts: Iterable[str]) -> dict[str, int]:
    """Character counts (NFC code points), sorted by frequency."""
    c = Counter()
    for t in texts:
        c.update(nfc(t))
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def describe_char(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return f"U+{ord(ch):04X}"
