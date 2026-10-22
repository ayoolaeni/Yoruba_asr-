import unicodedata

import pytest

from yoruba_asr.text import (has_digits, has_subdot, has_tone_mark, nfc, normalize_text, strip_tones,
                             tone_sequence)


def test_nfc_handles_decomposed_input():
    decomposed = unicodedata.normalize("NFD", "Ọ̀rẹ́")
    assert normalize_text(decomposed) == nfc("ọ̀rẹ́")
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_strip_tones_keeps_subdots():
    assert strip_tones("ọ̀rẹ́ àwọn ṣé") == "ọrẹ awọn ṣe"
    assert has_subdot(strip_tones("ẹ̀kọ́"))
    assert not has_tone_mark(strip_tones("ẹ̀kọ́ ń"))


def test_normalize_removes_punctuation_and_lowercases():
    assert normalize_text("Ṣé o ti jẹun, Bàbá?  “Bẹ́ẹ̀ ni!”") == nfc("ṣé o ti jẹun bàbá bẹ́ẹ̀ ni")
    assert normalize_text("gan-an") == "ganan"  # hyphens removed as in the original baseline


def test_normalize_without_tones():
    assert normalize_text("Ọ̀rẹ́ mi", keep_tones=False) == nfc("ọrẹ mi")


@pytest.mark.parametrize("text,expected", [
    ("ọkọ́", ["M", "H"]), ("ọkọ̀", ["M", "L"]), ("ọkọ", ["M", "M"]), ("ọ̀kọ̀", ["L", "L"]),
    ("ń lọ", ["H", "M"]), ("àwọn", ["L", "M"]),
])
def test_tone_sequence_spec_cases(text, expected):
    assert tone_sequence(text) == expected


def test_tone_sequence_macron_and_nfd():
    assert tone_sequence("ā") == ["M"]
    assert tone_sequence(unicodedata.normalize("NFD", "ẹ́")) == ["H"]
    assert tone_sequence("ǹ") == ["L"]


def test_digits():
    assert has_digits("ọjọ́ 12")
    assert not has_digits("ọjọ́ méjìlá")
