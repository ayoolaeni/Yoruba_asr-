import pytest

from yoruba_asr.text import nfc
from yoruba_asr.vocab import (PAD, UNK, VocabMismatchError, build_vocab, canonical_charset, load_shared_processor,
                              make_processor, save_vocab)


def test_canonical_set():
    cs = canonical_charset()
    for ch in "abdefghijklmnoprstuwyẹọṣ" + "àáèéìíòóùúńǹḿ" + "̣̀́̄":
        assert ch in cs, ch
    v = build_vocab(["ọmọ"])
    assert v["[PAD]"] == len(v) - 1 and v["[UNK]"] == len(v) - 2 and "|" in v and " " not in v


def test_hash_is_stable(tmp_path):
    a = save_vocab(build_vocab(["ọ̀rẹ́ mi", "bàbá"]), tmp_path / "s1", tmp_path / "r1")
    b = save_vocab(build_vocab(["bàbá", "ọ̀rẹ́ mi"]), tmp_path / "s2", tmp_path / "r2")
    assert a["sha256"] == b["sha256"]


def test_tokenizer_round_trip(tmp_path):
    save_vocab(build_vocab(["ọ̀rẹ́ mi ń lọ"]), tmp_path / "shared", tmp_path / "results")
    proc = make_processor(tmp_path / "shared" / "vocab.json")
    text = nfc("ọ̀rẹ́ mi ń lọ sí ẹ̀gbẹ́ odò")
    ids = proc.tokenizer(text).input_ids
    assert proc.tokenizer.unk_token_id not in ids
    assert nfc(proc.tokenizer.decode(ids, group_tokens=False)) == text


def test_hash_mismatch_fails(tmp_path):
    save_vocab(build_vocab(["a"]), tmp_path / "shared", tmp_path / "results")
    load_shared_processor(tmp_path / "shared", tmp_path / "results")
    (tmp_path / "shared" / "vocab.json").write_text('{"a": 0, "[UNK]": 1, "[PAD]": 2}', encoding="utf-8")
    with pytest.raises(VocabMismatchError):
        load_shared_processor(tmp_path / "shared", tmp_path / "results")


def test_unk_and_pad_tokens(tmp_path):
    save_vocab(build_vocab(["a"]), tmp_path / "shared", tmp_path / "results")
    proc = make_processor(tmp_path / "shared" / "vocab.json")
    assert proc.tokenizer.pad_token == PAD and proc.tokenizer.unk_token == UNK
