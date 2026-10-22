import numpy as np
import pandas as pd
import pytest

from yoruba_asr.evaluate.bootstrap import paired_bootstrap, system_ci
from yoruba_asr.evaluate.metrics import (cer, levenshtein, ter_matched_words, tone_error_rate, wer, wer_no_tone,
                                         word_alignment)


def test_wer_cer_known_pairs():
    assert wer(["a b c d"], ["a x c"]) == pytest.approx(50.0)
    assert wer(["a b", "c d"], ["a b", "c d"]) == 0.0
    assert cer(["abcd"], ["abxd"]) == pytest.approx(25.0)
    assert levenshtein("kitten", "sitting") == 3


def test_wer_no_tone():
    assert wer(["ọkọ́ mi"], ["ọkọ̀ mi"]) == pytest.approx(50.0)
    assert wer_no_tone(["ọkọ́ mi"], ["ọkọ̀ mi"]) == 0.0


def test_ter_hand_computed():
    # [M, H] vs [M, L] -> 1 substitution / 2 TBUs
    assert tone_error_rate(["ọkọ́"], ["ọkọ̀"]) == pytest.approx(50.0)
    # àwọn [L, M] vs awọn [M, M] + ń lọ [H, M] vs n lọ [M] (n unmarked is not a TBU -> deletion)
    assert tone_error_rate(["àwọn ń lọ"], ["awọn n lọ"]) == pytest.approx(100 * 2 / 4)


def test_ter_matched_words_and_confusion():
    rate, conf = ter_matched_words(["ọkọ́ mi dé"], ["ọkọ̀ mi lọ"])
    # matched words: ọkọ́/ọkọ̀ (2 TBUs, 1 error) and mi/mi (1 TBU); dé/lọ differ without tones -> excluded
    assert rate == pytest.approx(100 / 3)
    assert conf[0][2] == 1  # H -> L
    assert sum(map(sum, conf)) == 3


def test_alignment_ops():
    ops = [o for o, _, _ in word_alignment("a b c", "a c d")]
    assert sum(o != "equal" for o in ops) == levenshtein("a b c".split(), "a c d".split()) == 2
    ops = word_alignment("a b c", "a c")
    assert ("delete", "b", None) in ops and sum(o == "equal" for o, _, _ in ops) == 2


def _preds(hyps, refs, speakers):
    return pd.DataFrame({"path": [f"u{i}" for i in range(len(refs))], "speaker": speakers,
                         "reference": refs, "hypothesis": hyps})


def test_bootstrap_identical_systems():
    refs = ["a b c", "d e f", "g h", "i j"]
    p = _preds(["a b x", "d e f", "g", "i j"], refs, ["s1", "s1", "s2", "s3"])
    r = paired_bootstrap(p, p, "wer", n=200)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and not r["significant"]


def test_bootstrap_clear_difference():
    refs = [f"w{i} x{i} y{i}" for i in range(20)]
    spk = [f"s{i % 10}" for i in range(20)]
    perfect = _preds(refs, refs, spk)
    bad = _preds(["zz"] * 20, refs, spk)
    r = paired_bootstrap(perfect, bad, "wer", n=500)
    assert r["delta"] == pytest.approx(-100.0) and r["significant"] and r["ci_high"] < 0
    ci = system_ci(bad, "wer", n=100)
    assert ci["ci_low"] == pytest.approx(100.0) and ci["point"] == pytest.approx(100.0)
    assert np.isfinite(paired_bootstrap(perfect, bad, "ter", n=50)["delta"])
