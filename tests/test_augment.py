import numpy as np

from yoruba_asr.augment.build_augmented import clip_rng
from yoruba_asr.augment.transforms import (NoiseBank, RIRBank, apply_chain, choose_chain, gain, median_f0, reverb,
                                           speed, tempo)
from yoruba_asr.augment.tts import DummyTTS, coverage_report, sentence_representable

SR = 16000
CFG = {"tempo_factors": [0.9, 1.1], "speed_factors": [0.9, 1.1], "snr_db": [10, 30], "gain_db": [-6, 6],
       "rt60": [0.2, 0.8], "transform_prob": 0.5}


def tone(f0=200.0, seconds=1.5):
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_augmentation_is_deterministic():
    x = tone()
    rirs = RIRBank.build(2, 42)
    outs = []
    for _ in range(2):
        rng = clip_rng(42, 3, 1)
        chain = choose_chain(rng, ["tempo", "noise", "reverb", "gain"])
        y, applied = apply_chain(x, chain, rng, CFG, NoiseBank([]), rirs)
        outs.append((y, applied))
    assert outs[0][1] == outs[1][1]
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_chain_never_empty():
    for i in range(50):
        assert choose_chain(np.random.default_rng(i), ["tempo", "noise"], p=0.0)


def test_tempo_keeps_pitch_speed_changes_it():
    x = tone(200.0)
    f_orig = median_f0(x)
    for f in (0.9, 1.1):
        assert abs(median_f0(tempo(x, f)) - f_orig) / f_orig < 0.03
        assert abs(len(tempo(x, f)) - len(x) / f) < 0.05 * len(x)
    assert abs(median_f0(speed(x, 1.1)) - 1.1 * f_orig) / f_orig < 0.03


def test_reverb_and_gain_shapes():
    x = tone()
    rir = RIRBank.build(1, 1).rirs[0]
    assert len(reverb(x, rir)) == len(x)
    assert np.max(np.abs(gain(x, 12.0))) <= 0.99 + 1e-6


def test_tts_coverage_check():
    vocab = set("abdefghijklmnoprstuwy |ẹọṣàáèéìíòóùúńǹ") | {"̀", "́"}
    ok = coverage_report(["ọ̀rẹ́ mi ń lọ"], vocab)
    assert ok["enabled_effective"] and ok["tone_marks_representable"]
    no_tones = coverage_report(["ọ̀rẹ́ mi ń lọ"], set("abdefghijklmnoprstuwy |ẹọṣ"))
    assert not no_tones["enabled_effective"] and not no_tones["tone_marks_representable"]
    assert len(DummyTTS().synthesize("bàbá", 1)) > SR // 10


def test_one_rare_character_does_not_disable_tts():
    # Real MMS-TTS Yoruba vocabulary: no ḿ, no combining dot below (found on Kaggle, 2026-10-21).
    vocab = set(" '-abdefghijklmnoprstuwy|àáèéìíòóùúńǹṣẹọ—") | {"̀", "́", "<unk>"}
    sentences = ["ọ̀rẹ́ mi ń lọ"] * 50 + ["ḿbọ̀ wá"]
    rep = coverage_report(sentences, vocab)
    assert rep["enabled_effective"] and rep["share_of_sentences_representable"] > 0.9
    assert not sentence_representable("ḿbọ̀ wá", vocab, rep["normal_form"])
