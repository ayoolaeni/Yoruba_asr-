import json
from pathlib import Path

import pandas as pd
import pytest

from yoruba_asr.data.leakage import (LeakageError, assert_no_eval_audio, forbidden_sentences, leakage_key,
                                     remove_forbidden)


def _df(sentences, paths=None, corpus="x"):
    return pd.DataFrame({"sentence": sentences, "path": paths or [f"/a/{i}.wav" for i in range(len(sentences))],
                         "corpus": corpus})


def test_dev_test_sentences_removed_tone_insensitively():
    dev, test = _df(["Ọ̀rẹ́ mi dé."]), _df(["Ajá ń gbó"])
    forbidden = forbidden_sentences(dev, test)
    ood = _df(["ọrẹ mi de", "Ìlú wa dára", "AJÁ Ń GBÓ!"])
    kept, n = remove_forbidden(ood, forbidden)
    assert n == 2 and kept["sentence"].tolist() == ["Ìlú wa dára"]


def test_eval_audio_never_in_pools():
    dev, test = _df(["a"], ["/cv/clips/dev1.mp3"]), _df(["b"], ["/cv/clips/test1.mp3"])
    with pytest.raises(LeakageError):
        assert_no_eval_audio({"unlabelled": _df(["x"], ["/cv/clips/test1.mp3"])}, dev, test)
    assert assert_no_eval_audio({"unlabelled": _df(["x"], ["/cv/clips/other.mp3"])}, dev, test) == {"unlabelled": 0}


def test_prepared_pools_are_clean(prepared):
    _, work, cfg = prepared
    results = Path(cfg.paths.results_dir)
    report = json.loads((results / "leakage_report.json").read_text(encoding="utf-8"))
    assert report["removed_by_corpus"]["iroyin"] == 1  # the planted dev sentence
    assert all(v == 0 for v in report["cv_dev_test_audio_in_pools"].values())
    man = Path(cfg.paths.manifests_dir)
    dev = pd.read_csv(man / "cv_dev.csv")
    test = pd.read_csv(man / "cv_test.csv")
    forbidden = {leakage_key(s) for s in pd.concat([dev["sentence"], test["sentence"]])}
    for name in ("ood_train", "tts_sentences"):
        df = pd.read_csv(man / f"{name}.csv")
        assert not df["sentence"].map(leakage_key).isin(forbidden).any(), name
    unl = pd.read_csv(man / "unlabelled.csv")
    assert not set(unl["path"]) & (set(dev["path"]) | set(test["path"]))
