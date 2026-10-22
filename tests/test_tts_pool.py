"""TTS sentence pool: found when Kaggle unpacks validated_sentences.tsv elsewhere; rebuilt if empty."""
import shutil
from pathlib import Path

import pandas as pd

from conftest import smoke_cfg
from yoruba_asr.data.common_voice import load_sentence_pool
from yoruba_asr.data.leakage import leakage_key


def _cv_copy(fixture_dir, inp):
    src = next((fixture_dir / "common_voice").rglob("train.tsv")).parent
    dst = inp / "cv" / "yo"
    shutil.copytree(src, dst)
    return dst


def test_pool_found_in_another_folder(fixture_dir, tmp_path):
    inp = tmp_path / "input"
    cv_dir = _cv_copy(fixture_dir, inp)
    (inp / "elsewhere").mkdir()
    shutil.move(str(cv_dir / "validated_sentences.tsv"), str(inp / "elsewhere" / "validated_sentences.tsv"))
    assert len(load_sentence_pool(cv_dir)) > 0  # falls back to validated.tsv sentences
    pool = load_sentence_pool(cv_dir, [inp])
    assert len(pool) == 28 and "elsewhere" in pool.attrs["pool_source"]  # 10 train + 4 dev + 4 test + 10 extra


def test_augment_rebuilds_empty_tts_list(fixture_dir, tmp_path):
    from yoruba_asr.pipeline import make_ctx, rebuild_tts_sentences, stage_prepare

    inp = tmp_path / "input"
    _cv_copy(fixture_dir, inp)
    cfg = smoke_cfg(inp, tmp_path / "work")
    stage_prepare(cfg, resume=False)
    man = Path(cfg.paths.manifests_dir)
    pd.DataFrame(columns=["sentence", "text_tone", "text_notone"]).to_csv(man / "tts_sentences.csv", index=False)
    ctx = make_ctx(cfg)
    train = pd.read_csv(man / "cv_train.csv")
    tts = rebuild_tts_sentences(ctx, train)
    assert len(tts) > 0 and len(pd.read_csv(man / "tts_sentences.csv")) == len(tts)
    dev, test = pd.read_csv(man / "cv_dev.csv"), pd.read_csv(man / "cv_test.csv")
    forbidden = {leakage_key(s) for s in pd.concat([dev["sentence"], test["sentence"]])}
    assert not tts["sentence"].map(leakage_key).isin(forbidden).any()
