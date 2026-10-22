import shutil
import zipfile

from conftest import smoke_cfg
from yoruba_asr.config import discover_paths, load_config
from yoruba_asr.data.common_voice import load_split
from yoruba_asr.data.generic import find_manifest, load_manifest
from yoruba_asr.data.iroyin import load_iroyin
from yoruba_asr.data.slr86 import load_slr86

COLUMNS = {"path", "sentence", "speaker", "corpus", "split", "gender", "age"}


def test_common_voice_loader(fixture_dir, tmp_path):
    cfg = smoke_cfg(fixture_dir, tmp_path)
    found = discover_paths(cfg, verbose=False)
    train = load_split(found.common_voice, "train")
    assert COLUMNS <= set(train.columns)
    assert len(train) == 20 and set(train["split"]) == {"train"}
    assert set(train["gender"]) <= {"male", "female", ""}
    assert len(load_split(found.common_voice, "other")) == 5  # includes the corrupt clip


def test_slr86_and_manifest_loaders(fixture_dir):
    slr = load_slr86([fixture_dir / "slr86_female"])
    assert len(slr) == 8 and slr["speaker"].str.startswith("yof_").all()
    assert COLUMNS <= set(slr.columns)
    ir = load_iroyin(fixture_dir / "iroyin_speech")
    assert len(ir) == 6 and set(ir["corpus"]) == {"iroyin"}
    assert find_manifest(fixture_dir / "iroyin_speech").name == "manifest.csv"
    m = load_manifest(fixture_dir / "iroyin_speech" / "manifest.csv", corpus="x")
    assert all(p.endswith(".flac") for p in m["path"])


def test_discover_paths_finds_all(fixture_dir, tmp_path):
    found = discover_paths(smoke_cfg(fixture_dir, tmp_path), verbose=False)
    assert found.common_voice is not None and (found.common_voice / "train.tsv").exists()
    assert len(found.slr86) == 1
    assert found.iroyin is not None


def _copy_cv_layout(fixture_dir, inp, move_clips_to=None):
    cv_src = next((fixture_dir / "common_voice").rglob("train.tsv")).parent
    dst = inp / "cv" / "yo"
    shutil.copytree(cv_src, dst)
    if move_clips_to is not None:
        target = inp / move_clips_to
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst / "clips"), str(target))
        (dst / "clips").mkdir()
    return dst


def test_clips_in_another_folder_are_found(fixture_dir, tmp_path):
    from yoruba_asr.pipeline import check_cv_audio

    inp = tmp_path / "input"
    cv_dir = _copy_cv_layout(fixture_dir, inp, move_clips_to="elsewhere/audio_part")
    cfg = smoke_cfg(inp, tmp_path / "w")
    cv = {s: load_split(cv_dir, s) for s in ("train", "dev", "test", "other")}
    report = check_cv_audio(cv, cfg)
    assert report["train"]["remapped"] == 20 and report["train"]["missing"] == 0
    assert all("elsewhere" in p for p in cv["train"]["path"])


def test_missing_clips_give_clear_error(fixture_dir, tmp_path):
    import pytest

    from yoruba_asr.pipeline import MissingAudioError, check_cv_audio

    inp = tmp_path / "input"
    cv_dir = _copy_cv_layout(fixture_dir, inp)
    shutil.rmtree(cv_dir / "clips")
    cfg = smoke_cfg(inp, tmp_path / "w")
    cv = {s: load_split(cv_dir, s) for s in ("train", "dev", "test", "other")}
    with pytest.raises(MissingAudioError, match="COMMON VOICE AUDIO IS MISSING"):
        check_cv_audio(cv, cfg)


def test_discover_extracts_archives(fixture_dir, tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    shutil.copytree(fixture_dir / "common_voice", inp / "cv")
    with zipfile.ZipFile(inp / "yo_ng_female.zip", "w") as zf:
        for f in (fixture_dir / "slr86_female").iterdir():
            zf.write(f, f"yo_ng_female/{f.name}")
    cfg = load_config(overrides={"paths.input_root": str(inp), "paths.work_root": str(tmp_path / "w"),
                                 "paths.tmp_dir": str(tmp_path / "tmp")})
    found = discover_paths(cfg, verbose=False)
    assert found.extracted and len(found.slr86) == 1
    assert str(found.slr86[0]).startswith(str(tmp_path / "tmp"))
