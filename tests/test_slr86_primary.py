"""OpenSLR 86 as the main corpus: annotation cleaning, speaker-disjoint split, end-to-end prepare."""
import json
from pathlib import Path

import pandas as pd

from conftest import smoke_cfg
from yoruba_asr.data.leakage import leakage_key
from yoruba_asr.data.slr86 import clean_annotations, load_slr86, speaker_split


def test_annotation_tags_removed():
    assert clean_annotations("[breath] Ọmọ náà [snap] ń lọ.") == ("Ọmọ náà ń lọ.", ["[breath]", "[snap]"])
    assert clean_annotations("Ìlú wa dára [abrupt]")[1] == ["[abrupt]"]


def test_abrupt_recordings_dropped(tmp_path, fixture_dir):
    import shutil

    folder = tmp_path / "yo_ng_female"
    shutil.copytree(fixture_dir / "slr86_female", folder)
    lines = (folder / "line_index.tsv").read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0] + " [abrupt]"
    lines[1] = lines[1].replace("\t", "\t[breath] ")
    (folder / "line_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = load_slr86([folder])
    assert len(df) == 7 and not df["sentence"].str.contains(r"\[").any()


def test_speaker_split_is_disjoint():
    rows = [{"path": f"{s}_{i}.wav", "sentence": f"gbólóhùn {i % 30}", "speaker": s,
             "gender": "female" if s < "s10" else "male"} for s in [f"s{k:02d}" for k in range(20)] for i in range(10)]
    parts = speaker_split(pd.DataFrame(rows), 0.1, 0.1, seed=42, key=leakage_key)
    spk = {k: set(v["speaker"]) for k, v in parts.items()}
    assert not (spk["train"] & spk["dev"]) and not (spk["train"] & spk["test"]) and not (spk["dev"] & spk["test"])
    assert all(len(parts[k]) for k in ("dev", "test"))
    held = set(parts["dev"]["sentence"]) | set(parts["test"]["sentence"])
    assert not set(parts["train"]["sentence"]) & held
    again = speaker_split(pd.DataFrame(rows), 0.1, 0.1, seed=42, key=leakage_key)
    assert set(again["test"]["speaker"]) == spk["test"]  # deterministic


def test_prepare_with_slr86_as_primary(fixture_dir, tmp_path):
    from yoruba_asr.pipeline import stage_prepare

    import shutil

    inp = tmp_path / "input"
    folder = inp / "yo_ng_female"
    folder.mkdir(parents=True)
    src = fixture_dir / "slr86_female"
    lines = []
    for k in range(6):  # 6 speakers x 8 recordings, built from the fixture audio
        for line in (src / "line_index.tsv").read_text(encoding="utf-8").splitlines():
            fid, sent = line.split("\t")
            new = f"yof_{k:05d}_{fid.split('_')[2]}"
            shutil.copy(src / f"{fid}.wav", folder / f"{new}.wav")
            lines.append(f"{new}\t{sent} {['ọ̀rẹ́', 'ilé', 'ọjà', 'odò', 'igi', 'omi'][k]}")
    (folder / "line_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    work = tmp_path / "work"
    cfg = smoke_cfg(inp, work, extra={"data.primary": "auto", "data.slr86.split.dev_share": 0.25,
                                      "data.slr86.split.test_share": 0.25})
    summary = stage_prepare(cfg, resume=False)
    assert summary["primary_dataset"].startswith("OpenSLR 86")
    assert summary["cv_train"] > 0 and summary["cv_dev"] > 0 and summary["cv_test"] > 0
    leak = json.loads((Path(cfg.paths.results_dir) / "leakage_report.json").read_text(encoding="utf-8"))
    assert leak["speaker_overlap_ok"]
