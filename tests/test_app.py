import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
gr = pytest.importorskip("gradio")

import app as webapp  # noqa: E402


def test_app_builds_without_models(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  V1: nowhere/V1\nresults_dir: " + str(tmp_path / "none").replace("\\", "/") + "\n",
                   encoding="utf-8")
    demo = webapp.build_demo(cfg)
    assert isinstance(demo, gr.Blocks)


def test_results_missing_files_handled(tmp_path):
    res = webapp.load_results(tmp_path / "does_not_exist")
    assert res["tables"] == {} and res["figures"] == [] and "No results yet" in res["message"]


def test_app_transcribes_with_smoke_model(trained_model, tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(f"models:\n  V1: '{trained_model.as_posix()}'\ndefault_model: V1\nresults_dir: '{tmp_path.as_posix()}'\n",
                   encoding="utf-8")
    settings = webapp.load_settings(cfg)
    reg = webapp.available_models(settings)
    assert list(reg) == ["V1"]
    cache = webapp.ModelCache(reg, 2)
    sr = 16000
    r = cache.get("V1").transcribe_array((0.2 * np.sin(np.arange(sr * 2) / 3)).astype(np.float32), sr)
    assert isinstance(r["text"], str)
    assert "Reference" in webapp.diff_html("ọmọ mi dé", "ọmọ mi lọ")
    assert webapp.clip_metrics("Ọmọ mi dé.", "ọmọ mi dé")["WER %"] == 0.0
    assert isinstance(webapp.build_demo(cfg), gr.Blocks)
