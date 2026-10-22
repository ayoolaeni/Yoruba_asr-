"""10-step continued pretraining with a tiny random Wav2Vec2ForPreTraining."""
import math
from pathlib import Path

import pandas as pd
import pytest

from conftest import smoke_cfg
from yoruba_asr.adapt.cpt import CPTError, load_pretraining_model, run_cpt
from yoruba_asr.modeling import resolve_model_name, tiny_wav2vec2_config
from yoruba_asr.pipeline import _ensure_block


def test_cpt_runs_and_loss_is_finite(prepared, tmp_path):
    fixture_dir, work, _ = prepared
    cfg = smoke_cfg(fixture_dir, work, "cpt")
    _ensure_block(cfg, "cpt", "cpt.yaml")
    unl = pd.read_csv(Path(cfg.paths.manifests_dir) / "unlabelled.csv")
    s = run_cpt(cfg, unl, resolve_model_name(cfg), tmp_path / "enc", tmp_path / "ckpt", tmp_path / "res", resume=True)
    assert s["completed"] and s["completed_steps"] == 10
    log = pd.read_csv(tmp_path / "res" / "cpt_log.csv")
    assert len(log) and all(math.isfinite(v) for v in log["loss"])
    assert (tmp_path / "enc" / "config.json").exists() and (tmp_path / "res" / "cpt_loss.png").exists()


def test_cpt_resumes_from_checkpoint(prepared, tmp_path):
    fixture_dir, work, _ = prepared
    cfg = smoke_cfg(fixture_dir, work, "cpt", {"cpt.max_steps": 6, "cpt.checkpoint_every": 3,
                                                "cpt.time_budget_hours": 0.0})
    _ensure_block(cfg, "cpt", "cpt.yaml")
    unl = pd.read_csv(Path(cfg.paths.manifests_dir) / "unlabelled.csv")
    name = resolve_model_name(cfg)
    first = run_cpt(cfg, unl, name, tmp_path / "enc", tmp_path / "ckpt", tmp_path / "res")
    assert first["stop_reason"] == "session_time_budget" and first["completed_steps"] == 1
    cfg.cpt.time_budget_hours = 1.0
    second = run_cpt(cfg, unl, name, tmp_path / "enc", tmp_path / "ckpt", tmp_path / "res")
    assert second["completed"] and second["completed_steps"] == 6


def test_missing_quantizer_is_refused(tmp_path):
    from transformers import Wav2Vec2ForCTC

    Wav2Vec2ForCTC(tiny_wav2vec2_config()).save_pretrained(tmp_path / "ctc_only")
    c = type("C", (), {"mask_time_prob": 0.65, "mask_time_length": 4, "diversity_loss_weight": 0.1})()
    with pytest.raises(CPTError):
        load_pretraining_model(str(tmp_path / "ctc_only"), c)
