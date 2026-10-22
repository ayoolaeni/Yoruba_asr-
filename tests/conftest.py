"""Shared pytest fixtures: a tiny synthetic dataset and a prepared work directory (CPU, offline)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from make_tiny_fixture import make_fixture  # noqa: E402


def smoke_cfg(fixture_dir: Path, work: Path, stage: str = "prepare", extra: dict | None = None):
    from yoruba_asr.pipeline import stage_config

    ov = {"paths.input_root": str(fixture_dir), "paths.work_root": str(work), "paths.tmp_dir": str(work / "tmp"),
          **(extra or {})}
    return stage_config(stage, ov, smoke=True)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    return make_fixture(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def prepared(fixture_dir, tmp_path_factory):
    """Run the 'prepare' stage once on the fixture; returns (fixture_dir, work_root, cfg)."""
    from yoruba_asr.pipeline import stage_prepare

    work = tmp_path_factory.mktemp("work")
    cfg = smoke_cfg(fixture_dir, work)
    stage_prepare(cfg, resume=False)
    return fixture_dir, work, cfg


@pytest.fixture(scope="session")
def trained_model(prepared):
    """A tiny V1 model trained for a few steps (shared by inference/app tests)."""
    from yoruba_asr.pipeline import stage_train

    fixture_dir, work, _ = prepared
    cfg = smoke_cfg(fixture_dir, work, "V1", {"train.max_steps": 4, "train.eval_steps": 2, "train.save_steps": 2})
    stage_train(cfg, resume=True, variant="V1")
    return Path(cfg.paths.models_dir) / "V1"
