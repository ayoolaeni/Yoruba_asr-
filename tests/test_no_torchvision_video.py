"""Regression test for the Kaggle failure: torchvision without ``io.VideoReader`` must not break data loading."""
import sys
import types

import numpy as np


def test_cached_dataset_works_without_videoreader(prepared, monkeypatch):
    fake_tv = types.ModuleType("torchvision")
    fake_io = types.ModuleType("torchvision.io")  # deliberately has no VideoReader
    fake_tv.io = fake_io
    monkeypatch.setitem(sys.modules, "torchvision", fake_tv)
    monkeypatch.setitem(sys.modules, "torchvision.io", fake_io)

    from yoruba_asr.pipeline import _cv_cached, make_ctx

    _, _, cfg = prepared
    ds = _cv_cached(make_ctx(cfg, discover=False), "train")
    item = ds[0]
    assert item["input_values"].dtype == np.float32 and item["input_values"].ndim == 1
    assert isinstance(item["labels"], list) and all(isinstance(x, int) for x in item["labels"])
    assert len(item["input_values"]) == ds.lengths[0]
