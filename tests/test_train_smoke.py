"""10-step fine-tune with a tiny random Wav2Vec2ForCTC: runs, saves, resumes and evaluates."""
from pathlib import Path

from conftest import smoke_cfg
from yoruba_asr.evaluate.decode import decode_cached, evaluate_predictions, load_model_and_processor
from yoruba_asr.modeling import resolve_model_name
from yoruba_asr.pipeline import _cv_cached, make_ctx
from yoruba_asr.train.callbacks import latest_checkpoint
from yoruba_asr.train.datasets import load_cached_split
from yoruba_asr.train.finetune import is_complete, run_finetune
from yoruba_asr.vocab import load_shared_processor


def test_finetune_save_resume_evaluate(prepared, tmp_path):
    fixture_dir, work, _ = prepared
    cfg = smoke_cfg(fixture_dir, work, "V1", {"train.keep_checkpoints": True, "train.eval_steps": 3,
                                              "train.save_steps": 3})
    ctx = make_ctx(cfg, discover=False)
    proc = load_shared_processor(cfg.paths.shared_dir, cfg.paths.results_dir)
    train, dev = _cv_cached(ctx, "train"), _cv_cached(ctx, "dev")
    init = resolve_model_name(cfg)

    first = run_finetune(cfg, tag="test_ft", train_ds=train, eval_ds=dev, processor=proc, init_from=init,
                         mask_time_prob=0.05, max_steps=6, model_dir=tmp_path / "m1", results_dir=tmp_path / "r1")
    assert first["status"] == "complete" and is_complete(tmp_path / "m1")
    ckpt = latest_checkpoint(Path(cfg.paths.checkpoints_dir) / "test_ft")
    assert ckpt is not None and ckpt.name == "checkpoint-6"

    second = run_finetune(cfg, tag="test_ft", train_ds=train, eval_ds=dev, processor=proc, init_from=init,
                          mask_time_prob=0.05, max_steps=10, model_dir=tmp_path / "m2", results_dir=tmp_path / "r2")
    assert second["status"] == "complete" and second["global_step"] == 10
    assert (tmp_path / "r2" / "run_info.json").exists()

    model, processor, device = load_model_and_processor(tmp_path / "m2", "cpu")
    preds = decode_cached(model, processor, load_cached_split(Path(cfg.paths.cache_dir) / "cv_test"), device, 4)
    m = evaluate_predictions(preds, "test", tmp_path / "eval")
    assert m["n_utts"] == len(preds) == 8
    assert {"wer", "cer", "wer_no_tone", "ter", "tone_confusion"} <= set(m)
