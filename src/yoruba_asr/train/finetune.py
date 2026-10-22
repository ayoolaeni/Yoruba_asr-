"""CTC fine-tuning with the Hugging Face Trainer (resumable, time-budgeted, GPU-time logged)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

from ..config import Config
from ..evaluate.metrics import make_compute_metrics, preprocess_logits
from ..modeling import load_ctc_model
from ..utils import RunInfo, get_logger, has_cuda, load_json, now_iso, save_json, seed_everything
from .callbacks import MetricsLogCallback, TimeBudgetCallback, find_resume_checkpoint
from .collator import DataCollatorCTCWithPadding

log = get_logger(__name__)
COMPLETE_MARKER = "training_complete.json"


class CTCTrainer(Trainer):
    """Trainer that groups by length using the dataset's ``lengths`` attribute (no audio decoding)."""

    def _get_train_sampler(self, *args, **kwargs):
        ds = args[0] if args else kwargs.get("train_dataset")
        ds = ds if ds is not None else self.train_dataset
        if self.args.group_by_length and hasattr(ds, "lengths"):
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        dataset=ds, lengths=list(ds.lengths))
        return super()._get_train_sampler(*args, **kwargs)


def training_arguments(cfg: Config, output_dir: Path, max_steps: int) -> TrainingArguments:
    t = cfg.train
    use_fp16 = bool(t.fp16) and has_cuda()
    eval_steps = min(int(t.eval_steps), int(max_steps))
    save_steps = min(int(t.save_steps), int(max_steps))
    return TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=bool(t.group_by_length), length_column_name="input_length",
        per_device_train_batch_size=int(t.per_device_train_batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        per_device_eval_batch_size=int(t.per_device_eval_batch_size),
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=save_steps,
        logging_steps=int(t.logging_steps), max_steps=int(max_steps),
        learning_rate=float(t.learning_rate), warmup_steps=int(t.warmup_steps), lr_scheduler_type="linear",
        adam_beta1=float(t.adam_beta1), adam_beta2=float(t.adam_beta2), adam_epsilon=float(t.adam_epsilon),
        weight_decay=float(t.weight_decay), optim="adamw_torch",
        fp16=use_fp16, save_total_limit=int(t.save_total_limit),
        load_best_model_at_end=True, metric_for_best_model=t.metric_for_best_model, greater_is_better=False,
        report_to="none", seed=int(cfg.seed), data_seed=int(cfg.seed),
        dataloader_num_workers=int(t.dataloader_num_workers), remove_unused_columns=False,
        save_safetensors=True, disable_tqdm=False, logging_first_step=True,
    )


def is_complete(model_dir: str | Path) -> bool:
    return (Path(model_dir) / COMPLETE_MARKER).exists()


def run_finetune(cfg: Config, *, tag: str, train_ds, eval_ds, processor, init_from: str, mask_time_prob: float,
                 max_steps: int, model_dir: str | Path, results_dir: str | Path, resume: bool = True,
                 previous_outputs: list[Path] | None = None, extra_info: dict | None = None) -> dict:
    """Fine-tune a CTC model and save the best (dev-WER) checkpoint to ``model_dir``.

    Returns ``{"status": "complete"|"incomplete", ...}``. An incomplete run (time budget hit)
    leaves checkpoints in ``checkpoints/<tag>`` so the next session can resume.
    """
    model_dir, results_dir = Path(model_dir), Path(results_dir)
    if resume and is_complete(model_dir):
        log.info("[%s] already complete at %s - skipping training", tag, model_dir)
        return {"status": "complete", "skipped": True, **(load_json(model_dir / COMPLETE_MARKER) or {})}

    seed_everything(int(cfg.seed))
    work_dir = Path(cfg.paths.work_dir)
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / tag
    rel = (Path(cfg.paths.checkpoints_dir) / tag).relative_to(work_dir) if \
        str(Path(cfg.paths.checkpoints_dir)).startswith(str(work_dir)) else Path("checkpoints") / tag
    resume_ckpt = find_resume_checkpoint(ckpt_dir, previous_outputs or [], str(rel)) if resume else None
    if not resume and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    model, load_info = load_ctc_model(init_from, cfg, processor, mask_time_prob)
    args = training_arguments(cfg, ckpt_dir, max_steps)
    budget = TimeBudgetCallback(float(cfg.train.time_budget_hours))
    trainer = CTCTrainer(
        model=model, args=args, data_collator=DataCollatorCTCWithPadding(processor),
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=processor,
        compute_metrics=make_compute_metrics(processor.tokenizer),
        preprocess_logits_for_metrics=preprocess_logits, callbacks=[budget, MetricsLogCallback()],
    )
    run = RunInfo(results_dir / "run_info.json", stage=tag)
    start_step = 0
    if resume_ckpt is not None:
        log.info("[%s] resuming from %s", tag, resume_ckpt)
        start_step = int((load_json(resume_ckpt / "trainer_state.json") or {}).get("global_step", 0))
    log.info("[%s] training: %d clips, max_steps=%d, mask_time_prob=%.3f, init=%s",
             tag, len(train_ds), max_steps, mask_time_prob, init_from)
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    state = trainer.state
    complete = state.global_step >= max_steps and not budget.triggered
    history = pd.DataFrame(state.log_history)
    results_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(results_dir / "train_log.csv", index=False)
    info = run.finish(steps=state.global_step - start_step, global_step=state.global_step,
                      train_clips=len(train_ds), max_steps=max_steps, mask_time_prob=mask_time_prob,
                      init_from=str(init_from), status="complete" if complete else "incomplete")
    summary = {
        "tag": tag, "status": "complete" if complete else "incomplete", "global_step": state.global_step,
        "max_steps": max_steps, "best_dev_wer": state.best_metric, "best_checkpoint": state.best_model_checkpoint,
        "mask_time_prob": mask_time_prob, "init_from": str(init_from), "train_clips": len(train_ds),
        "gpu_hours_total": info["total_gpu_hours"], "finished_at": now_iso(),
        "newly_initialised": load_info.get("missing_keys", []), **(extra_info or {}),
    }
    if not complete:
        save_json(summary, results_dir / "run_status.json")
        log.warning("[%s] stopped at step %d/%d (time budget). Run the same stage again with RESUME=True.",
                    tag, state.global_step, max_steps)
        return summary

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))  # best checkpoint (load_best_model_at_end=True)
    processor.save_pretrained(str(model_dir))
    save_json(summary, model_dir / COMPLETE_MARKER)
    save_json(summary, results_dir / "run_status.json")
    if not cfg.train.get("keep_checkpoints", False):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    log.info("[%s] done: best dev WER=%s -> %s", tag, state.best_metric, model_dir)
    del trainer, model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return summary
