"""Continued self-supervised pretraining of XLS-R on unlabelled Yoruba audio (wav2vec 2.0 objective).

Modelled on Hugging Face's ``run_wav2vec2_pretraining_no_trainer.py``: contrastive loss + diversity
loss, Gumbel temperature decay, AdamW with linear warm-up/decay, FP16 via ``accelerate``.
Checkpoints (model, optimiser, scheduler, scaler, RNG, step, temperature) every N steps; exact resume.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import modeling as _modeling  # noqa: F401  (installs the gradient-checkpointing warning filter)
from ..audio import SR, safe_load_trimmed
from ..utils import RunInfo, get_logger, load_json, now_iso, save_json, seed_everything

log = get_logger(__name__)
QUANTIZER_KEYS = ("quantizer.", "project_q", "project_hid")


class CPTError(RuntimeError):
    pass


class UnlabelledDataset(torch.utils.data.Dataset):
    """Unlabelled audio, random-cropped to ``max_crop`` seconds (crop seeded by seed/epoch/index)."""

    def __init__(self, manifest: pd.DataFrame, feature_extractor, max_crop_s: float, seed: int, top_db: float = 30):
        self.paths = manifest["path"].astype(str).tolist()
        self.durations = manifest["duration"].astype(float).tolist()
        self.fe = feature_extractor
        self.max_len = int(max_crop_s * SR)
        self.seed = seed
        self.top_db = top_db
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        x = safe_load_trimmed(self.paths[i], SR, self.top_db)
        if x is None or len(x) < SR // 2:
            x = np.zeros(SR, dtype=np.float32) if x is None else np.pad(x, (0, SR // 2 - len(x)))
        if len(x) > self.max_len:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            start = int(rng.integers(0, len(x) - self.max_len + 1))
            x = x[start: start + self.max_len]
        iv = self.fe(x, sampling_rate=SR).input_values[0].astype(np.float32)
        return {"input_values": iv}


def epoch_batches(durations: list[float], batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Shuffle, then sort within chunks of 50 batches by duration (less padding), then shuffle batches."""
    rng = np.random.default_rng([seed, epoch, 4242])
    perm = rng.permutation(len(durations))
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(perm), chunk):
        part = sorted(perm[s: s + chunk], key=lambda k: -durations[k])
        batches += [list(map(int, part[j: j + batch_size])) for j in range(0, len(part), batch_size)]
    order = rng.permutation(len(batches))
    return [batches[k] for k in order]


class PretrainingCollator:
    def __init__(self, model, feature_extractor, mask_time_prob: float, mask_time_length: int):
        self.model, self.fe = model, feature_extractor
        self.mask_time_prob, self.mask_time_length = mask_time_prob, mask_time_length

    def __call__(self, features: list[dict]) -> dict:
        from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

        batch = self.fe.pad(features, padding=True, return_tensors="pt")
        bsz, n_samples = batch["input_values"].shape
        seq_len = int(self.model._get_feat_extract_output_lengths(n_samples))
        sub_mask = self.model._get_feature_vector_attention_mask(seq_len, batch["attention_mask"]) \
            if batch.get("attention_mask") is not None else None
        mask = _compute_mask_indices((bsz, seq_len), self.mask_time_prob, self.mask_time_length,
                                     attention_mask=sub_mask, min_masks=2)
        neg = _sample_negative_indices((bsz, seq_len), self.model.config.num_negatives, mask_time_indices=mask)
        batch["mask_time_indices"] = torch.tensor(mask, dtype=torch.bool)
        batch["sampled_negative_indices"] = torch.tensor(neg, dtype=torch.long)
        if sub_mask is not None:
            batch["sub_attention_mask"] = sub_mask
        return batch


def load_pretraining_model(name: str, c):
    """Load ``Wav2Vec2ForPreTraining`` and refuse to continue if the quantiser was not in the checkpoint."""
    from transformers import Wav2Vec2ForPreTraining

    model, info = Wav2Vec2ForPreTraining.from_pretrained(
        name, output_loading_info=True, mask_time_prob=float(c.mask_time_prob),
        mask_time_length=int(c.mask_time_length), diversity_loss_weight=float(c.diversity_loss_weight),
        mask_feature_prob=0.0, layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0, feat_proj_dropout=0.0)
    missing = [k for k in info.get("missing_keys", []) if any(q in k for q in QUANTIZER_KEYS)]
    if missing:
        raise CPTError(f"Quantiser/projection weights are newly initialised for {name}: {missing}. "
                       "CPT must not run with a random quantiser - see docs/OPEN_QUESTIONS.md.")
    return model, info


def _lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def run_cpt(cfg, unlabelled: pd.DataFrame, model_name: str, out_dir: str | Path, ckpt_dir: str | Path,
            results_dir: str | Path, resume: bool = True, time_budget_hours: float | None = None) -> dict:
    """Run (or resume) CPT. Exports the encoder to ``out_dir`` whenever it stops."""
    from accelerate import Accelerator

    from ..vocab import make_feature_extractor

    c = cfg.cpt
    out_dir, ckpt_dir, results_dir = Path(out_dir), Path(ckpt_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "cpt_state.json"
    final_info = load_json(out_dir / "cpt_complete.json")
    if resume and final_info and final_info.get("completed"):
        log.info("CPT already completed: %s", out_dir)
        return final_info

    seed_everything(int(cfg.seed))
    cuda = torch.cuda.is_available()
    accelerator = Accelerator(mixed_precision="fp16" if (cuda and c.fp16) else "no", cpu=not cuda)
    fe = make_feature_extractor()
    model, load_info = load_pretraining_model(model_name, c)
    if c.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if c.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(c.learning_rate), betas=(float(c.adam_beta1), float(c.adam_beta2)),
                                  eps=float(c.adam_epsilon), weight_decay=float(c.weight_decay))
    max_steps = int(c.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(int(c.warmup_steps), max_steps))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    raw_model = accelerator.unwrap_model(model)

    st = {"completed_steps": 0, "epoch": 0, "batch_in_epoch": 0, "total_hours": 0.0, "low_ppl_streak": 0,
          "gumbel_temperature": float(c.max_gumbel_temperature), "aborted": None}
    if resume and state_path.exists():
        accelerator.load_state(str(ckpt_dir / "state"))
        st.update(load_json(state_path))
        log.info("CPT: resumed at step %d (epoch %d, batch %d)", st["completed_steps"], st["epoch"], st["batch_in_epoch"])
    elif not resume and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    raw_model.set_gumbel_temperature(st["gumbel_temperature"])

    dataset = UnlabelledDataset(unlabelled, fe, float(c.max_crop_seconds), int(cfg.seed))
    collator = PretrainingCollator(raw_model, fe, float(c.mask_time_prob), int(c.mask_time_length))
    bs, ga = int(c.per_device_batch_size), int(c.gradient_accumulation_steps)
    budget_s = float(time_budget_hours or c.time_budget_hours) * 3600
    log_path = results_dir / "cpt_log.csv"
    log_rows = []
    if resume and log_path.exists():
        try:
            log_rows = pd.read_csv(log_path).to_dict("records")
        except pd.errors.EmptyDataError:  # stopped before the first log step
            log_rows = []
    run = RunInfo(results_dir / "cpt_run_info.json", stage="cpt")
    t0, steps_at_start = time.time(), st["completed_steps"]
    hours_at_start = st["total_hours"]
    acc = {"loss": 0.0, "contrastive": 0.0, "diversity": 0.0, "ppl": 0.0, "n": 0}
    stop_reason = None

    def save_checkpoint():
        st["total_hours"] = hours_at_start + (time.time() - t0) / 3600
        accelerator.save_state(str(ckpt_dir / "state"))
        save_json(st, state_path)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

    while stop_reason is None:
        dataset.epoch = st["epoch"]
        batches = epoch_batches(dataset.durations, bs, int(cfg.seed), st["epoch"])
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches[st["batch_in_epoch"]:], collate_fn=collator,
            num_workers=int(c.get("num_workers", 0)))
        micro = 0
        for batch in loader:
            st["batch_in_epoch"] += 1
            batch.pop("sub_attention_mask", None)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            num_losses = batch["mask_time_indices"].sum().clamp(min=1)
            with accelerator.autocast():
                out = model(**batch)
            loss = out.loss / num_losses / ga
            accelerator.backward(loss)
            acc["loss"] += float(out.loss.detach() / num_losses)
            acc["contrastive"] += float(out.contrastive_loss.detach() / num_losses)
            acc["diversity"] += float(out.diversity_loss.detach() / num_losses)  # HF scales it by #masked
            acc["ppl"] += float(out.codevector_perplexity.detach())
            acc["n"] += 1
            micro += 1
            if micro % ga != 0:
                continue
            grad_norm = accelerator.clip_grad_norm_(params, float(c.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            st["completed_steps"] += 1
            st["gumbel_temperature"] = max(float(c.max_gumbel_temperature) *
                                           float(c.gumbel_temperature_decay) ** st["completed_steps"],
                                           float(c.min_gumbel_temperature))
            raw_model.set_gumbel_temperature(st["gumbel_temperature"])
            step = st["completed_steps"]
            mean_loss = acc["loss"] / max(acc["n"], 1)
            mean_ppl = acc["ppl"] / max(acc["n"], 1)

            # Stability checks: NaN loss or collapsed codebook usage.
            if not math.isfinite(mean_loss):
                st["aborted"] = f"NaN/inf loss at step {step}"
            st["low_ppl_streak"] = st["low_ppl_streak"] + 1 if mean_ppl < float(c.perplexity_floor) else 0
            if st["low_ppl_streak"] >= int(c.perplexity_patience):
                st["aborted"] = f"codevector perplexity < {c.perplexity_floor} for {c.perplexity_patience} steps"

            if step % int(c.log_every) == 0 or st["aborted"]:
                row = {"step": step, "loss": mean_loss, "contrastive_loss": acc["contrastive"] / max(acc["n"], 1),
                       "diversity_loss": acc["diversity"] / max(acc["n"], 1), "codevector_perplexity": mean_ppl,
                       "gumbel_temperature": st["gumbel_temperature"],
                       "grad_norm": float(grad_norm) if grad_norm is not None else None,
                       "lr": scheduler.get_last_lr()[0], "elapsed_h": (time.time() - t0) / 3600}
                log_rows.append(row)
                log.info("CPT step %d | loss %.4f | ppl %.1f | temp %.3f | gn %s", step, mean_loss, mean_ppl,
                         st["gumbel_temperature"], row["grad_norm"])
            acc = {k: 0.0 if k != "n" else 0 for k in acc}
            total_h = hours_at_start + (time.time() - t0) / 3600
            if st["aborted"]:
                stop_reason = "aborted"
            elif step >= max_steps:
                stop_reason = "max_steps"
            elif total_h >= float(c.max_total_hours):
                stop_reason = "max_total_hours"
            elif time.time() - t0 > budget_s:
                stop_reason = "session_time_budget"
            if stop_reason or step % int(c.checkpoint_every) == 0:
                save_checkpoint()
            if stop_reason:
                break
        else:
            st["epoch"] += 1
            st["batch_in_epoch"] = 0
            if micro % ga != 0:
                optimizer.zero_grad(set_to_none=True)  # drop a partial accumulation at epoch end
        if len(dataset) == 0:
            raise CPTError("Unlabelled pool is empty")

    completed = stop_reason in ("max_steps", "max_total_hours")
    if st["aborted"]:
        log.error("CPT aborted: %s (checkpoint saved)", st["aborted"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(str(out_dir))
    fe.save_pretrained(str(out_dir))
    _plot_log(log_rows, results_dir / "cpt_loss.png")
    info = run.finish(steps=st["completed_steps"] - steps_at_start, global_step=st["completed_steps"],
                      stop_reason=stop_reason)
    summary = {"completed": completed, "stop_reason": stop_reason, "aborted": st["aborted"],
               "completed_steps": st["completed_steps"], "max_steps": max_steps,
               "total_hours": round(hours_at_start + (time.time() - t0) / 3600, 3),
               "unlabelled_clips": len(dataset), "unlabelled_hours": round(float(unlabelled["duration"].sum()) / 3600, 2),
               "gpu_hours_total": info["total_gpu_hours"], "finished_at": now_iso(),
               "missing_keys_on_load": load_info.get("missing_keys", [])}
    save_json(summary, out_dir / "cpt_complete.json")
    save_json(summary, results_dir / "cpt_status.json")
    if completed and not cfg.train.get("keep_checkpoints", False):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return summary


def _plot_log(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(df["step"], df["loss"], "k-", label="total loss")
    ax1.plot(df["step"], df["contrastive_loss"], "k--", label="contrastive")
    ax1.set_xlabel("update step"); ax1.set_ylabel("loss per masked frame")
    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["codevector_perplexity"], color="0.5", linestyle=":", label="codevector perplexity")
    ax2.set_ylabel("perplexity")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")
    ax1.set_title("Continued pretraining (CPT)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
