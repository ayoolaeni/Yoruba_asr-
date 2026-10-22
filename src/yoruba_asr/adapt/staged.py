"""Staged fine-tuning (Stage B): B1 on out-of-domain labelled speech, then B2 on Common Voice.

B1 is identical for V3 and V4 (same CPT encoder, same OOD data, same settings), so it is trained
once into ``models/B1`` and reused.
"""
from __future__ import annotations

from pathlib import Path

from ..train.finetune import is_complete, run_finetune
from ..utils import get_logger, load_json

log = get_logger(__name__)


def run_stage_b1(cfg, *, ood_train, cv_dev, processor, cpt_dir: Path, models_dir: Path, results_dir: Path,
                 resume: bool, previous_outputs) -> dict:
    """Fine-tune the CPT encoder on OOD labelled data; select the best checkpoint on CV dev."""
    b1_dir = models_dir / "B1"
    if ood_train is None or len(ood_train) == 0:
        log.warning("No OOD labelled corpus available: stage B1 skipped (V3 = CPT + CV fine-tuning).")
        return {"stage_b1": "skipped", "init_for_b2": str(cpt_dir)}
    if resume and is_complete(b1_dir):
        return {"stage_b1": "complete", "init_for_b2": str(b1_dir), **(load_json(b1_dir / "training_complete.json") or {})}
    summary = run_finetune(
        cfg, tag="B1", train_ds=ood_train, eval_ds=cv_dev, processor=processor, init_from=str(cpt_dir),
        mask_time_prob=float(cfg.model.mask_time_prob), max_steps=int(cfg.adapt.b1_max_steps), model_dir=b1_dir,
        results_dir=results_dir / "B1", resume=resume, previous_outputs=previous_outputs,
        extra_info={"stage": "B1", "data": "OOD labelled (ÌròyìnSpeech + SLR86)"})
    if summary["status"] != "complete":
        return {"stage_b1": "incomplete", **summary}
    return {"stage_b1": "complete", "init_for_b2": str(b1_dir), **summary}
