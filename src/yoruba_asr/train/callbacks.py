"""Trainer callbacks and checkpoint/resume helpers."""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from transformers import TrainerCallback

from ..utils import get_logger

log = get_logger(__name__)


class TimeBudgetCallback(TrainerCallback):
    """Stops training gracefully (and saves a checkpoint) before Kaggle's 12-hour limit."""

    def __init__(self, hours: float):
        self.budget = float(hours) * 3600
        self.start = time.time()
        self.triggered = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start > self.budget and not self.triggered:
            log.warning("Time budget of %.2f h reached at step %d: saving and stopping.",
                        self.budget / 3600, state.global_step)
            self.triggered = True
            control.should_save = True
            control.should_training_stop = True
        return control


class MetricsLogCallback(TrainerCallback):
    """Prints dev metrics in a compact form."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            keys = [k for k in ("eval_wer", "eval_cer", "eval_wer_no_tone", "eval_loss") if k in metrics]
            log.info("step %d | %s", state.global_step, " ".join(f"{k}={metrics[k]:.4f}" for k in keys))


_CKPT = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(*dirs: str | Path) -> Path | None:
    """Most advanced ``checkpoint-N`` folder (with a trainer_state.json) across ``dirs``."""
    best: tuple[int, Path] | None = None
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for sub in d.iterdir():
            m = _CKPT.match(sub.name)
            if m and sub.is_dir() and (sub / "trainer_state.json").exists():
                step = int(m.group(1))
                if best is None or step > best[0]:
                    best = (step, sub)
    return best[1] if best else None


def find_resume_checkpoint(output_dir: Path, previous_outputs: list[Path], rel: str) -> Path | None:
    """Checkpoint in ``output_dir`` or, failing that, in an attached previous notebook output.

    A checkpoint found in a (read-only) previous output is copied into ``output_dir`` first so the
    Trainer can rotate checkpoints and find the best one at the same absolute path.
    """
    ckpt = latest_checkpoint(output_dir)
    if ckpt is not None:
        return ckpt
    for prev in previous_outputs:
        src_dir = Path(prev) / rel
        ckpt = latest_checkpoint(src_dir)
        if ckpt is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub in src_dir.iterdir():
                if _CKPT.match(sub.name):
                    shutil.copytree(sub, output_dir / sub.name, dirs_exist_ok=True)
            log.info("Copied checkpoints from previous output %s", src_dir)
            return latest_checkpoint(output_dir)
    return None
