"""Small helpers: seeding, logging, timing, JSON I/O, hashing and GPU information."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "yoruba_asr") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_info() -> dict:
    """Name/memory of the GPU, or a CPU description."""
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
            info["cuda"] = torch.version.cuda
        else:
            info["gpu_name"] = "cpu"
    except ImportError:
        info["gpu_name"] = "cpu"
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def disk_free_gb(path: str | Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(p).free / 1024**3, 2)


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class RunInfo:
    """Records wall-clock time, GPU name, peak GPU memory and steps/sec for one stage.

    Sessions accumulate in ``run_info.json`` so multi-session stages report the total.
    """

    def __init__(self, path: str | Path, stage: str):
        self.path = Path(path)
        self.stage = stage
        self.start = time.time()
        self.extra: dict[str, Any] = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def finish(self, steps: int | None = None, **extra: Any) -> dict:
        wall = time.time() - self.start
        info = gpu_info()
        peak = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024**3
        except ImportError:
            pass
        session = {
            "stage": self.stage,
            "finished_at": now_iso(),
            "wall_seconds": round(wall, 1),
            "gpu_name": info.get("gpu_name"),
            "peak_gpu_memory_gb": round(peak, 2),
            "gpu_hours": round(wall / 3600, 3) if info.get("cuda_available") else 0.0,
            "steps": steps,
            "steps_per_sec": round(steps / wall, 4) if steps and wall > 0 else None,
            **self.extra,
            **extra,
        }
        previous = load_json(self.path, default={}) or {}
        sessions = previous.get("sessions", []) + [session]
        out = {
            "stage": self.stage,
            "sessions": sessions,
            "total_wall_hours": round(sum(s["wall_seconds"] for s in sessions) / 3600, 3),
            "total_gpu_hours": round(sum(s.get("gpu_hours") or 0 for s in sessions), 3),
            "environment": info,
        }
        save_json(out, self.path)
        return out


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    t0 = time.time()
    yield
    msg = f"{label}: {time.time() - t0:.1f}s"
    (logger or get_logger()).info(msg)


def human_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h"
