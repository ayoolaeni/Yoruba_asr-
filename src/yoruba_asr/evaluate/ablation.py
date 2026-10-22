"""Leave-one-out ablation of V2 (Section 15.5).

Each run differs from ``v2_ref`` in exactly one respect and uses the same (short) step budget.
Runs are resumable and can be selected one at a time (``--only no_tts``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ABLATION_COLUMNS = ["run", "description", "transforms", "use_tts", "mask_time_prob", "max_steps", "train_clips",
                    "dev_wer", "dev_ter", "test_wer", "test_ter", "test_cer", "test_wer_no_tone", "status"]

DESCRIPTIONS = {
    "v2_ref": "full V2 recipe (reference)",
    "no_tempo": "without tempo perturbation",
    "no_noise": "without additive noise",
    "no_reverb": "without reverberation",
    "no_tts": "without TTS synthetic speech",
    "no_tuned_mask": "mask_time_prob = 0.05 instead of the tuned value",
    "plus_speed": "+ pitch-changing speed perturbation",
}


def resolve_run(name: str, spec: dict, base_transforms: list[str], tuned_mask: float) -> dict:
    spec = dict(spec or {})
    transforms = list(base_transforms)
    if spec.get("remove_transform"):
        transforms = [t for t in transforms if t != spec["remove_transform"]]
    if spec.get("add_transform") and spec["add_transform"] not in transforms:
        transforms.append(spec["add_transform"])
    return {
        "run": name, "description": DESCRIPTIONS.get(name, name), "transforms": transforms,
        "use_tts": bool(spec.get("use_tts", True)),
        "mask_time_prob": float(spec.get("mask_time_prob", tuned_mask)),
        # Runs whose transform set equals the main set reuse the main augmented copies.
        "aug_set": "main" if sorted(transforms) == sorted(base_transforms) else "abl_" + name,
    }


def update_table(path: Path, row: dict) -> pd.DataFrame:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ABLATION_COLUMNS)
    df = df[df["run"] != row["run"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    order = {k: i for i, k in enumerate(DESCRIPTIONS)}
    df = df.sort_values("run", key=lambda s: s.map(lambda x: order.get(x, 99)))
    df.to_csv(path, index=False)
    return df
