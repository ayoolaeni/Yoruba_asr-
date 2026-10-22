"""Offline generation of the augmented training set (2 copies per CV train clip, FLAC + manifest)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..audio import SR, duration_reason, safe_load_trimmed, save_flac
from ..utils import get_logger, save_json
from .transforms import NoiseBank, RIRBank, apply_chain, choose_chain, median_f0, tempo

log = get_logger(__name__)


def clip_rng(seed: int, clip_index: int, copy_index: int) -> np.random.Generator:
    """Generator seeded by (seed, clip, copy): the augmented set is identical on every run."""
    return np.random.default_rng([int(seed), int(clip_index), int(copy_index)])


def build_augmented_set(train: pd.DataFrame, out_dir: str | Path, cfg_aug, seed: int, transforms: list[str],
                        min_s: float, max_s: float, top_db: float, musan_dir=None, rir_dir=None,
                        name: str = "main") -> tuple[pd.DataFrame, dict]:
    """Create ``copies_per_clip`` augmented copies of every row of ``train`` (needs ``path, text_tone``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_augmented.csv"
    noise_bank = NoiseBank.from_dir(musan_dir)
    rir_bank = RIRBank.build(int(cfg_aug["rir_bank_size"]), seed, tuple(cfg_aug["rt60"]), rir_dir)
    copies = int(cfg_aug["copies_per_clip"])
    rows, dropped = [], {"corrupt": 0, "too_short": 0, "too_long": 0}
    records = train.reset_index(drop=True).to_dict("records")
    for ci, r in enumerate(tqdm(records, desc=f"augment[{name}]", mininterval=10)):
        audio = None
        for k in range(copies):
            path = out_dir / f"aug_{ci:06d}_{k}.flac"
            rng = clip_rng(seed, ci, k)
            chain = choose_chain(rng, list(transforms), float(cfg_aug["transform_prob"]))
            if path.exists():
                # Deterministic: re-derive the description without recomputing audio.
                import soundfile as sf

                info = sf.info(str(path))
                rows.append({**_row(r, path, info.frames / info.samplerate), "chain": "+".join(chain)})
                continue
            if audio is None:
                audio = safe_load_trimmed(r["path"], SR, top_db)
                if audio is None:
                    dropped["corrupt"] += copies
                    break
            y, applied = apply_chain(audio, chain, rng, cfg_aug, noise_bank, rir_bank)
            dur = len(y) / SR
            reason = duration_reason(dur, min_s, max_s)
            if reason:
                dropped[reason] += 1
                continue
            save_flac(path, y)
            rows.append({**_row(r, path, dur), "chain": "+".join(chain), "chain_detail": ";".join(applied)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    stats = {"name": name, "source_clips": len(records), "copies_per_clip": copies, "written": len(manifest),
             "dropped": dropped, "transforms": list(transforms),
             "hours": round(float(manifest["duration"].sum()) / 3600, 3) if len(manifest) else 0.0,
             "noise_source": "musan" if noise_bank.files else "generated coloured noise",
             "rir_source": rir_bank.labels[0].split(":")[0] if rir_bank.labels else None}
    save_json(stats, out_dir / "augment_stats.json")
    log.info("augmented[%s]: %d clips written, dropped %s", name, len(manifest), dropped)
    return manifest, stats


def _row(r: dict, path: Path, dur: float) -> dict:
    return {"path": str(path), "sentence": r.get("sentence", r["text_tone"]), "text_tone": r["text_tone"],
            "text_notone": r["text_notone"], "speaker": r.get("speaker", "unknown"), "corpus": "cv_aug",
            "split": "train", "duration": dur, "source_path": r["path"], "synthetic": False}


def pitch_check(train: pd.DataFrame, cfg_aug, seed: int, top_db: float, n: int = 20) -> dict:
    """Tempo perturbation must keep pitch: median F0 change < 3% (median over clips)."""
    rng = np.random.default_rng([seed, 31337])
    idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
    results = []
    for i in idx:
        r = train.iloc[int(i)]
        x = safe_load_trimmed(r["path"], SR, top_db)
        if x is None:
            continue
        f = float(rng.choice(cfg_aug["tempo_factors"]))
        f0_orig, f0_aug = median_f0(x), median_f0(tempo(x, f))
        rel = abs(f0_aug - f0_orig) / f0_orig if np.isfinite(f0_orig) and f0_orig > 0 and np.isfinite(f0_aug) else None
        results.append({"path": str(r["path"]), "tempo_factor": f, "median_f0_original": f0_orig,
                        "median_f0_augmented": f0_aug, "relative_diff": rel})
    rels = [x["relative_diff"] for x in results if x["relative_diff"] is not None]
    limit = float(cfg_aug["pitch_check_max_rel_diff"])
    median_rel = float(np.median(rels)) if rels else float("nan")
    out = {"clips": results, "n_measured": len(rels), "median_relative_diff": median_rel,
           "share_within_limit": float(np.mean([x < limit for x in rels])) if rels else None,
           "limit": limit, "passed": bool(rels) and median_rel < limit}
    return out
