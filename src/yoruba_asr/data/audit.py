"""Corpus audit (Research Question 1): clip counts, hours, speakers, durations, tone coverage."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..text import char_inventory, describe_char, has_subdot, has_tone_mark
from ..utils import save_json
from .common_voice import REFERENCE_COUNTS


def _stats(group: pd.DataFrame) -> dict:
    kept = group[group["drop_reason"].isna()] if "drop_reason" in group else group
    dur = kept["duration"].dropna().astype(float) if "duration" in kept else pd.Series(dtype=float)
    texts = kept["sentence"].astype(str).tolist() if "sentence" in kept else []
    reasons = Counter(group["drop_reason"].dropna()) if "drop_reason" in group else Counter()
    sr = Counter(int(x) for x in kept["sample_rate"].dropna()) if "sample_rate" in kept else Counter()
    return {
        "clips_found": int(len(group)),
        "clips_kept": int(len(kept)),
        "hours": round(float(dur.sum()) / 3600, 3) if len(dur) else 0.0,
        "speakers": int(kept["speaker"].nunique()) if "speaker" in kept else 0,
        "sample_rates": {str(k): v for k, v in sorted(sr.items())},
        "duration_mean_s": round(float(dur.mean()), 2) if len(dur) else None,
        "duration_min_s": round(float(dur.min()), 2) if len(dur) else None,
        "duration_max_s": round(float(dur.max()), 2) if len(dur) else None,
        "filtered": dict(reasons),
        "share_with_tone_mark": round(float(np.mean([has_tone_mark(t) for t in texts])), 4) if texts else None,
        "share_with_subdot": round(float(np.mean([has_subdot(t) for t in texts])), 4) if texts else None,
        "share_toneless": round(float(np.mean([not has_tone_mark(t) for t in texts])), 4) if texts else None,
    }


def build_audit(rows: pd.DataFrame, extra: dict | None = None) -> dict:
    """``rows`` holds every loaded item with ``corpus, split, sentence, speaker, duration,
    sample_rate, drop_reason`` (``drop_reason`` is NaN when the item is kept)."""
    audit: dict = {"corpora": {}, "extra": extra or {}}
    for (corpus, split), g in rows.groupby(["corpus", "split"], sort=True):
        audit["corpora"].setdefault(corpus, {})[split] = _stats(g)
    for corpus, g in rows.groupby("corpus"):
        kept = g[g["drop_reason"].isna()] if "drop_reason" in g else g
        inv = char_inventory(kept["sentence"].astype(str)) if "sentence" in kept else {}
        audit["corpora"][corpus]["_character_inventory"] = {
            ch: {"count": n, "name": describe_char(ch)} for ch, n in inv.items()}
    cv = audit["corpora"].get("common_voice", {})
    if not cv:
        return audit
    audit["common_voice_reference_check"] = {
        split: {"expected": exp, "found": cv.get(split, {}).get("clips_found"),
                "matches": cv.get(split, {}).get("clips_found") == exp}
        for split, exp in REFERENCE_COUNTS.items() if split in cv or split in ("train", "dev", "test")
    }
    return audit


def audit_markdown(audit: dict) -> str:
    lines = ["# Corpus audit", "", "| corpus | split | found | kept | hours | speakers | mean dur (s) | "
             "min | max | tone-marked | sub-dotted | filtered |", "|" + "---|" * 12]
    for corpus, splits in audit["corpora"].items():
        for split, s in splits.items():
            if split.startswith("_"):
                continue
            filt = ", ".join(f"{k}: {v}" for k, v in s["filtered"].items()) or "-"
            fmt = lambda x: "-" if x is None else (f"{x:.1%}" if isinstance(x, float) and x <= 1 else str(x))  # noqa: E731
            lines.append(f"| {corpus} | {split} | {s['clips_found']} | {s['clips_kept']} | {s['hours']:.2f} | "
                         f"{s['speakers']} | {s['duration_mean_s']} | {s['duration_min_s']} | {s['duration_max_s']} | "
                         f"{fmt(s['share_with_tone_mark'])} | {fmt(s['share_with_subdot'])} | {filt} |")
    lines += ["", "## Common Voice 27.0 reference counts", "", "| split | expected | found | match |",
              "|---|---|---|---|"]
    for split, r in audit.get("common_voice_reference_check", {}).items():
        lines.append(f"| {split} | {r['expected']} | {r['found']} | {'yes' if r['matches'] else '**NO**'} |")
    extra = audit.get("extra", {})
    if extra:
        lines += ["", "## Notes", ""]
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
    for corpus, splits in audit["corpora"].items():
        inv = splits.get("_character_inventory", {})
        if inv:
            lines += ["", f"## Character inventory - {corpus}", "",
                      " ".join(f"`{c}`({v['count']})" for c, v in inv.items() if c != " ")]
    return "\n".join(lines) + "\n"


def write_audit(audit: dict, results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    save_json(audit, results_dir / "corpus_audit.json")
    (results_dir / "corpus_audit.md").write_text(audit_markdown(audit), encoding="utf-8")
