"""Export result tables for the thesis (Markdown, CSV and LaTeX)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import load_json

VARIANT_NAMES = {"V1": "Source-only baseline", "V2": "Augmentation only", "V3": "Domain adaptation only",
                 "V4": "Hybrid (augmentation + adaptation)"}


def _fmt(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "")
    return out


def _md(df: pd.DataFrame) -> str:
    df = _fmt(df)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    df = _fmt(df)
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")  # noqa: E731
    lines = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline", " & ".join(esc(c) for c in cols) + " \\\\",
             "\\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(r[c]) for c in cols) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def export_tables(results_dir: str | Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    tables: list[tuple[str, str, pd.DataFrame]] = []
    summary = results_dir / "summary.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        s.insert(1, "description", s["variant"].map(VARIANT_NAMES))
        keep = [c for c in ("variant", "description", "wer", "cer", "wer_no_tone", "ter", "ter_matched_words",
                            "n_utts", "train_clips", "train_hours", "gpu_hours", "mask_time_prob") if c in s]
        tables.append(("Table 4.1: Test-set results of the four variants (%)", "tab:main_results", s[keep]))
    for fname, caption, label in [
        ("significance.csv", "Table 4.2: Speaker-block bootstrap significance (difference A - B, percentage points)",
         "tab:significance"),
        ("factorial_effects.csv", "Table 4.3: Factorial main effects and interaction (percentage points)",
         "tab:effects"),
        ("factorial_effects_ci.csv", "Table 4.3b: Factorial effects with speaker-bootstrap 95% CIs (percentage points)",
         "tab:effects_ci"),
        ("ablation.csv", "Table 4.4: Ablation study (V2, 1,500 steps each)", "tab:ablation"),
    ]:
        p = results_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:  # e.g. significance with only one trained variant
                continue
            if df.empty:
                continue
            if fname == "significance.csv":
                df = df[[c for c in ("comparison", "metric", "value_a", "value_b", "delta", "ci_low", "ci_high",
                                     "significant", "n_speakers") if c in df]]
            tables.append((caption, label, df))
    audit = load_json(results_dir / "corpus_audit.json")
    if audit:
        rows = []
        for corpus, splits in audit["corpora"].items():
            for split, st in splits.items():
                if not split.startswith("_"):
                    rows.append({"corpus": corpus, "split": split, "clips": st["clips_kept"], "hours": st["hours"],
                                 "speakers": st["speakers"], "tone_marked_%": 100 * (st["share_with_tone_mark"] or 0)})
        tables.insert(0, ("Table 4.0: Corpus audit", "tab:audit", pd.DataFrame(rows)))

    md = ["# Thesis tables", ""]
    tex = []
    csv_parts = []
    for caption, label, df in tables:
        md += [f"## {caption}", "", _md(df), ""]
        tex.append(_latex(df, caption, label))
        csv_parts.append(df.assign(table=label))
    out = {"md": results_dir / "thesis_tables.md", "csv": results_dir / "thesis_tables.csv",
           "tex": results_dir / "thesis_tables.tex"}
    out["md"].write_text("\n".join(md) if tables else "# Thesis tables\n\nNo results yet.\n", encoding="utf-8")
    out["tex"].write_text("\n\n".join(tex), encoding="utf-8")
    (pd.concat(csv_parts, ignore_index=True) if csv_parts else pd.DataFrame()).to_csv(out["csv"], index=False)
    return out
