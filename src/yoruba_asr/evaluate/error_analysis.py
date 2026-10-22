"""Error analysis: tonal error share, H/M/L confusion, breakdowns, qualitative examples and figures.

Figures are 300 dpi PNGs drawn in greyscale with hatching/markers so they stay readable in print.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..text import TONE_LABELS, has_subdot
from ..utils import get_logger, load_json, save_json
from .metrics import all_metrics, word_alignment

log = get_logger(__name__)
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "++"]
GREYS = ["0.15", "0.45", "0.7", "0.9", "0.3", "0.6"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def length_bucket(d: float) -> str:
    if not np.isfinite(d):
        return "unknown"
    if d < 3:
        return "<3 s"
    if d < 6:
        return "3-6 s"
    if d < 10:
        return "6-10 s"
    return ">10 s"


def breakdown(preds: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(by, dropna=False):
        if len(g) == 0:
            continue
        m = all_metrics(g["reference"].tolist(), g["hypothesis"].fillna("").tolist())
        rows.append({by: key if key != "" else "unknown", "n_utts": len(g), "wer": m["wer"], "ter": m["ter"],
                     "cer": m["cer"], "wer_no_tone": m["wer_no_tone"]})
    return pd.DataFrame(rows)


def subdot_word_error(preds: pd.DataFrame) -> pd.DataFrame:
    """Word error rate (sub+del over reference words) for words with vs without sub-dotted letters."""
    counts = {True: [0, 0], False: [0, 0]}
    for r, h in zip(preds["reference"], preds["hypothesis"].fillna("")):
        for op, rw, _hw in word_alignment(str(r), str(h)):
            if rw is None:
                continue
            k = has_subdot(rw)
            counts[k][1] += 1
            counts[k][0] += op != "equal"
    return pd.DataFrame([{"word_type": "with sub-dot" if k else "without sub-dot", "ref_words": v[1],
                          "word_error_rate": 100 * v[0] / v[1] if v[1] else np.nan} for k, v in counts.items()])


def highlight(ref: str, hyp: str) -> tuple[str, str]:
    """Markdown: substitutions **bold**, deletions ~~struck~~ (in REF), insertions _italic_ (in HYP)."""
    r_out, h_out = [], []
    for op, rw, hw in word_alignment(ref, hyp):
        if op == "equal":
            r_out.append(rw); h_out.append(hw)
        elif op == "substitute":
            r_out.append(f"**{rw}**"); h_out.append(f"**{hw}**")
        elif op == "delete":
            r_out.append(f"~~{rw}~~")
        else:
            h_out.append(f"_{hw}_")
    return " ".join(r_out), " ".join(h_out)


def examples_markdown(preds: pd.DataFrame, variant: str, n: int = 20, seed: int = 42) -> str:
    errs = preds[preds["reference"] != preds["hypothesis"].fillna("")]
    pick = errs.sample(n=min(n, len(errs)), random_state=seed) if len(errs) else errs
    lines = [f"# {variant}: qualitative examples", "",
             "Substitutions in **bold**, deletions ~~struck~~, insertions in _italics_.", ""]
    for i, (_, row) in enumerate(pick.iterrows(), 1):
        r, h = highlight(str(row["reference"]), str(row["hypothesis"]) if isinstance(row["hypothesis"], str) else "")
        lines += [f"{i}. REF: {r}", f"   HYP: {h}", ""]
    return "\n".join(lines)


def plot_confusion(conf, title: str, path: Path) -> None:
    plt = _plt()
    conf = np.asarray(conf, dtype=float)
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(conf[i, j])}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), TONE_LABELS); ax.set_yticks(range(3), TONE_LABELS)
    ax.set_xlabel("hypothesis tone"); ax.set_ylabel("reference tone"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_variant_bars(summary: pd.DataFrame, cis: dict, path: Path) -> None:
    """WER/CER/TER for V1-V4 with bootstrap 95% CIs."""
    plt = _plt()
    metrics = [m for m in ("wer", "cer", "ter") if m in summary.columns]
    variants = summary["variant"].tolist()
    x = np.arange(len(metrics))
    w = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, v in enumerate(variants):
        vals = [float(summary.loc[summary.variant == v, m].iloc[0]) for m in metrics]
        err = None
        if v in cis:
            lo = [vals[i] - cis[v].get(m, {}).get("ci_low", vals[i]) for i, m in enumerate(metrics)]
            hi = [cis[v].get(m, {}).get("ci_high", vals[i]) - vals[i] for i, m in enumerate(metrics)]
            err = [lo, hi]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=v, color=GREYS[k % 6], edgecolor="black",
               hatch=HATCHES[k % 6], yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x, [m.upper() for m in metrics]); ax.set_ylabel("error rate (%)")
    ax.legend(fontsize=8, ncol=4); ax.set_title("Test-set error rates (95% speaker-bootstrap CI)")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_interaction(summary: pd.DataFrame, path: Path, metric: str = "wer") -> None:
    plt = _plt()
    s = summary.set_index("variant")
    if not all(v in s.index for v in ("V1", "V2", "V3", "V4")):
        return
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot([0, 1], [s.loc["V1", metric], s.loc["V2", metric]], "k-o", label="no domain adaptation")
    ax.plot([0, 1], [s.loc["V3", metric], s.loc["V4", metric]], "k--s", label="with domain adaptation")
    ax.set_xticks([0, 1], ["no augmentation", "augmentation"]); ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title("Factorial interaction"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_ablation(abl: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    if abl.empty or "test_wer" not in abl:
        return
    abl = abl.sort_values("test_wer")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = ["0.2" if n == "v2_ref" else "0.7" for n in abl["run"]]
    ax.barh(abl["run"], abl["test_wer"], color=colors, edgecolor="black")
    ax.set_xlabel("test WER (%)"); ax.set_title("Ablation (1,500 steps each; dark = full V2)")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_dev_curves(results_dir: Path, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    styles = ["k-o", "k--s", "k-.^", "k:d", "k-x"]
    found = False
    for k, tag in enumerate(["V1", "V2", "V3", "V4", "B1"]):
        p = results_dir / tag / "train_log.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "eval_wer" not in df:
            continue
        df = df.dropna(subset=["eval_wer"])
        ax.plot(df["step"], 100 * df["eval_wer"], styles[k], markersize=3, label=tag)
        found = True
    if found:
        ax.set_xlabel("update step"); ax.set_ylabel("dev WER (%)"); ax.legend(fontsize=8)
        ax.set_title("Fine-tuning dev-WER curves")
        fig.tight_layout(); fig.savefig(path, dpi=300)
    plt.close(fig)


def analyse_variant(preds: pd.DataFrame, variant: str, out_dir: Path, n_examples: int = 20, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = all_metrics(preds["reference"].tolist(), preds["hypothesis"].fillna("").tolist())
    preds = preds.copy()
    preds["length_bucket"] = preds.get("duration", pd.Series(np.nan, index=preds.index)).astype(float).map(length_bucket)
    for col in ("gender", "age"):
        preds[col] = preds.get(col, "").fillna("").astype(str).replace("", "unknown")
    tables = {by: breakdown(preds, by) for by in ("gender", "age", "length_bucket")}
    tables["subdot_words"] = subdot_word_error(preds)
    for name, t in tables.items():
        t.to_csv(out_dir / f"breakdown_{name}.csv", index=False)
    plot_confusion(m["tone_confusion"], f"{variant}: tone confusion", out_dir / "tone_confusion.png")
    (out_dir / "examples.md").write_text(examples_markdown(preds, variant, n_examples, seed), encoding="utf-8")
    res = {"variant": variant, "tonal_word_error_share": m["wer"] - m["wer_no_tone"], **m,
           "breakdowns": {k: v.to_dict("records") for k, v in tables.items()}}
    save_json(res, out_dir / "error_analysis.json")
    return res


def write_figures(results_dir: Path, summary: pd.DataFrame | None, cis: dict | None) -> list[str]:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if summary is not None and len(summary):
        plot_variant_bars(summary, cis or {}, fig_dir / "variants_wer_cer_ter.png"); made.append("variants_wer_cer_ter.png")
        plot_interaction(summary, fig_dir / "factorial_interaction.png")
        if (fig_dir / "factorial_interaction.png").exists():
            made.append("factorial_interaction.png")
    abl = results_dir / "ablation.csv"
    if abl.exists():
        plot_ablation(pd.read_csv(abl), fig_dir / "ablation.png"); made.append("ablation.png")
    plot_dev_curves(results_dir, fig_dir / "dev_wer_curves.png")
    if (fig_dir / "dev_wer_curves.png").exists():
        made.append("dev_wer_curves.png")
    cpt_png = results_dir / "V3" / "cpt_loss.png"
    if cpt_png.exists():
        import shutil

        shutil.copy2(cpt_png, fig_dir / "cpt_loss.png"); made.append("cpt_loss.png")
    return made


def load_cis(results_dir: Path) -> dict:
    return load_json(results_dir / "system_cis.json", default={}) or {}
