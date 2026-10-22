"""Yoruba ASR web demo (Gradio).

Tabs: Transcribe | Compare models | Results | About.
Run locally:  python app/app.py        (then open http://127.0.0.1:7870)
Environment variables (optional):
  YORUBA_ASR_MODELS_YAML   path to a models.yaml (default: app/models.yaml)
  YORUBA_ASR_RESULTS_DIR   results folder (default: from models.yaml)
"""
from __future__ import annotations

import html
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr  # noqa: E402

from yoruba_asr.evaluate.metrics import all_metrics, word_alignment  # noqa: E402
from yoruba_asr.text import normalize_text, strip_tones  # noqa: E402

VARIANT_INFO = {
    "V1": "Source-only baseline (no augmentation, no adaptation)",
    "V2": "Augmentation only (tempo, noise, reverb, gain + MMS-TTS)",
    "V3": "Domain adaptation only (continued pretraining + staged fine-tuning)",
    "V4": "Hybrid (augmentation + domain adaptation)",
}
NO_MODELS_MSG = ("No trained models were found. Copy the Kaggle output folders `models/V1` ... `models/V4` into "
                 "`yoruba-asr/models/` (or edit `app/models.yaml` to point to them / to Hugging Face repos).")


# --------------------------------------------------------------------------------------
# configuration and model registry
# --------------------------------------------------------------------------------------
def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("YORUBA_ASR_MODELS_YAML", APP_DIR / "models.yaml"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    data.setdefault("models", {})
    data.setdefault("results_dir", "results")
    data.setdefault("default_model", "V4")
    data.setdefault("max_cached_models", 2)
    data.setdefault("max_audio_seconds", 30)
    if os.environ.get("YORUBA_ASR_RESULTS_DIR"):
        data["results_dir"] = os.environ["YORUBA_ASR_RESULTS_DIR"]
    data["_base"] = str(path.parent.parent if path.parent == APP_DIR else path.parent)
    return data


def _resolve(source: str, base: str) -> tuple[str, str] | None:
    """('dir', path) for an existing local model folder, ('hub', id) for a repo id, None if missing."""
    p = Path(source)
    if not p.is_absolute():
        p = Path(base) / p
    if (p / "config.json").exists():
        return "dir", str(p)
    if "/" in source and not Path(source).exists() and not source.startswith((".", "/", "models")) \
            and ":" not in source:
        return "hub", source
    return None


def available_models(settings: dict) -> "OrderedDict[str, tuple[str, str]]":
    out: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for variant, source in (settings.get("models") or {}).items():
        r = _resolve(str(source), settings["_base"])
        if r:
            out[str(variant)] = r
    return out


class ModelCache:
    """Loads models lazily and keeps at most ``max_items`` in memory (least recently used evicted)."""

    def __init__(self, registry: dict, max_items: int = 2):
        self.registry, self.max_items = registry, max_items
        self._cache: OrderedDict[str, object] = OrderedDict()

    def get(self, variant: str):
        from yoruba_asr.inference import Transcriber

        if variant in self._cache:
            self._cache.move_to_end(variant)
            return self._cache[variant]
        kind, src = self.registry[variant]
        t = Transcriber.from_dir(src) if kind == "dir" else Transcriber.from_hub(src)
        t.name = variant
        self._cache[variant] = t
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return t


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _audio_to_array(audio, max_seconds: float) -> tuple[np.ndarray, int, str]:
    if audio is None:
        raise gr.Error("Please record or upload some audio first.")
    if isinstance(audio, (str, Path)):
        from yoruba_asr.audio import load_audio

        x, sr = load_audio(audio, 16000), 16000
    else:
        sr, x = audio
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
    note = ""
    if len(x) > max_seconds * sr:
        x = x[: int(max_seconds * sr)]
        note = f"Audio longer than {max_seconds:g} s was truncated to {max_seconds:g} s. "
    return x, int(sr), note


def diff_html(reference: str, hypothesis: str) -> str:
    """Word-level differences: substitutions, deletions (missing words) and insertions."""
    styles = {"substitute": "background:rgba(230,159,0,.35)", "delete": "background:rgba(213,94,0,.35);"
              "text-decoration:line-through", "insert": "background:rgba(0,114,178,.30)"}
    ref_out, hyp_out = [], []
    for op, rw, hw in word_alignment(reference, hypothesis):
        if op == "equal":
            ref_out.append(html.escape(rw)); hyp_out.append(html.escape(hw))
        elif op == "substitute":
            ref_out.append(f'<span style="{styles[op]}">{html.escape(rw)}</span>')
            hyp_out.append(f'<span style="{styles[op]}">{html.escape(hw)}</span>')
        elif op == "delete":
            ref_out.append(f'<span style="{styles[op]}">{html.escape(rw)}</span>')
        else:
            hyp_out.append(f'<span style="{styles[op]}">{html.escape(hw)}</span>')
    legend = ('<small><span style="%s">substituted</span> &nbsp; <span style="%s">missing</span> &nbsp; '
              '<span style="%s">inserted</span></small>') % (styles["substitute"], styles["delete"], styles["insert"])
    return (f"<div style='line-height:1.9'><b>Reference:</b> {' '.join(ref_out)}<br>"
            f"<b>Model:</b> {' '.join(hyp_out)}<br>{legend}</div>")


def clip_metrics(reference: str, hypothesis: str) -> dict:
    ref = normalize_text(reference)
    m = all_metrics([ref], [hypothesis])
    return {"WER %": round(m["wer"], 1), "CER %": round(m["cer"], 1), "WER no tone %": round(m["wer_no_tone"], 1),
            "TER %": round(m["ter"], 1) if np.isfinite(m["ter"]) else None}


# --------------------------------------------------------------------------------------
# results tab
# --------------------------------------------------------------------------------------
RESULT_TABLES = [("summary.csv", "Test-set results (%)"), ("significance.csv", "Speaker-block bootstrap significance"),
                 ("factorial_effects.csv", "Factorial effects (percentage points)"), ("ablation.csv", "Ablation study")]
FIGURES = ["variants_wer_cer_ter.png", "factorial_interaction.png", "ablation.png", "cpt_loss.png",
           "dev_wer_curves.png"]


def load_results(results_dir: str | Path) -> dict:
    """Tables and figure paths that exist; ``message`` explains when nothing is available yet."""
    results_dir = Path(results_dir)
    tables, figures = {}, []
    for fname, title in RESULT_TABLES:
        p = results_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                tables[title] = df.round(3) if len(df) else df
            except Exception:
                continue
    for f in FIGURES:
        p = results_dir / "figures" / f
        if p.exists():
            figures.append((str(p), f.replace("_", " ").replace(".png", "")))
    msg = "" if (tables or figures) else (f"No results yet. Run the evaluation stages on Kaggle and copy the "
                                          f"`results/` folder to `{results_dir}`.")
    return {"tables": tables, "figures": figures, "message": msg}


ABOUT = """
## Enhancing ASR Accuracy for Low-Resource Nigerian Tonal Languages Through Hybrid Data Augmentation and Domain Adaptation: A Case Study of Yoruba

All models fine-tune **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`, wav2vec 2.0) with a CTC head and greedy decoding (no language model). They share one character vocabulary, which keeps Yoruba **tone marks** (à á ā) and **sub-dots** (ẹ ọ ṣ).

| Variant | Data augmentation | Domain adaptation |
|---|---|---|
| **V1** source-only baseline | No | No |
| **V2** augmentation only | Yes: tempo, noise, reverberation, gain (all pitch-preserving) + MMS-TTS speech | No |
| **V3** domain adaptation only | No | Continued self-supervised pretraining, then staged fine-tuning (out-of-domain → Common Voice) |
| **V4** hybrid | Yes | Yes |

**Metrics:** WER, CER, tone-insensitive WER (tone marks removed) and Tone Error Rate (TER, errors in the H/M/L tone sequence).

**Data and licences:** Mozilla Common Voice 27.0 Yoruba (CC0; the data may not be re-hosted), Google Yoruba corpus OpenSLR 86 (CC BY-SA 4.0), ÌròyìnSpeech (text CC BY 4.0). The synthetic speech, including this app's demo clips, comes from `facebook/mms-tts-yor` (CC BY-NC 4.0), which allows **research use only**.

**Limitations:** the models are trained on read speech in Standard Yoruba from a small number of speakers. Expect higher error rates on spontaneous, dialectal, code-switched or noisy speech. Numbers written as digits were excluded from training.

**How to cite:** Bolaji Oluseye (2026). *Enhancing ASR Accuracy for Low-Resource Nigerian Tonal Languages Through Hybrid Data Augmentation and Domain Adaptation: A Case Study of Yoruba.* Final-year project.
"""


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
MIC_HELP = """
**The microphone is detected by your browser, not by this app.** If you see *"No microphone found"*:
1. Plug in or switch on a microphone or headset. Many desktop PCs have none built in.
2. **Windows:** *Settings → Privacy & security → Microphone*: turn on *Microphone access* and *Let desktop apps / your browser access your microphone*. Check *Settings → System → Sound → Input* to confirm a device is listed.
3. **Browser:** click the icon to the left of the address bar → *Site settings → Microphone → Allow*, then reload the page (F5).
4. Open the app at `http://127.0.0.1:7870` or `http://localhost:7870`. Browsers only allow the microphone on localhost or https pages.

You can always **upload** an audio file instead (wav, mp3, flac, m4a, ogg) or click one of the demo clips.
"""
METRIC_HEADERS = ["WER %", "CER %", "WER no tone %", "TER %"]
COMPARE_HEADERS = ["model", "transcription", "without tones", "seconds"]


def build_demo(settings_path: str | Path | None = None) -> gr.Blocks:
    settings = load_settings(settings_path)
    configured = [str(v) for v in (settings.get("models") or {})] or list(VARIANT_INFO)
    cache = ModelCache({}, int(settings["max_cached_models"]))
    def model_choices(installed) -> list[tuple[str, str]]:
        return [(f"{v} – {VARIANT_INFO.get(v, v)}" + ("" if v in installed else " (not installed yet)"), v)
                for v in configured]

    def best_installed(installed) -> str:
        """The configured default if installed, otherwise the most advanced installed variant."""
        if settings["default_model"] in installed:
            return settings["default_model"]
        avail = [v for v in configured if v in installed]
        return avail[-1] if avail else (settings["default_model"] if settings["default_model"] in configured
                                        else configured[-1])

    _installed0 = available_models(settings)
    choices = model_choices(_installed0)
    default = best_installed(_installed0)
    results_dir = Path(settings["results_dir"])
    if not results_dir.is_absolute():
        results_dir = Path(settings["_base"]) / results_dir
    max_s = float(settings["max_audio_seconds"])
    examples = sorted(str(p) for p in (APP_DIR / "assets").glob("demo_*.wav"))

    def registry():
        """Re-scan the model folders on every request, so new models appear without restarting the app."""
        reg = available_models(settings)
        cache.registry = reg
        for v in [v for v in list(cache._cache) if v not in reg]:
            cache._cache.pop(v)
        return reg

    def model_status() -> str:
        reg = registry()
        lines = ["| Model | Description | Status |", "|---|---|---|"]
        for v in configured:
            status = "✅ installed" if v in reg else "⏳ not installed yet"
            lines.append(f"| **{v}** | {VARIANT_INFO.get(v, '')} | {status} |")
        if not reg:
            lines += ["", f"**No trained models yet.** {NO_MODELS_MSG} Then click *Refresh model status*."]
        return "\n".join(lines)

    def require(variant: str):
        reg = registry()
        if not reg:
            raise gr.Error(NO_MODELS_MSG)
        if variant not in reg:
            raise gr.Error(f"{variant} is not installed yet. Choose an installed model in the 'Model' menu: "
                           f"{', '.join(reg)}.")
        return reg

    def do_transcribe(audio, variant, reference):
        require(variant)
        x, sr, note = _audio_to_array(audio, max_s)
        t_load = time.time()
        transcriber = cache.get(variant)
        load_s = time.time() - t_load
        t0 = time.time()
        r = transcriber.transcribe_array(x, sr)
        loaded = f" (model loaded in {load_s:.1f} s, first use only)" if load_s > 0.5 else ""
        info = (f"{note}**Model:** {variant} – {VARIANT_INFO.get(variant, '')}  \n"
                f"**Processing time:** {time.time() - t0:.2f} s for {len(x) / sr:.1f} s of audio{loaded}")
        if reference and reference.strip():
            df = pd.DataFrame([clip_metrics(reference, r["text"])], columns=METRIC_HEADERS)
            return (r["text"], r["text_notone"], info, diff_html(normalize_text(reference), r["text"]),
                    gr.update(value=df, visible=True))
        return r["text"], r["text_notone"], info, "", gr.update(visible=False)

    def do_compare(audio, reference):
        reg = registry()
        if not reg:
            raise gr.Error(NO_MODELS_MSG)
        x, sr, note = _audio_to_array(audio, max_s)
        rows = []
        for v in reg:
            transcriber = cache.get(v)
            t0 = time.time()
            r = transcriber.transcribe_array(x, sr)
            row = {"model": v, "transcription": r["text"], "without tones": strip_tones(r["text"]),
                   "seconds": round(time.time() - t0, 2)}
            if reference and reference.strip():
                row.update(clip_metrics(reference, r["text"]))
            rows.append(row)
        return gr.update(value=pd.DataFrame(rows), visible=True), note

    def refresh_results():
        res = load_results(results_dir)
        md = res["message"] or f"Results loaded from `{results_dir}`."
        tables = [res["tables"].get(title) for _, title in RESULT_TABLES]
        out = [md]
        out += [gr.update(value=t, visible=True) if t is not None else gr.update(visible=False) for t in tables]
        out.append(gr.update(value=res["figures"], visible=bool(res["figures"])))
        return out

    with gr.Blocks(title="Yoruba ASR (XLS-R)") as demo:
        gr.Markdown("# Yoruba speech recognition\nRecord or upload Yoruba speech and get a transcription "
                    "**with tone marks**.")
        with gr.Accordion("Model status", open=not bool(available_models(settings))):
            status_md = gr.Markdown(model_status())
            status_btn = gr.Button("Refresh model status", size="sm")
        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                     label="Yoruba audio: upload a file or record (max 30 s)")
                    with gr.Accordion("Microphone not working?", open=False):
                        gr.Markdown(MIC_HELP)
                    model = gr.Dropdown(choices=choices, value=default, label="Model")
                    reference = gr.Textbox(label="Reference text (optional): shows WER/CER/TER for this clip",
                                           lines=2, placeholder="Type what was actually said, with tone marks")
                    run = gr.Button("Transcribe", variant="primary")
                    if examples:
                        gr.Examples(examples=[[e] for e in examples], inputs=[audio],
                                    label="Demo clips (synthesised with MMS-TTS): click one to load it")
                with gr.Column():
                    out_text = gr.Textbox(label="Transcription (with tone marks)", lines=3)
                    out_notone = gr.Textbox(label="Transcription (without tone marks)", lines=3)
                    out_info = gr.Markdown()
                    out_diff = gr.HTML()
                    out_metrics = gr.Dataframe(headers=METRIC_HEADERS, label="Per-clip metrics", interactive=False,
                                               visible=False)
            run.click(do_transcribe, [audio, model, reference], [out_text, out_notone, out_info, out_diff, out_metrics])
        with gr.Tab("Compare models"):
            gr.Markdown("Runs the same audio through every installed model (V1–V4).")
            c_audio = gr.Audio(sources=["upload", "microphone"], type="numpy",
                               label="Yoruba audio: upload a file or record (max 30 s)")
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=[c_audio], label="Demo clips")
            c_ref = gr.Textbox(label="Reference text (optional)", lines=2)
            c_run = gr.Button("Run every installed model", variant="primary")
            c_note = gr.Markdown()
            c_table = gr.Dataframe(headers=COMPARE_HEADERS, label="One row per model", interactive=False,
                                   wrap=True, visible=False)
            c_run.click(do_compare, [c_audio, c_ref], [c_table, c_note])
        with gr.Tab("Results"):
            r_btn = gr.Button("Reload results", size="sm")
            r_msg = gr.Markdown()
            r_tables = [gr.Dataframe(label=title, interactive=False, wrap=True, visible=False)
                        for _, title in RESULT_TABLES]
            r_gallery = gr.Gallery(label="Figures", columns=2, height="auto", visible=False)
            r_btn.click(refresh_results, None, [r_msg] + r_tables + [r_gallery])
            demo.load(refresh_results, None, [r_msg] + r_tables + [r_gallery])
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
        def refresh_models(current):
            """Status table + dropdown: keep the user's choice if installed, else pick the best installed."""
            installed = registry()
            value = current if current in installed else best_installed(installed)
            return model_status(), gr.update(choices=model_choices(installed), value=value)

        def load_models():
            """On page (re)load always start on the best installed model (e.g. a newly added V3)."""
            installed = registry()
            return model_status(), gr.update(choices=model_choices(installed), value=best_installed(installed))

        status_btn.click(refresh_models, model, [status_md, model])
        demo.load(load_models, None, [status_md, model])
    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7870))))


if __name__ == "__main__":
    main()
