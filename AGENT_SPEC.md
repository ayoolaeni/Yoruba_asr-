# Yoruba ASR — Build Specification for the AI Developer Agent

> **Project:** Enhancing ASR Accuracy for Low-Resource Nigerian Tonal Languages Through Hybrid Data Augmentation and Domain Adaptation: A Case Study of Yoruba
>
> **Who this file is for:** an AI coding agent that will build the complete software for this final-year project, test it, and then hand over **one Kaggle notebook** that trains and evaluates the models on Kaggle's free GPU.
>
> **Read this whole file before writing any code.** Sections marked **MUST** are requirements. Sections marked **SHOULD** are strong recommendations. Anything not covered here is your judgement call, but record the decision in `docs/DECISIONS.md`.

---

## Table of Contents

1. [What you are building](#1-what-you-are-building)
2. [Ground rules for the agent](#2-ground-rules-for-the-agent)
3. [Research design you must preserve](#3-research-design-you-must-preserve)
4. [Tech stack](#4-tech-stack)
5. [Repository layout](#5-repository-layout)
6. [Configuration system](#6-configuration-system)
7. [Data sources and loaders](#7-data-sources-and-loaders)
8. [Text normalisation and tone handling](#8-text-normalisation-and-tone-handling)
9. [Audio preprocessing](#9-audio-preprocessing)
10. [Vocabulary and processor](#10-vocabulary-and-processor)
11. [Increment 1 — Baseline fine-tuning (V1)](#11-increment-1--baseline-fine-tuning-v1)
12. [Increment 2 — Hybrid augmentation module (V2)](#12-increment-2--hybrid-augmentation-module-v2)
13. [Increment 3 — Domain adaptation module (V3)](#13-increment-3--domain-adaptation-module-v3)
14. [Increment 4 — Hybrid system (V4)](#14-increment-4--hybrid-system-v4)
15. [Evaluation, significance testing, ablation, error analysis](#15-evaluation-significance-testing-ablation-error-analysis)
16. [The application (CLI + web demo)](#16-the-application-cli--web-demo)
17. [Testing requirements](#17-testing-requirements)
18. [The final Kaggle notebook (main hand-over)](#18-the-final-kaggle-notebook-main-hand-over)
19. [Kaggle platform constraints](#19-kaggle-platform-constraints)
20. [Milestones and definition of done](#20-milestones-and-definition-of-done)
21. [Final deliverables checklist](#21-final-deliverables-checklist)
22. [Reference code from the original Colab baseline](#22-reference-code-from-the-original-colab-baseline)

---

## 1. What you are building

A reproducible Yoruba speech-recognition system built on **XLS-R 300M (wav2vec 2.0) with a CTC head**. It must train and compare **four model variants** in a 2 × 2 factorial experiment:

| Variant | Data augmentation | Domain adaptation | Built in |
|---|---|---|---|
| **V1** Source-only baseline | No | No | Increment 1 |
| **V2** Augmentation-only | Yes | No | Increment 2 |
| **V3** Domain-adaptation-only | No | Yes | Increment 3 |
| **V4** Hybrid | Yes | Yes | Increment 4 |

You will deliver three things:

1. **A Python package** (`yoruba_asr`) containing the data pipeline, augmentation, domain adaptation, training, evaluation and inference code, plus a command-line interface (CLI).
2. **A web demo application** (Gradio). Users record or upload Yoruba audio, pick a model (V1–V4), and see the transcription. They can also compare models side by side and view the evaluation results.
3. **One Kaggle notebook** (`kaggle/yoruba_asr_kaggle.ipynb`). It runs the whole training and evaluation workflow on Kaggle, one stage per run. Deliver it **only after** the package and the app are complete and tested (see [Section 18](#18-the-final-kaggle-notebook-main-hand-over)).

The project is developed with the **Incremental Model**. Build and test the increments in order: data pipeline + V1 → augmentation + V2 → domain adaptation + V3 → integration + V4 → evaluation → app → Kaggle notebook.

---

## 2. Ground rules for the agent

**MUST:**

- **Never train or tune on the Common Voice test split.** Never put test sentences, test speakers or test audio into training, augmentation, TTS synthesis, vocabulary decisions or continued pretraining. The dev split is only for checkpoint selection and for tuning `mask_time_prob`.
- **Keep all four variants comparable.** They share the same backbone, vocabulary, data splits, optimiser, learning rate, batch size, number of update steps and seed. Only the augmentation and adaptation treatments may differ.
- **Do not invent dataset identifiers, URLs, licences or file formats.** If something is unknown (for example the exact ÌròyìnSpeech download layout), make the loader accept a **local folder + manifest CSV**. Document what the user must provide, and flag it in `docs/OPEN_QUESTIONS.md`.
- **Do not re-host Common Voice data** anywhere public (GitHub, public Kaggle datasets, Hugging Face). Its terms forbid it. The notebook reads it from a **private** Kaggle dataset the user uploads.
- **Use Unicode NFC everywhere** text is read, compared or written.
- **Make every random process seeded** (`seed = 42` by default) and every run resumable from checkpoints.
- **Fit Kaggle limits.** Every training stage must fit in one 12-hour session or resume cleanly across sessions, and must fit in 16 GB of GPU memory.
- **Use one GPU only**, even when Kaggle offers "GPU T4 × 2". Set `CUDA_VISIBLE_DEVICES=0` before importing torch.
- **Write tests** (see [Section 17](#17-testing-requirements)) and keep them passing on CPU with tiny synthetic data.

**SHOULD:**

- Prefer clear, well-commented, typed Python (3.10+) over clever code. The user is a final-year student and will explain this code in Chapter Four of the thesis.
- Log every important number (clip counts, hours, vocabulary size, filtered items, GPU time) to `results/` as JSON. The thesis needs these figures.
- Stop and write to `docs/OPEN_QUESTIONS.md`, rather than guessing, when a decision would change the research design.

---

## 3. Research design you must preserve

| Item | Value |
|---|---|
| Backbone | `facebook/wav2vec2-xls-r-300m` |
| Head / loss | Linear CTC head; CTC loss, `ctc_loss_reduction="mean"`, `ctc_zero_infinity=True` |
| Decoding (main comparison) | Greedy CTC decoding, **no** language model |
| Optimiser | AdamW, β₁ = 0.9, β₂ = 0.999, ε = 1e-8 |
| Learning rate | 3e-4, linear decay, 150 warm-up steps |
| Batch | `per_device_train_batch_size=8`, `gradient_accumulation_steps=4` (effective 32), eval batch 8 |
| Training budget | **`max_steps=3000` for every variant's final Common Voice stage** (fixed steps, not epochs) |
| Eval / save | Every 200 steps; keep best checkpoint by **dev WER**; `save_total_limit=2` |
| Precision / memory | FP16 mixed precision; gradient checkpointing; feature encoder frozen |
| Dropouts | `attention_dropout=0.0`, `hidden_dropout=0.0`, `feat_proj_dropout=0.0`, `layerdrop=0.0` |
| `mask_time_prob` | 0.05 for V1 and V3; value tuned in Increment 2 for V2 and V4 (search {0.05, 0.075, 0.10}) |
| Max clip length | 15 s (also drop clips < 0.5 s) |
| Seed | 42 |
| Metrics | WER, CER, tone-insensitive WER, Tone Error Rate (TER) |
| Significance | Blockwise bootstrap by speaker, 1,000 resamples, 95% CI |

The **tone-preservation principle**: any transformation of audio or text must not distort the pitch contours or tone labels that distinguish Yoruba words. Pitch-changing speed perturbation is therefore **only** used in the ablation study.

---

## 4. Tech stack

**MUST use:**

- Python 3.10+
- PyTorch (CUDA build on Kaggle; CPU build for local tests)
- Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`
- `jiwer` (WER/CER), `librosa`, `soundfile`, `torchaudio`
- `audiomentations` (noise, gain, reverberation), `pyroomacoustics` (simulated room impulse responses)
- `pandas`, `numpy`, `matplotlib`, `pyyaml`, `tqdm`
- `gradio` (web demo), `typer` or `argparse` (CLI)
- `pytest` (tests)

**Version pinning:** Pin exact versions in `requirements.txt` and `requirements-kaggle.txt` once everything works. Use the **current** `transformers` API (for example `eval_strategy` and `processing_class` in `TrainingArguments` / `Trainer`). If an API differs from this spec, follow the installed library and record it in `docs/DECISIONS.md`.

**Do not** add heavy toolkits (ESPnet, NeMo, Kaldi, fairseq). Everything goes through Hugging Face.

---

## 5. Repository layout

Create exactly this structure (you may add files, but do not rename these):

```
yoruba-asr/
├── README.md                      # user-facing readme (write a new one for humans at the end)
├── AGENT_SPEC.md                  # copy of THIS file
├── pyproject.toml                 # package metadata; installs `yoruba_asr` + console script `yoruba-asr`
├── requirements.txt               # local/dev (CPU OK)
├── requirements-kaggle.txt        # minimal extra installs needed on Kaggle
├── configs/
│   ├── base.yaml                  # shared settings (Section 6)
│   ├── v1_baseline.yaml
│   ├── v2_augmentation.yaml
│   ├── v3_domain_adaptation.yaml
│   ├── v4_hybrid.yaml
│   ├── cpt.yaml                   # continued pretraining settings
│   ├── ablation.yaml
│   └── smoke.yaml                 # tiny settings for CPU tests / Kaggle smoke run
├── src/yoruba_asr/
│   ├── __init__.py
│   ├── config.py                  # load/merge YAML, dataclasses, path discovery
│   ├── utils.py                   # seeding, logging, timing, JSON helpers, GPU info
│   ├── text.py                    # NFC, normalisation, tone stripping, tone labels
│   ├── audio.py                   # load/resample/trim/duration filter
│   ├── data/
│   │   ├── common_voice.py        # Common Voice 27.0 loader
│   │   ├── slr86.py               # Google Yoruba corpus (OpenSLR 86) loader
│   │   ├── iroyin.py              # ÌròyìnSpeech loader (manifest-based)
│   │   ├── generic.py             # generic manifest loader (path,sentence[,speaker])
│   │   ├── leakage.py             # sentence/speaker overlap checks vs. CV dev/test
│   │   └── audit.py               # corpus audit report (hours, speakers, tone coverage...)
│   ├── vocab.py                   # vocabulary build, tokenizer/processor creation
│   ├── augment/
│   │   ├── transforms.py          # tempo, noise, RIR, gain, speed (ablation only)
│   │   ├── tts.py                 # MMS-TTS Yoruba synthesis + coverage check
│   │   └── build_augmented.py     # offline generation of augmented training set
│   ├── adapt/
│   │   ├── cpt.py                 # continued self-supervised pretraining (wav2vec 2.0 objective)
│   │   └── staged.py              # staged fine-tuning orchestration (OOD -> CV)
│   ├── train/
│   │   ├── collator.py            # DataCollatorCTCWithPadding
│   │   ├── finetune.py            # CTC fine-tuning with HF Trainer
│   │   └── callbacks.py           # time budget, GPU-hours logging, resume helpers
│   ├── evaluate/
│   │   ├── metrics.py             # WER, CER, tone-insensitive WER, TER
│   │   ├── decode.py              # batched greedy decoding of a test set
│   │   ├── bootstrap.py           # blockwise (speaker) bootstrap significance
│   │   ├── effects.py             # main effects + interaction from V1–V4
│   │   ├── ablation.py            # leave-one-out augmentation runs
│   │   └── error_analysis.py      # tone confusion matrix, breakdowns, examples, plots
│   ├── inference.py               # Transcriber class (load model, transcribe file/array)
│   └── cli.py                     # `yoruba-asr` commands (Section 16.1)
├── app/
│   ├── app.py                     # Gradio web demo (Section 16.2)
│   └── assets/                    # sample clips you generate yourself (NOT Common Voice)
├── kaggle/
│   ├── build_notebook.py          # generates the Kaggle notebook from templates + package source
│   ├── templates/                 # notebook cell templates
│   └── yoruba_asr_kaggle.ipynb    # FINAL generated notebook (Section 18)
├── scripts/
│   ├── make_tiny_fixture.py       # creates synthetic tiny dataset for tests
│   └── export_results_tables.py   # results -> CSV/Markdown/LaTeX tables for the thesis
├── tests/                         # pytest suite (Section 17)
├── results/                       # created at runtime (gitignored except .gitkeep)
└── docs/
    ├── DECISIONS.md
    ├── OPEN_QUESTIONS.md
    ├── DATA_SETUP.md              # how the user downloads/uploads each dataset
    ├── KAGGLE_GUIDE.md            # step-by-step run guide for the notebook
    └── RESULTS_SCHEMA.md          # JSON/CSV output formats
```

---

## 6. Configuration system

All behaviour is driven by YAML configs merged in this order: `base.yaml` → variant config → command-line overrides (`--set key=value`). `config.py` exposes typed dataclasses.

**`configs/base.yaml` (MUST contain at least):**

```yaml
seed: 42
paths:
  input_root: "/kaggle/input"          # searched recursively for datasets
  work_dir: "/kaggle/working"          # persisted outputs
  tmp_dir: "/tmp/yoruba_asr"           # extraction/cache (not persisted)
  results_dir: "${paths.work_dir}/results"
  models_dir: "${paths.work_dir}/models"
data:
  common_voice:
    release: "27.0"
    locate_by: "train.tsv"              # find folder containing this file
  slr86: { enabled: true }
  iroyin: { enabled: true, manifest: null }   # path to manifest CSV if not auto-detected
  bibletts: { enabled: false }
  min_seconds: 0.5
  max_seconds: 15
  sampling_rate: 16000
text:
  keep_tones: true                      # training targets keep tone marks
  normal_form: "NFC"
model:
  name: "facebook/wav2vec2-xls-r-300m"
  attention_dropout: 0.0
  hidden_dropout: 0.0
  feat_proj_dropout: 0.0
  layerdrop: 0.0
  mask_time_prob: 0.05
  mask_time_length: 10
  mask_feature_prob: 0.0
  ctc_loss_reduction: "mean"
  ctc_zero_infinity: true
  freeze_feature_encoder: true
  gradient_checkpointing: true
train:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  per_device_eval_batch_size: 8
  learning_rate: 3.0e-4
  warmup_steps: 150
  max_steps: 3000
  eval_steps: 200
  save_steps: 200
  logging_steps: 50
  save_total_limit: 2
  fp16: true
  group_by_length: true
  metric_for_best_model: "wer"
  time_budget_hours: 11.0               # stop & checkpoint before Kaggle's 12 h limit
eval:
  bootstrap_resamples: 1000
  ci: 0.95
```

`config.py` MUST also provide `discover_paths()`. It finds the dataset folders under `paths.input_root` by marker files (for example `train.tsv` for Common Voice, `line_index.tsv` for SLR86) and also handles `.tar.gz`/`.zip` archives by extracting them to `tmp_dir`. Kaggle dataset slugs are chosen by the user, so **never hard-code them**.

`smoke.yaml` MUST override things for a fast end-to-end check: `max_steps: 10`, `eval_steps: 5`, at most 32 train / 16 dev / 16 test clips, a tiny randomly-initialised Wav2Vec2 config instead of the 300M model (Section 17), `fp16: false` on CPU.

---

## 7. Data sources and loaders

All loaders return a `pandas.DataFrame` with **at least** these columns:

| column | type | meaning |
|---|---|---|
| `path` | str | absolute audio path |
| `sentence` | str | raw transcript (before normalisation) |
| `speaker` | str | speaker/client id (or `"unknown"`) |
| `corpus` | str | `common_voice`, `slr86`, `iroyin`, `bibletts`, `tts` |
| `split` | str | `train`, `dev`, `test`, `other`, `ood_train`, `unlabelled` |
| `gender`, `age` | str | when available (Common Voice), else empty |

### 7.1 Common Voice Scripted Speech 27.0 — Yoruba (primary, MUST)

- The user downloads it from the **Mozilla Data Collective** (free account; it is no longer loadable from the Hugging Face Hub). They upload the `.tar.gz` as a **private** Kaggle dataset.
- Expected contents: a folder containing `train.tsv`, `dev.tsv`, `test.tsv`, `validated.tsv`, `invalidated.tsv`, `other.tsv` and a `clips/` folder of MP3 files. TSV columns include `client_id`, `path`, `sentence`, `age`, `gender` (read with `sep="\t"`, `quoting=csv.QUOTE_NONE`).
- Reference statistics for release 27.0 (2026-09-11): 8.34 h recorded, 5.96 h validated, 138 speakers, 4,970 clips. Official buckets: **train 1,468, dev 983, test 1,096**, other 1,152, invalidated 269. Your audit MUST report the actual counts found. If they differ, report it; do not stop.
- Use the **official** train/dev/test splits unchanged.
- `other.tsv` clips (unvalidated) are used **only as unlabelled audio** for continued pretraining. Never use them as labelled data.

### 7.2 Google Yoruba speech corpus — OpenSLR 86 (SHOULD)

- Licence CC BY-SA 4.0. Audio comes in female/male archives with a `line_index.tsv` that maps file ids to transcripts. **Inspect the real file format before parsing** and support the columns you actually find.
- Role: out-of-domain (OOD) labelled data for staged fine-tuning, and unlabelled audio for CPT.

### 7.3 ÌròyìnSpeech (SHOULD)

- About 42 h of Yoruba news and creative-writing speech (Ògúnrẹ̀mí et al., 2024). The text is CC BY 4.0; check the audio terms at download.
- The exact release layout is not specified here, so implement `iroyin.py` on top of `generic.py`. It reads a user-supplied manifest CSV (`path,sentence[,speaker]`) and **also** tries to auto-detect common layouts (a TSV/CSV with `path`/`audio` and `sentence`/`text`/`transcript` columns). If detection fails, raise a clear error that explains how to create the manifest.
- Role: OOD labelled data (staged fine-tuning) and unlabelled audio (CPT).

### 7.4 BibleTTS Yoruba (optional)

- Single-speaker studio audio (CC BY-SA). Disabled by default. If enabled, it is used **only** as unlabelled CPT audio.

### 7.5 Leakage control (MUST) — `data/leakage.py`

- Build `forbidden_sentences` = the normalised **tone-stripped** sentences of CV **dev + test**.
- Remove any OOD or TTS item whose normalised tone-stripped sentence is in `forbidden_sentences`, and log how many were removed.
- Assert that no CV dev/test audio path appears in any training or unlabelled pool.
- Assert that CV train, dev and test speakers do not overlap. If they do (they should not), record it in the audit.
- Write `results/leakage_report.json`.

### 7.6 Corpus audit (MUST) — `data/audit.py`

For each corpus and split, report: number of clips, total hours, number of speakers, sample-rate distribution, mean/min/max duration, clips filtered (too short, too long, corrupted), the share of transcripts containing at least one tone mark, the share of sub-dotted letters, and the character inventory. Save `results/corpus_audit.json` and `results/corpus_audit.md`. This answers Research Question 1 in the thesis.

---

## 8. Text normalisation and tone handling

Implement in `text.py`. Behaviour MUST match the original baseline (Section 22) and add the extra tone utilities below.

```python
import re, unicodedata

TONE_MARKS = {"̀", "́", "̄"}   # grave (low), acute (high), macron (mid)
SUBDOT = "̣"                               # combining dot below — NEVER stripped
PUNCT = r"[\,\?\.\!\;\:\"\“\”\‘\’\…\(\)\[\]\{\}\–\—\«\»\-]"

def nfc(s: str) -> str: return unicodedata.normalize("NFC", s)

def strip_tones(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if ch not in TONE_MARKS)
    return unicodedata.normalize("NFC", s)

def normalize_text(s: str, keep_tones: bool = True) -> str:
    s = nfc(s).lower()
    s = re.sub(PUNCT, "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return nfc(s if keep_tones else strip_tones(s))
```

**Additional MUST-haves:**

1. **Numerals.** Detect digits. If a transcript contains digits, either expand them with a Yoruba number-to-words function (only if you can implement one you trust, with tests) **or** drop the utterance and log it. Default: drop and log. Record the choice in `DECISIONS.md`.
2. **Apostrophes and hyphens inside words.** Keep the behaviour of the original `PUNCT` (hyphens and en/em dashes removed). Log the counts.
3. **Tone labelling for TER.** Write `tone_sequence(text) -> list[str]`, returning one label per **tone-bearing unit (TBU)**:
   - Decompose to NFD and walk the characters.
   - A TBU is every vowel base letter `a e i o u` (sub-dotted `ẹ ọ` count as `e`/`o` bases with a sub-dot), **plus** any `n`/`m` that directly carries a tone mark (a syllabic nasal).
   - Label each TBU from the combining marks that follow it: acute → `H`, grave → `L`, macron or none → `M`.
   - Ignore spaces and other consonants.
   - Unit-test it with: `ọkọ́ → [M, H]`, `ọkọ̀ → [M, L]`, `ọkọ → [M, M]`, `ọ̀kọ̀ → [L, L]`, `ń lọ → [H, M]`, `àwọn → [L, M]`.
4. **Both target versions.** Every dataset row stores `text_tone` (training target when `keep_tones=true`) and `text_notone` (for tone-insensitive scoring).
5. **OOD tone-quality gate.** For each OOD corpus, compute the share of transcripts with no tone marks at all. If it is more than 20%, mark the corpus `needs_manual_check` in the audit. The notebook must print a clear message telling the user to listen to a 50-utterance sample. In code, exclude a corpus that fails this gate unless `data.<corpus>.force_include: true`.

---

## 9. Audio preprocessing

Implement in `audio.py`:

- `load_audio(path, sr=16000) -> np.ndarray[float32]`: decode MP3/WAV/FLAC, convert to mono, resample to 16 kHz (librosa or torchaudio), and peak-safe normalise only if it clips.
- `trim_silence(x, top_db=30)`: trim leading/trailing silence (librosa `effects.trim`).
- Duration filter: keep `0.5 s ≤ duration ≤ 15 s` **after** trimming. Log the drops by reason.
- Corrupt or empty files: catch the exception, log the path, and skip.
- Feature extraction: `Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)`.

**Storage strategy (Kaggle disk is limited):**

- Common Voice (small): preprocess to a Hugging Face `Dataset` with `input_values`, `input_length`, `labels`, and `save_to_disk` under `work_dir/cache/` so later stages reuse it.
- Large corpora (ÌròyìnSpeech, BibleTTS): **do not** precompute float arrays (42 h of float32 is about 10 GB). Keep manifests of file paths and load and process audio **on the fly** in the collator or dataset `__getitem__`.
- Augmented audio: write as 16 kHz **16-bit FLAC** to `work_dir/augmented/`, with a manifest CSV.

---

## 10. Vocabulary and processor

Implement in `vocab.py`:

1. Build the **shared vocabulary once**, before any variant is trained. It is the union of:
   - the characters of `text_tone` in **CV train + OOD train + TTS sentences** (not CV dev/test), **and**
   - a canonical Yoruba character set: `a b d e ẹ f g h i j k l m n o ọ p r s ṣ t u w y`, every precomposed NFC form of vowels and n/m with acute/grave/macron, and the standalone combining marks `U+0300`, `U+0301`, `U+0304`, `U+0323`. Characters such as `ẹ́` have no precomposed form and stay as base + combining mark after NFC.
2. Replace the space with the word delimiter `|`, then add `[UNK]` and `[PAD]` (`[PAD]` is the CTC blank).
3. Save `vocab.json` to `work_dir/shared/vocab.json` and write its SHA-256 into `results/vocab_info.json`. **Every variant MUST load this exact file.** Training must fail if the hash differs.
4. Create `Wav2Vec2CTCTokenizer(vocab.json, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")` and a `Wav2Vec2Processor`. Save them with each model.
5. Report characters in CV dev/test that are not in the vocabulary (they become `[UNK]`). There should be none or very few.

---

## 11. Increment 1 — Baseline fine-tuning (V1)

**Goal:** a reference model with no augmentation and no adaptation.

**Steps:**

1. Run `discover_paths()`, load CV, normalise, filter durations and run the leakage checks.
2. Build or load the shared vocabulary and processor.
3. Prepare the datasets (`input_values`, `input_length`, `labels`) and cache them.
4. Load `Wav2Vec2ForCTC.from_pretrained(model.name, **dropouts, mask_time_prob=0.05, ctc_*, pad_token_id=tokenizer.pad_token_id, vocab_size=len(tokenizer))`.
5. Call `model.freeze_feature_encoder()` and `model.gradient_checkpointing_enable()`.
6. Train with `Trainer` and `TrainingArguments` per Section 3, using `group_by_length=True, length_column_name="input_length"`, `load_best_model_at_end=True`, `metric_for_best_model="wer"`, `greater_is_better=False`, `report_to="none"`.
7. `compute_metrics` returns `wer`, `cer` and `wer_no_tone` on dev (same as the original notebook).
8. Save the best model and processor to `models_dir/V1/`.
9. Run the full test evaluation (Section 15) and write `results/V1/metrics.json` and `results/V1/predictions.csv` (`path, speaker, reference, hypothesis, reference_notone, hypothesis_notone`).

**Collator (MUST match the original):**

```python
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    def __call__(self, features):
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, padding=True, return_tensors="pt")
        lab = self.processor.tokenizer.pad(labels, padding=True, return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        return batch
```

**Time-budget callback (MUST):** `callbacks.TimeBudgetCallback(hours=train.time_budget_hours)` stops training gracefully and saves a checkpoint before Kaggle's 12-hour limit. `finetune.py` MUST support `resume_from_checkpoint`, auto-detecting the latest checkpoint in the output dir **or** in an attached input dataset (the previous Kaggle version's output).

**GPU-time logging (MUST):** record wall-clock time, GPU name, peak GPU memory and steps/sec to `results/<variant>/run_info.json`.

---

## 12. Increment 2 — Hybrid augmentation module (V2)

**Goal:** add tone-preserving augmentation, train V2 from the **original** XLS-R weights, and compare with V1.

### 12.1 Offline audio transforms — `augment/transforms.py`

| Transform | Settings | Pitch | Use |
|---|---|---|---|
| Tempo perturbation | factor ∈ {0.9, 1.1}; **time-stretch with pitch preserved** (`librosa.effects.time_stretch` or a sox `tempo` equivalent) | preserved | main |
| Additive noise | noise from user-provided MUSAN subset (noise/music/babble) if available, else coloured noise generated in code; SNR ~ U(10, 30) dB | preserved | main |
| Reverberation | RIRs simulated with `pyroomacoustics` (random shoebox rooms, RT60 0.2–0.8 s), or OpenSLR RIRs if provided | preserved | main |
| Gain | U(−6, +6) dB, then clip-safe | preserved | main |
| Speed perturbation | factor ∈ {0.9, 1.1} by **resampling** (changes pitch) | **shifted** | **ablation only** |

`build_augmented.py`:

- For **each CV train clip**, create **2 augmented copies**. Each copy applies a random chain drawn from {tempo, noise, reverb, gain} (each included with p = 0.5, and at least one transform always applied), seeded by `(seed, clip_index, copy_index)` so the set is identical every run.
- Keep the original transcript (tone-preserved) for every copy.
- Write FLAC files + `manifest_augmented.csv` with the transform chain recorded per row.
- **Pitch check (MUST):** for 20 random tempo-perturbed clips, estimate F0 (`librosa.pyin`) on the original and the augmented audio. Assert that the median F0 differs by less than 3%, and log `results/V2/pitch_check.json`.

### 12.2 TTS synthetic speech — `augment/tts.py`

- Model: `facebook/mms-tts-yor` via `transformers` (`VitsModel` + `AutoTokenizer`). Its licence is CC BY-NC 4.0, so research use only; state this in the model card.
- Text source: sentences from the Common Voice `validated_sentences` / sentence pool **not** recorded in any split, **or** validated sentences that are not in CV dev/test (after the leakage filter). Normalise them first.
- **Coverage check (MUST):** before synthesising, check which characters of the normalised Yoruba text the TTS tokenizer cannot represent (especially tone marks and sub-dots). Write `results/V2/tts_coverage.json`. If tone marks are **not** representable, set `tts.enabled_effective=false`, skip TTS, and print a clear warning. Never silently feed toneless text while labelling it as toned.
- Volume: at most **one synthetic clip per real CV train clip** (≤ 1,468 by default).
- Resample to 16 kHz, apply the same duration filter, and write FLAC + `manifest_tts.csv` (flag `synthetic=true`).
- Export 50 random synthetic clips + transcripts to `results/V2/tts_listening_sample/` for the user to listen to.

### 12.3 Latent masking tuning

- Train three short runs of V2 (`max_steps=1000`, otherwise identical) with `mask_time_prob ∈ {0.05, 0.075, 0.10}` and choose the value with the lowest **dev** WER.
- Save it to `results/V2/mask_tuning.json`. V2 and V4 MUST read the chosen value from this file.
- If GPU quota is tight, allow `--skip-mask-tuning` to use 0.075, and record this.

### 12.4 Train V2

Training set = CV train (real) + augmented copies + TTS clips. Start from `facebook/wav2vec2-xls-r-300m`. Use the same hyperparameters and `max_steps=3000`. Output goes to `models_dir/V2/` and `results/V2/`.

---

## 13. Increment 3 — Domain adaptation module (V3)

### 13.1 Stage A — Continued self-supervised pretraining (CPT) — `adapt/cpt.py`

- Load `Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")`. **Check the load log.** If the quantiser (`quantizer.*`) or `project_q`/`project_hid` weights are reported as newly initialised, **stop**, write the issue to `OPEN_QUESTIONS.md`, and do not continue CPT with random quantiser weights.
- Unlabelled pool: CV `other` clips + ÌròyìnSpeech audio + SLR86 audio (+ BibleTTS if enabled). **Exclude all CV dev/test audio.** Apply the same 0.5–15 s filter. Use crops of at most 15 s; random-crop longer files.
- Objective: the standard wav2vec 2.0 contrastive loss plus the diversity loss, using the `transformers` helpers:
  - `from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices`
  - `mask_time_prob=0.65`, `mask_time_length=10`, `num_negatives=100` (from model config), `diversity_loss_weight` from config.
  - Gumbel temperature decays from 2.0 to 0.5 (factor 0.999995 per step) via `model.set_gumbel_temperature(...)`.
- Optimiser: AdamW, LR 5e-5, warm-up 500 steps, linear decay, FP16, gradient checkpointing, gradient clipping 1.0, and gradient accumulation to an effective batch of about 32 crops or 1,600 s of audio (configurable).
- Write a custom loop modelled on Hugging Face's `run_wav2vec2_pretraining_no_trainer.py` example (use `accelerate`).
- Log every 50 steps: loss, contrastive loss, diversity loss, perplexity of the codevectors, gumbel temperature and grad norm. Save the logs to `results/V3/cpt_log.csv` and a plot to `results/V3/cpt_loss.png`.
- **Budget:** about 20 GPU-hours in total, split across **two** committed Kaggle sessions (`time_budget_hours: 11`). Checkpoint every 500 steps (model, optimiser, scheduler, step, temperature, RNG states) and resume exactly.
- **Stability checks (MUST):** abort and save if the loss becomes NaN or codevector perplexity collapses (< 2) for 500 consecutive steps. Record this in the log.
- Output: `models_dir/cpt_encoder/`, saved with `save_pretrained`.

### 13.2 Stage B — Staged fine-tuning — `adapt/staged.py`

1. **B1 (OOD labelled):** load `Wav2Vec2ForCTC.from_pretrained(models_dir/cpt_encoder, vocab_size=..., pad_token_id=..., ...)` with the **shared vocabulary**. The CTC head is newly initialised here, which is expected; log it. Fine-tune on ÌròyìnSpeech + SLR86 (leakage-filtered), `max_steps=2000` (configurable), other hyperparameters as in Section 3, and select the best checkpoint on **CV dev**.
2. **B2 (CV):** continue from B1's best checkpoint and fine-tune on **CV train only**, `max_steps=3000`, `mask_time_prob=0.05`.
3. Output: `models_dir/V3/`, `results/V3/`.

If no OOD labelled corpus is available, B1 is skipped. V3 then becomes CPT + CV fine-tuning; record this in `DECISIONS.md` and the results JSON (`"stage_b1": "skipped"`).

---

## 14. Increment 4 — Hybrid system (V4)

- Start from `models_dir/cpt_encoder/`, run **B1 exactly as for V3** (you may reuse V3's B1 checkpoint, since it is identical by construction), then run **B2 on the V2 training set** (CV train + augmented + TTS) with the **tuned `mask_time_prob`** and `max_steps=3000`.
- Output: `models_dir/V4/`, `results/V4/`.
- Build `inference.Transcriber` (Section 16) and test it on V4.

---

## 15. Evaluation, significance testing, ablation, error analysis

### 15.1 Metrics — `evaluate/metrics.py`

- `wer(refs, hyps)`, `cer(refs, hyps)` with `jiwer`, on tone-preserved normalised text.
- `wer_no_tone`: WER after `strip_tones` on both sides.
- `tone_error_rate(refs, hyps)`: for each utterance, compute the edit distance (substitutions + deletions + insertions) between `tone_sequence(ref)` and `tone_sequence(hyp)`. TER = Σ edits / Σ len(tone_sequence(ref)) × 100.
- Secondary metric `ter_matched_words`: align words with `jiwer` alignment; for word pairs whose tone-stripped forms are **equal**, count the TBUs whose tone labels differ, divided by the TBUs in those words. This isolates pure tone errors.
- Also report the tone confusion counts (3 × 3, H/M/L) from `ter_matched_words` alignments.

### 15.2 Test evaluation — `evaluate/decode.py`

Batched greedy decoding of the full CV test set (after the duration filter) with the best checkpoint. Save `predictions.csv` and `metrics.json`:

```json
{"variant": "V1", "n_utts": 1090, "wer": 0.0, "cer": 0.0, "wer_no_tone": 0.0,
 "ter": 0.0, "ter_matched_words": 0.0, "tone_confusion": [[0,0,0],[0,0,0],[0,0,0]],
 "train_clips": 0, "train_hours": 0.0, "gpu_hours": 0.0, "mask_time_prob": 0.05,
 "checkpoint": "models/V1", "vocab_sha256": "..."}
```

### 15.3 Blockwise bootstrap — `evaluate/bootstrap.py`

Following Liu & Peng (2020): resample **speakers** (blocks) with replacement 1,000 times. For each resample, recompute the corpus-level WER of system A and system B over all utterances of the sampled speakers, then take Δ = WER_A − WER_B. Report the mean Δ, the 95% percentile CI and `significant = CI excludes 0`. Do the same for TER. Compare these pairs: V2–V1, V3–V1, V4–V1, V4–V2, V4–V3. Output: `results/significance.csv`.

### 15.4 Factorial effects — `evaluate/effects.py`

- Augmentation main effect = mean[(V2 − V1), (V4 − V3)]
- Adaptation main effect = mean[(V3 − V1), (V4 − V2)]
- Interaction = (V4 − V3) − (V2 − V1)

Compute these for WER, CER, wer_no_tone and TER. Output: `results/factorial_effects.csv`.

### 15.5 Ablation — `evaluate/ablation.py`

Leave-one-out runs of V2 with `max_steps=1500` each: −tempo, −noise, −reverb, −TTS, −tuned-masking (use 0.05), plus **+speed perturbation** (pitch-changing). Each is compared with a **V2 run of the same 1,500 steps**. Output: `results/ablation.csv` with dev and test WER/TER. These runs MUST be resumable and selectable individually (`--only no_tts`) so they can be spread across Kaggle sessions.

### 15.6 Error analysis — `evaluate/error_analysis.py`

- Per variant: the share of purely tonal word errors (WER − wer_no_tone), the H/M/L confusion matrix plot, and WER/TER by gender, by age group (where the CV metadata exists), by utterance length bucket (<3 s, 3–6 s, 6–10 s, >10 s) and by whether a word contains sub-dotted letters.
- 20 qualitative examples per variant: reference vs hypothesis, with errors highlighted in Markdown.
- Figures (PNG, 300 dpi, matplotlib, readable in black-and-white print): a bar chart of WER/CER/TER for V1–V4 with bootstrap CIs, a factorial interaction plot, the ablation bar chart, a CPT loss curve, and fine-tuning dev-WER curves.
- `scripts/export_results_tables.py` produces `results/thesis_tables.md` (and `.csv`) ready to paste into Chapter Four.

### 15.7 Zero-shot references (optional, if quota allows)

Zero-shot MMS (`facebook/mms-1b-all` with the Yoruba adapter `target_lang="yor"`) and Whisper (a small/medium checkpoint, language Yoruba) on the same test set, with the same normalisation. Report them separately as **reference points, not part of the factorial comparison**. Whisper output may lack diacritics; report both WER and wer_no_tone.

---

## 16. The application (CLI + web demo)

### 16.1 CLI — `yoruba-asr` (MUST)

```
yoruba-asr audit        --config configs/base.yaml
yoruba-asr prepare      --config configs/base.yaml           # load, normalise, filter, leakage, vocab, cache
yoruba-asr augment      --config configs/v2_augmentation.yaml
yoruba-asr tune-mask    --config configs/v2_augmentation.yaml
yoruba-asr cpt          --config configs/cpt.yaml [--resume]
yoruba-asr train        --config configs/v1_baseline.yaml [--resume]   # also v2/v3/v4
yoruba-asr evaluate     --variant V1 [--variant V2 ...]
yoruba-asr significance
yoruba-asr effects
yoruba-asr ablation     --config configs/ablation.yaml [--only no_tts]
yoruba-asr analyse
yoruba-asr transcribe   path/to/audio.wav --model models/V4 [--no-tone]
yoruba-asr export-tables
```

Every command prints a short summary at the end and writes its outputs under `results/`.

### 16.2 Web demo — `app/app.py` (Gradio, MUST)

**Purpose:** let the supervisor and examiners try the models and see the results during the defence.

**Tabs:**

1. **Transcribe**
   - Input: microphone recording **or** file upload (wav/mp3/flac/m4a/ogg), resampled to 16 kHz, max 30 s (longer audio is split into ≤ 15 s chunks on silence and the transcriptions joined).
   - Model selector: V1 / V2 / V3 / V4 (only those found on disk or on the Hub are shown; V4 by default).
   - Output: the transcription **with tone marks**, the same text **without tone marks**, the processing time and the model name.
   - Optional "Reference text" box: if filled, show the WER, CER and TER for this clip, with the differences highlighted.
2. **Compare models**
   - The same audio run through every available variant. A table shows one row per model with its transcription (and per-clip metrics if a reference is given).
3. **Results**
   - Loads `results/summary.csv`, `significance.csv`, `factorial_effects.csv` and `ablation.csv` if present. Shows them as tables and as the figures from Section 15.6. Shows "No results yet" gracefully if they are missing.
4. **About**
   - The project title, the four variants, the dataset licences (Common Voice CC0; MMS-TTS non-commercial), the limitations (read speech, Standard Yoruba, limited speakers) and how to cite.

**Model loading:** `Transcriber.from_dir(path)` or `Transcriber.from_hub(repo_id)`. A config file `app/models.yaml` maps the variant name to a local path or Hub repo id. Cache loaded models (load lazily, keep at most 2 in memory). It runs on CPU (expect a few seconds per clip); use the GPU if available.

**Must not:** ship any Common Voice audio in `app/assets/`. For demo samples, synthesise 3 clips with MMS-TTS or ask the user to add their own recordings.

**Run:** `python app/app.py` (local) or deploy to a Hugging Face Space (write `app/README_SPACE.md` with Space instructions and a `requirements.txt`). Optionally provide a `Dockerfile`.

### 16.3 `inference.Transcriber` (MUST)

```python
class Transcriber:
    @classmethod
    def from_dir(cls, path: str, device: str | None = None) -> "Transcriber": ...
    @classmethod
    def from_hub(cls, repo_id: str, device: str | None = None) -> "Transcriber": ...
    def transcribe_array(self, audio: np.ndarray, sr: int) -> dict:   # {"text", "text_notone", "seconds"}
    def transcribe_file(self, path: str) -> dict: ...
    def transcribe_batch(self, paths: list[str], batch_size: int = 8) -> list[dict]: ...
```

---

## 17. Testing requirements

Use `pytest`; all tests MUST pass on CPU in under ~5 minutes with no internet, using fixtures generated by `scripts/make_tiny_fixture.py`. That script creates a fake Common Voice folder (TSVs + 40 short sine/noise WAV clips with Yoruba transcripts written by you), a fake SLR86 folder and a fake manifest corpus.

**Required tests:**

| Test file | What it checks |
|---|---|
| `test_text.py` | NFC handling; `strip_tones` keeps sub-dots; `normalize_text` removes punctuation; `tone_sequence` cases from Section 8 |
| `test_audio.py` | resampling to 16 kHz; mono conversion; duration filter; corrupt-file skip |
| `test_loaders.py` | CV loader reads the TSVs/columns; SLR86 and manifest loaders; `discover_paths` finds folders and extracts archives |
| `test_leakage.py` | dev/test sentences are removed from OOD/TTS sets; dev/test audio is never in pools |
| `test_vocab.py` | vocabulary contains the canonical Yoruba set; hash is stable; tokenizer round-trip of tone-marked text |
| `test_augment.py` | augmentation is deterministic with a seed; tempo keeps pitch (synthetic tone F0 within 3%); speed changes pitch |
| `test_metrics.py` | WER/CER on known pairs; TER on hand-computed examples; `ter_matched_words`; bootstrap on a toy case with a known answer |
| `test_effects.py` | main-effect and interaction formulas |
| `test_train_smoke.py` | 10-step fine-tune with a **tiny random Wav2Vec2ForCTC** (e.g. hidden 32, 2 layers, 2 conv layers) runs, saves, resumes and evaluates |
| `test_cpt_smoke.py` | 10-step CPT with a tiny random `Wav2Vec2ForPreTraining` runs and the loss is finite |
| `test_inference.py` | `Transcriber` loads the smoke model and returns text |
| `test_app.py` | Gradio app builds without launching; Results tab handles missing files |
| `test_notebook.py` | the generated `.ipynb` is valid JSON/nbformat, has the parameter cell first, and every `STAGE` value maps to code |

Also provide `make smoke` (or `python -m yoruba_asr.cli ... --config configs/smoke.yaml`). It runs **prepare → augment → cpt → train V1–V4 → evaluate → significance → effects → analyse** end to end on the tiny fixture with the tiny model in a few minutes on CPU.

---

## 18. The final Kaggle notebook (main hand-over)

**Deliver this only after Sections 5–17 are complete and the smoke pipeline passes.**

### 18.1 Form

- One file: `kaggle/yoruba_asr_kaggle.ipynb`, generated by `kaggle/build_notebook.py`. Commit both.
- **Self-contained:** the notebook embeds the `yoruba_asr` package source. `build_notebook.py` writes each module into the notebook with `%%writefile` cells, then runs `pip install -e .`. The user only needs to attach their data datasets, not a code dataset.
- It must run top to bottom with **Save Version → Save & Run All (Commit)**, with no manual input after the parameter cell.

### 18.2 Cell structure (in this order)

1. **Title & instructions (Markdown):** what the notebook does, the required Kaggle settings (Accelerator: GPU T4 or P100; Internet: On; Persistence: Files), and the datasets to attach.
2. **PARAMETERS (code, tagged `parameters`) — the only cell the user edits:**

```python
STAGE   = "prepare"   # one of: "smoke", "prepare", "V1", "augment", "tune_mask", "V2",
                      #         "cpt", "V3", "V4", "evaluate", "ablation", "analyse", "all_light"
RESUME  = True        # resume from checkpoints found in /kaggle/working or attached previous outputs
ABLATION_ONLY = None  # e.g. "no_tts" to run one ablation per session
PUSH_TO_HUB = False   # needs Kaggle Secret HF_TOKEN
HUB_REPO_PREFIX = "your-hf-username/yoruba-xlsr"   # used only if PUSH_TO_HUB
CPT_MAX_HOURS = 11.0
EXTRA_OVERRIDES = {}  # e.g. {"train.max_steps": 3000}
```

3. **Environment check:** print GPU name/memory, CUDA, torch and transformers versions, disk space in `/kaggle/working` and `/tmp`, and the remaining quota warning text. Set `CUDA_VISIBLE_DEVICES=0`.
4. **Install:** `pip install -q` from the pinned `requirements-kaggle.txt` (skip packages already in the Kaggle image when the versions are compatible).
5. **Package source:** `%%writefile` cells for every module, configs and `pyproject.toml`, then `pip install -e .`.
6. **Locate data:** run `discover_paths()` and print what was found (Common Voice, SLR86, ÌròyìnSpeech, MUSAN, previous outputs). Raise a clear error if Common Voice is missing.
7. **Previous outputs:** if the user attached the output of an earlier notebook version as an input dataset, copy or symlink its `models/`, `cache/`, `shared/` and `results/` into `/kaggle/working`. This chains sessions (prepare → V1 → … ).
8. **Run stage:** a dispatcher that calls the CLI functions for `STAGE`.
9. **Summary:** print the key metrics so far, list the files written, show any figures produced inline, and (if `PUSH_TO_HUB`) push the model folder for this stage.
10. **Next step (Markdown printed by code):** tell the user which `STAGE` to run next and which outputs to attach.

### 18.3 Session plan (print this in the notebook and in `docs/KAGGLE_GUIDE.md`)

| Run | `STAGE` | Attach as input | Approx. GPU hours (T4) |
|---|---|---|---|
| 0 | `smoke` | Common Voice | < 0.3 |
| 1 | `prepare` | Common Voice (+ SLR86, ÌròyìnSpeech, MUSAN) | ~0.5 |
| 2 | `V1` | data + output of run 1 | 4–5 |
| 3 | `augment` + `tune_mask` | data + previous output | 3–4 |
| 4 | `V2` | data + previous output | 4–5 |
| 5–6 | `cpt` (twice, resumes) | data + previous output | ~20 |
| 7 | `V3` | data + previous output | 5–6 |
| 8 | `V4` | data + previous output | 4–5 |
| 9 | `evaluate` (+ significance, effects) | previous output | ~1 |
| 10+ | `ablation` (one per session if needed) | previous output | ~1.5 each |
| last | `analyse` | previous output | < 0.5 |

**These times are estimates**; the notebook MUST measure and log actual GPU time per stage. Kaggle's weekly GPU quota is about 30 hours (shown in account settings), so the full plan spans about two weeks.

### 18.4 Output contract

When the run finishes, `/kaggle/working/` MUST contain:

```
results/  (all JSON/CSV/PNG/MD described above, plus run_info.json per stage)
models/V1..V4/  (best checkpoint + processor, each with a model_card.md)
models/cpt_encoder/
shared/vocab.json
cache/  (reusable prepared datasets; can be deleted if disk is tight)
```

Keep `/kaggle/working` under **~18 GB**. Delete intermediate checkpoints after the best one is saved, store audio as FLAC, and never write raw float arrays for large corpora.

### 18.5 Notebook acceptance test

Before delivery, run the generated notebook locally with `STAGE="smoke"` (through `jupyter nbconvert --execute` using the tiny fixture and the tiny model). Also confirm that every other `STAGE` resolves to a function without error in dry-run mode (`--dry-run` prints the plan without training).

---

## 19. Kaggle platform constraints

- **GPU:** one NVIDIA T4 (16 GB) or P100 (16 GB). P100 has no tensor cores (FP16 still works but is slower). T4 does not support bf16, so use FP16.
- **Session limit:** 12 hours of execution per GPU session. Design for `time_budget_hours=11`.
- **Weekly GPU quota:** about 30 hours (check in Kaggle settings; it can vary).
- **Disk:** `/kaggle/working` is persisted with the version (about 20 GB). `/tmp` is larger but **not** persisted. `/kaggle/input` is read-only.
- **Internet:** must be turned on in the notebook settings (phone-verified account) to download models and pip packages.
- **Secrets:** an HF token in Kaggle *Add-ons → Secrets*, read with `from kaggle_secrets import UserSecretsClient` only if `PUSH_TO_HUB=True`. Never print tokens.
- **Datasets the user uploads (private):** Common Voice Yoruba 27.0 archive (required); SLR86 archives; the ÌròyìnSpeech release; optionally a MUSAN subset (noise, plus a little music and speech, ≤ 2 GB).

---

## 20. Milestones and definition of done

| Milestone | Done when |
|---|---|
| M0 Scaffold | Repo layout, configs, `pyproject`, CI-style `pytest` running on the fixture |
| M1 Data + V1 | Audit, leakage report, shared vocab, V1 training and evaluation work on the fixture (smoke) and are ready for Kaggle |
| M2 Augmentation + V2 | Transforms + pitch check, TTS + coverage check, mask tuning, V2 smoke run |
| M3 Adaptation + V3 | CPT loop with checkpoint/resume and stability checks, staged FT, V3 smoke run |
| M4 Hybrid + V4 | V4 smoke run, `Transcriber` works |
| M5 Evaluation | Metrics incl. TER, bootstrap, effects, ablation runner, error analysis, thesis tables and figures |
| M6 App | Gradio app with all 4 tabs; CLI `transcribe`; Space instructions |
| M7 Kaggle notebook | Generated self-contained notebook passes the acceptance test (Section 18.5) |
| M8 Docs | Human `README.md`, `DATA_SETUP.md`, `KAGGLE_GUIDE.md`, `RESULTS_SCHEMA.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md` complete |

**Definition of done (overall):**

- [ ] All tests pass on CPU; `make smoke` runs end to end.
- [ ] No Common Voice audio or text is committed to the repo.
- [ ] The four variants share one vocabulary file (hash checked) and identical hyperparameters except the treatments.
- [ ] The CV test set is used only in final evaluation (verified by `leakage_report.json`).
- [ ] Every stage is resumable and respects the 11-hour budget.
- [ ] The Kaggle notebook is self-contained and documented, and prints the next step after every run.
- [ ] The app runs locally on CPU and shows results when present.

---

## 21. Final deliverables checklist

Hand back to the user:

1. The repository (zipped, or a GitHub link if a repo was provided).
2. **`kaggle/yoruba_asr_kaggle.ipynb`**: the training notebook, the main hand-over.
3. `docs/KAGGLE_GUIDE.md`: a click-by-click guide covering how to download Common Voice from the Mozilla Data Collective, upload private datasets to Kaggle, set the accelerator and internet, run each `STAGE`, attach previous outputs, and find results.
4. `docs/DATA_SETUP.md`: where to get each dataset, its licence, and the expected folder layout or manifest format.
5. The app (`app/app.py`) with run instructions and optional Hugging Face Space files.
6. A short **handover note** listing what was tested, what was assumed (`DECISIONS.md`) and what the user must check (`OPEN_QUESTIONS.md`), for example the ÌròyìnSpeech layout, MMS-TTS tone coverage and the actual Kaggle quota.

---

## 22. Reference code from the original Colab baseline

The project started from this Colab notebook (Hugging Face XLSR fine-tuning recipe adapted for Yoruba). Your implementation MUST reproduce its baseline behaviour (normalisation, vocab construction, collator, metrics, model settings), with these changes:

- Kaggle paths instead of Google Drive.
- A **shared** vocabulary built as in Section 10 (the original built it from train + dev).
- `max_steps=3000` instead of `num_train_epochs=30`.
- A time-budget callback and resume support.

Key parts of the original:

```python
# Config
MODEL_NAME  = "facebook/wav2vec2-xls-r-300m"
KEEP_TONES  = True
MAX_SECONDS = 15
SEED = 42

# Loading Common Voice splits
def load_split(name):
    df = pd.read_csv(f"{yo_dir}/{name}.tsv", sep="\t", quoting=csv.QUOTE_NONE, usecols=["path", "sentence"])
    df["path"] = df["path"].apply(lambda p: f"{yo_dir}/clips/{p}")
    return df.dropna().reset_index(drop=True)

# Vocabulary (original version — replace with Section 10)
chars = sorted(set("".join(train_df.sentence) + "".join(dev_df.sentence)))
vocab = {c: i for i, c in enumerate(chars)}
vocab["|"] = vocab.pop(" ")
vocab["[UNK]"] = len(vocab); vocab["[PAD]"] = len(vocab)

# Feature extraction / labels
def prepare(batch):
    audio, _ = librosa.load(batch["path"], sr=16000)
    batch["input_values"] = feature_extractor(audio, sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Metrics
def compute_metrics(pred):
    ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
    hyp = tokenizer.batch_decode(ids)
    ref = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=hyp, references=ref),
            "cer": cer_metric.compute(predictions=hyp, references=ref),
            "wer_no_tone": wer_metric.compute(predictions=[strip_tones(h) for h in hyp],
                                              references=[strip_tones(r) for r in ref])}

# Model
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
    mask_time_prob=0.05, layerdrop=0.0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
    pad_token_id=tokenizer.pad_token_id, vocab_size=len(tokenizer))
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# Training arguments (original — change num_train_epochs to max_steps=3000)
args = TrainingArguments(
    output_dir=OUT_DIR, group_by_length=True, length_column_name="input_length",
    per_device_train_batch_size=8, gradient_accumulation_steps=4, per_device_eval_batch_size=8,
    eval_strategy="steps", eval_steps=200, save_steps=200, logging_steps=50,
    num_train_epochs=30, learning_rate=3e-4, warmup_steps=150, fp16=True, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
    report_to="none", seed=SEED)
```

---

### Key references (for model cards and docs)

- Baevski et al. (2020), *wav2vec 2.0*, NeurIPS.
- Babu et al. (2022), *XLS-R*, Interspeech.
- Pratap et al. (2024), *Scaling Speech Technology to 1,000+ Languages* (MMS), JMLR.
- Ardila et al. (2020), *Common Voice*, LREC.
- Gutkin et al. (2020), *Developing an Open-Source Corpus of Yoruba Speech*, Interspeech.
- Ògúnrẹ̀mí et al. (2024), *ÌròyìnSpeech*, LREC-COLING.
- Liu & Peng (2020), *Statistical Testing on ASR Performance via Blockwise Bootstrap*, Interspeech.
- Olusanya (2026), *Tone in Yoruba ASR*, LoResLM.

*End of specification.*
