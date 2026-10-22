# Kaggle guide: training and evaluating V1–V4 step by step

You train everything with **one notebook**, `kaggle/yoruba_asr_kaggle.ipynb`, and run **one stage per Kaggle session**. Each run's output becomes the input of the next run, so the work chains across sessions.

---

## Quickest start: no upload at all (OpenSLR 86)

If uploading Common Voice gives you trouble, the notebook can train on the **Google Yoruba speech corpus (OpenSLR 86)**, which it **downloads by itself**:
1. Import the notebook, then set **Accelerator: GPU T4** and **Internet: On**. You do not need to attach any dataset.
2. In the PARAMETERS cell keep `DOWNLOAD_OPENSLR86 = True` and set `PRIMARY_DATASET = "slr86"`. With `"auto"` (the default), the notebook uses Common Voice when it is attached and complete, and OpenSLR 86 otherwise.
3. Run the stages exactly as in the plan below. Each session re-downloads the ~0.9 GB archives into `/tmp` (1–3 minutes).

OpenSLR 86 has 3,583 studio recordings (about 3–4 h) from 36 speakers (19 female, 17 male), 48 kHz, licence CC BY-SA 4.0. The notebook splits it **by speaker** into train/dev/test (80/10/10, seeded), removes training sentences that also appear in dev/test, strips annotation tags such as `[breath]` and `[snap]`, and drops the 16 recordings marked `[abrupt]`. The dataset choice is written to `results/data_summary.json` (`primary_dataset`). Use the same choice for every stage.

> For the thesis: Common Voice 27.0 is the dataset named in the research design. If you train on OpenSLR 86 instead, say so in Chapter Three. In that case V3/V4's out-of-domain stage (B1) uses ÌròyìnSpeech only, and is skipped if ÌròyìnSpeech is not attached.

---

## 0. One-time setup

### 0.1 Kaggle account
1. Create or log in to your account at kaggle.com.
2. **Verify your phone number** (*Settings → Phone verification*). Without it you cannot use GPUs or turn on Internet in notebooks.
3. Check your weekly GPU quota in *Settings → Quota*. It is usually about 30 h per week. The full plan needs about 45–55 GPU hours, so expect to spread it over **two to three weeks**.

### 0.2 Download Common Voice Yoruba (required)
1. Open the **Mozilla Data Collective** website (Common Voice is no longer available on the Hugging Face Hub) and create a free account.
2. Find **Common Voice Scripted Speech 27.0 – Yoruba** and download the `.tar.gz` (about 175 MB; your file is called `…cv-corpus-27.0-2026-09-11-yo.tar.gz`).
3. Do not extract it. Kaggle accepts the archive as it is.

> Your PC already has this archive in `yoruba_asr_xlsr_colab/model code and Dataset from mozilla collective/`. The **extracted** folder next to it is incomplete: it has only 3,180 of the 4,970 clips. Always upload the `.tar.gz`.

### 0.3 Upload it as a PRIVATE Kaggle dataset
1. Go to *Datasets → New Dataset*.
2. Drag in the `.tar.gz`, give it a title such as `cv-yoruba-27`, and keep **Visibility: Private**. Common Voice's terms forbid re-hosting it publicly.
3. Click *Create*. Kaggle may extract the archive itself; the notebook handles both the archive and an extracted folder.

### 0.4 Optional datasets (recommended for V3/V4)
Upload each one as its own **private** dataset (details in `DATA_SETUP.md`):
- **OpenSLR 86**: `yo_ng_female.zip` and `yo_ng_male.zip` from https://www.openslr.org/86/
- **ÌròyìnSpeech**: the audio plus a `manifest.csv` (`path,sentence,speaker`). Put the word **iroyin** in the folder or dataset name.
- **MUSAN** subset (optional): a folder named `musan` containing noise/music/speech WAVs, at most ~2 GB.

Without OOD data, V3/V4 still run: stage B1 is skipped and recorded as `"stage_b1": "skipped"`. CPT then uses only CV `other` clips, which is a weak domain-adaptation signal (see `OPEN_QUESTIONS.md`).

### 0.5 Create the notebook
1. *Code → New Notebook → File → Import Notebook* and select `kaggle/yoruba_asr_kaggle.ipynb`.
2. Right-hand panel → *Session options*:
   - **Accelerator: GPU T4** (T4 ×2 also works; only one GPU is used) or **P100**.
   - **Internet: On**.
3. *Add Input* → *Your Datasets* → add Common Voice (plus the optional datasets).

---

## 1. Running a stage

1. In the **PARAMETERS** cell (the first code cell) set `STAGE`, for example `STAGE = "prepare"`. Leave the other values as they are.
2. Click **Save Version** → *Save & Run All (Commit)* → *Save*. The run happens in the background, so you can close the browser.
3. When it finishes, open the version (*Versions*) → **Output** to see the files, and scroll to the end of the log: the last cell prints the **next STAGE**.
4. **Chain to the next run:** edit the notebook → *Add Input* → *Your Work → Notebooks →* this notebook. This attaches the output of the latest version. Keep the data datasets attached. Set the next `STAGE` and *Save & Run All* again.

The notebook copies the previous output (`shared/`, `cache/`, `manifests/`, `models/`, `checkpoints/`, `augmented/`, `results/`) into `/kaggle/working`, so each run continues where the last one stopped.

---

## 2. Session plan

| Run | `STAGE` | Attach as input | GPU h (T4, estimate) | What it produces |
|---|---|---|---|---|
| 0 | `smoke` | Common Voice | < 0.3 | tiny-model check of the whole pipeline on 32/16/16 real clips (`smoke/` folder) |
| 1 | `prepare` | Common Voice (+ SLR86, ÌròyìnSpeech, MUSAN) | ~0.5 | audit, leakage report, **shared vocabulary**, cached CV train/dev/test |
| 2 | `V1` | data + output of run 1 | 4–5 | `models/V1`, `results/V1/metrics.json` |
| 3 | `augment+tune_mask` | data + previous output | 3–4 | augmented FLAC, TTS clips, pitch check, `results/V2/mask_tuning.json` |
| 4 | `V2` | data + previous output | 4–5 | `models/V2` |
| 5–6 | `cpt` (twice; it resumes) | data + previous output | ~20 | `models/cpt_encoder`, `results/V3/cpt_log.csv` |
| 7 | `V3` | data + previous output | 5–6 | `models/B1`, `models/V3` |
| 8 | `V4` | data + previous output | 4–5 | `models/V4` |
| 9 | `evaluate` | previous output | ~1 | `summary.csv`, `significance.csv`, `factorial_effects.csv` |
| 10+ | `ablation` (use `ABLATION_ONLY` for one run per session if quota is tight) | data + previous output | ~1.5 each | `ablation.csv` |
| last | `analyse` | previous output | < 0.5 | error analysis, figures, `thesis_tables.md/.tex` |

The hours are estimates; the actual GPU time of every stage is logged in `results/**/run_info.json`. If your quota is tight, set `EXTRA_OVERRIDES = {"mask_tuning.skip": True}` in run 3. This uses mask_time_prob 0.075 and records the choice.

**Ablation names:** `v2_ref`, `no_tempo`, `no_noise`, `no_reverb`, `no_tts`, `no_tuned_mask`, `plus_speed`. Run `v2_ref` first. Every row is compared against it.

---

## 3. When a run stops early

- **Time budget:** training stops by itself at about 11 h, before Kaggle's 12 h limit, and saves a checkpoint. The last cell then says *"run the same STAGE again"*. Attach the new output and run the same `STAGE` with `RESUME = True`.
- **CPT** is designed to take two sessions: run `STAGE = "cpt"` until the log says `completed = True` (about 20 GPU hours in total).
- **Out of memory (CUDA OOM):** set `EXTRA_OVERRIDES = {"train.per_device_train_batch_size": 4, "train.gradient_accumulation_steps": 8}`. The effective batch stays at 32, so the variants remain comparable. Record the change in `DECISIONS.md`.
- **Disk full:** `/kaggle/working` holds about 20 GB. Checkpoints are deleted when a variant finishes, and `models/B1` is deleted after V4. If disk is still short, delete `cache/` in a run that no longer needs it (evaluate/analyse do need it).
- **"Common Voice was NOT found":** the dataset is not attached, or it was uploaded somewhere else. Use *Add Input* again.

---

## 4. Getting the results

After `analyse`, open the last version → **Output** → *Download*:
- `results/thesis_tables.md` / `.tex`: tables for Chapter Four
- `results/figures/*.png`: 300 dpi figures
- `results/corpus_audit.md`, `leakage_report.json`, `significance.csv`, `factorial_effects.csv`, `ablation.csv`
- `models/V1` … `models/V4`: copy them into `yoruba-asr/models/` on your PC to use the web app

## 5. Pushing models to the Hugging Face Hub (optional)
1. Create a token at huggingface.co → *Settings → Access Tokens* (write access).
2. In Kaggle: *Add-ons → Secrets → Add* with label **HF_TOKEN**, then attach it to the notebook.
3. Set `PUSH_TO_HUB = True` and `HUB_REPO_PREFIX = "your-username/yoruba-xlsr"`. After a `V1`…`V4` run, the model is uploaded to a **private** repo named `your-username/yoruba-xlsr-v1` (and so on).

## 6. Optional: quick sanity run
`STAGE = "all_light"` runs every stage with very short budgets (300 steps, no mask tuning, a short CPT) in a separate `light/` folder. It checks the whole real-data path in one session (~3–4 h), but **its numbers are not for the thesis**.
