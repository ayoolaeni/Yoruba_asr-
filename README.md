# Yoruba ASR: hybrid data augmentation + domain adaptation on XLS-R

**Enhancing ASR Accuracy for Low-Resource Nigerian Tonal Languages Through Hybrid Data Augmentation and Domain Adaptation: A Case Study of Yoruba** (final-year project, Bolaji Oluseye)

A reproducible Yoruba speech-recognition system built on **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`, wav2vec 2.0 + CTC). It trains four variants in a 2 × 2 factorial design:

| Variant | Data augmentation | Domain adaptation |
|---|---|---|
| **V1** source-only baseline | – | – |
| **V2** augmentation only | tempo, noise, reverb, gain (all pitch-preserving) + MMS-TTS speech | – |
| **V3** adaptation only | – | continued wav2vec 2.0 pretraining → OOD fine-tuning → CV fine-tuning |
| **V4** hybrid | ✓ | ✓ |

The four variants share the backbone, the **hash-checked vocabulary** (tone marks and sub-dots kept), the data splits and every hyperparameter. The models are evaluated with **WER, CER, tone-insensitive WER and Tone Error Rate (TER)**, plus speaker-block bootstrap significance tests, factorial effects, an ablation study and error analysis.

## What is in this repository
| Part | Where |
|---|---|
| Python package (data, augmentation, CPT, training, evaluation, inference, CLI) | `src/yoruba_asr/` |
| **Kaggle training notebook (main hand-over)** | `kaggle/yoruba_asr_kaggle.ipynb` |
| Web demo (Gradio) | `app/app.py` |
| Configs | `configs/*.yaml` |
| Tests (67, CPU, offline, about 1 min) | `tests/` |
| Guides | `docs/KAGGLE_GUIDE.md`, `docs/DATA_SETUP.md`, `docs/RESULTS_SCHEMA.md`, `docs/DECISIONS.md`, `docs/OPEN_QUESTIONS.md` |

## Training on Kaggle (where the real models are trained)
Read **`docs/KAGGLE_GUIDE.md`**. In short:
1. Upload the Common Voice 27.0 Yoruba `.tar.gz` as a **private** Kaggle dataset (plus OpenSLR 86 and ÌròyìnSpeech if you have them).
2. Import `kaggle/yoruba_asr_kaggle.ipynb`, then set Accelerator = **GPU T4** and Internet = **On**.
3. Set `STAGE` in the first cell and run *Save & Run All*. Then attach that run's output and run the next stage: `prepare → V1 → augment+tune_mask → V2 → cpt (×2) → V3 → V4 → evaluate → ablation → analyse`.

The notebook is self-contained (it writes the package source itself), resumes from checkpoints, stops safely before the 12 h limit and prints the next step after every run.

## Local installation (Windows/Linux/macOS, CPU is fine)
```bash
cd yoruba-asr
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

## Running the web app with Docker (recommended)
1. Install Docker Desktop and start it.
2. Put the trained models in `models/V1` … `models/V4`. They are **not** in this GitHub repository (1.2 GB each); they ship in the client package or the Kaggle output. Use `python scripts/import_kaggle_output.py <kaggle_output.zip>` to copy them in.
3. Windows: double-click `START_WINDOWS.bat`. macOS/Linux: run `bash start_mac_linux.sh`. Either way, this runs `docker compose up -d --build`.
4. Open http://localhost:7870. Stop with `STOP_WINDOWS.bat` / `bash stop_mac_linux.sh`.

The step-by-step guide for non-technical users is `HOW_TO_RUN.txt`. `python scripts/make_client_package.py` builds the ready-to-send folder and zip (app + models + results + scripts).

## Running the web app without Docker
1. Copy the trained models from the Kaggle output (`models/V1` … `models/V4`) into `yoruba-asr/models/`, and optionally `results/` into `yoruba-asr/results/`.
2. Run `python app/app.py` and open http://127.0.0.1:7870. (Port 7870 is used because another project's Docker container already occupies 7860 on this PC. Override with `set GRADIO_SERVER_PORT=xxxx`.)

The tabs are:
- **Transcribe**: microphone or upload (wav/mp3/flac/m4a/ogg, up to 30 s). It shows the text with and without tone marks, the processing time and, if you type a reference, WER/CER/TER with the differences highlighted.
- **Compare models**: runs one clip through every available variant.
- **Results**: thesis tables and figures.
- **About**.

Models can also come from the Hugging Face Hub: edit `app/models.yaml`. Space deployment is described in `app/README_SPACE.md`; there is also a `Dockerfile`.

## Command line
```bash
yoruba-asr audit        --config configs/base.yaml
yoruba-asr prepare      --config configs/base.yaml
yoruba-asr augment      --config configs/v2_augmentation.yaml
yoruba-asr tune-mask    --config configs/v2_augmentation.yaml [--skip-mask-tuning]
yoruba-asr cpt          --config configs/cpt.yaml [--resume]
yoruba-asr train        --config configs/v1_baseline.yaml [--resume]     # v2_augmentation / v3_domain_adaptation / v4_hybrid
yoruba-asr evaluate     --variant V1 --variant V2 ...
yoruba-asr significance
yoruba-asr effects
yoruba-asr ablation     --config configs/ablation.yaml [--only no_tts]
yoruba-asr analyse
yoruba-asr transcribe   path/to/audio.wav --model models/V4 [--no-tone]
yoruba-asr export-tables
yoruba-asr stage <STAGE> [--dry-run]       # same stage names as the notebook
yoruba-asr smoke                           # whole pipeline on a tiny synthetic dataset (CPU, ~2 min)
```
Any config value can be overridden with `--set key=value` (e.g. `--set paths.work_root=./work`). Local runs need `paths.input_root` and `paths.work_root` set to local folders.

## Tests
```bash
python -m pytest                       # 67 tests
python -m yoruba_asr.cli smoke         # end-to-end smoke pipeline
python kaggle/build_notebook.py --test # rebuild + execute the Kaggle notebook locally (STAGE="smoke")
```

## Research safeguards
- The Common Voice **test split is used only for final evaluation**. Dev/test sentences (tone-stripped) and audio are removed from every training, TTS and unlabelled pool (`results/leakage_report.json`).
- Every random process is seeded (`seed = 42`); augmented copies are reproducible from `(seed, clip, copy)`.
- Tone preservation: tempo perturbation is checked to keep F0 within 3%. Pitch-shifting speed perturbation appears only in the ablation.
- Common Voice data is never committed or re-hosted.

## Licences
Code: MIT. Common Voice: CC0 (no re-hosting). OpenSLR 86: CC BY-SA 4.0. ÌròyìnSpeech text: CC BY 4.0. `facebook/mms-tts-yor` and the audio it generates (the demo clips and the V2/V4 synthetic data): **CC BY-NC 4.0**, research use only.

## References
Baevski et al. (2020) *wav2vec 2.0*; Babu et al. (2022) *XLS-R*; Pratap et al. (2024) *MMS*; Ardila et al. (2020) *Common Voice*; Gutkin et al. (2020) *Yoruba speech corpus*; Ògúnrẹ̀mí et al. (2024) *ÌròyìnSpeech*; Liu & Peng (2020) *Blockwise bootstrap*; Olusanya (2026) *Tone in Yoruba ASR*.
