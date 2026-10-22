# Yoruba ASR on Kaggle: XLS-R + hybrid data augmentation + domain adaptation (V1–V4)

**Project:** Enhancing ASR Accuracy for Low-Resource Nigerian Tonal Languages Through Hybrid Data Augmentation and Domain Adaptation: A Case Study of Yoruba.

This notebook runs the **whole training and evaluation workflow** of the project, **one stage per run**. The `yoruba_asr` package source is embedded below, so you only attach your **data**. You do not need a code dataset.

### Kaggle settings (right-hand panel → *Session options*)
1. **Accelerator:** GPU T4 (×1 or ×2; only one GPU is used) or GPU P100.
2. **Internet:** **On**. This needs a phone-verified account. It is used to download XLS-R, MMS-TTS and pip packages.
3. **Persistence:** *Files only* (optional, for interactive sessions).

### Datasets to attach (*Add Input*)
| Dataset | Required? | How |
|---|---|---|
| Common Voice Scripted Speech **27.0 Yoruba** (`.tar.gz` from the Mozilla Data Collective) | **yes** | upload it as a **PRIVATE** Kaggle dataset; never make it public |
| Google Yoruba corpus **OpenSLR 86** (`yo_ng_female.zip`, `yo_ng_male.zip`) | recommended (V3/V4 stage B1 + CPT) | private dataset |
| **ÌròyìnSpeech** (audio + a `manifest.csv` with `path,sentence,speaker`) | recommended | private dataset; put "iroyin" in the folder name |
| MUSAN subset (noise/music/speech) | optional (V2/V4 noise) | private dataset named `musan` |
| **Output of the previous run of this notebook** | from run 2 on | *Add Input → Your Work → this notebook → previous version* |

### How to use
1. Edit **only the PARAMETERS cell**: set `STAGE` (see the session plan printed below).
2. Click **Save Version → Save & Run All (Commit)**.
3. When the run finishes, open its **Output**, then attach that output as an input to the next version and set the next `STAGE`. The last cell tells you what to run next.

Full click-by-click guide: `docs/KAGGLE_GUIDE.md` in the project repository.
