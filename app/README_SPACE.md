---
title: Yoruba ASR (XLS-R)
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app/app.py
pinned: false
license: cc-by-nc-4.0
---

# Deploying the demo to a Hugging Face Space

1. Push the trained models to the Hub. Set `PUSH_TO_HUB=True` in the Kaggle notebook, or upload `models/V1..V4`
   yourself (for example with `huggingface-cli upload your-name/yoruba-xlsr-v4 models/V4`).
2. Create a new **Gradio** Space and upload these folders/files from the repository:
   - `app/` (including `assets/`)
   - `src/yoruba_asr/` (the app imports it from `src/`)
   - `app/requirements.txt`, copied to the Space root as `requirements.txt`
   - this file, copied to the Space root as `README.md` (the YAML header configures the Space)
3. Edit `app/models.yaml` so each variant points to its Hub repo, for example:
   ```yaml
   models:
     V1: "your-name/yoruba-xlsr-v1"
     V4: "your-name/yoruba-xlsr-v4"
   ```
   Private repos need a Space secret `HF_TOKEN`.
4. Optionally copy `results/` (CSV files and `figures/`) so the **Results** tab can show your thesis numbers.

The Space runs on a free CPU. Expect a few seconds per clip after the first model has loaded.
Licence note: the demo clips and any TTS-augmented models use `facebook/mms-tts-yor` (CC BY-NC 4.0), so they
are for research and demonstration only.
