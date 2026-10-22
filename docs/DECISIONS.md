# Decisions

Choices made where the specification left room (or where the installed libraries differed from it).

## Data and text
1. **Digits:** utterances containing digits are **dropped and logged** (default in Section 8). No Yoruba number-to-words expander is included, because its correctness could not be verified with tests.
2. **Silence trimming** (`top_db=30`) is applied consistently to training, dev, test and inference audio. The duration filter (0.5–15 s) uses the trimmed duration for Common Voice; for large OOD corpora it uses header durations (no decoding at prepare time), and the audio is trimmed when loaded.
3. **Storage:** Common Voice train/dev/test are cached as Hugging Face datasets (`input_values`, `input_length`, `labels`) under `cache/`. OOD, augmented and TTS audio are loaded **on the fly** from FLAC/WAV manifests. A custom `CTCTrainer` groups batches by length using the manifest durations, so no audio is decoded just to find lengths.
4. **TTS sentence pool:** `validated_sentences.tsv`, minus CV dev/test sentences (tone-stripped match) and digit sentences. Sentences never recorded in CV train come first. Only sentences written with characters already in CV train or the canonical Yoruba set are kept. This excludes, for example, `U+031F` ("plus below"), an old encoding of ẹ/ọ found in some sentences.
5. **TTS coverage policy:** `facebook/mms-tts-yor` can represent acute/grave tones (precomposed and combining) and ẹ ọ ṣ, but **not the macron** (mid-tone mark, rarely written). TTS is disabled if high/low tones or sub-dots cannot be represented. Sentences containing any unrepresentable character are skipped individually, never silently stripped. Verified on 2026-10-20 against the real tokenizer (44 symbols).
6. **Unlabelled pool** also contains the audio of OOD utterances dropped for digits (their audio is valid, and CPT needs no text), but never CV dev/test audio.
7. The **vocabulary** keeps the ASCII apostrophe because the original `PUNCT` set does not remove it. The `Wav2Vec2CTCTokenizer` adds `<s>`/`</s>` as extra tokens, as in the original notebook; they never appear in labels.

## Runs actually performed (2026-10-21, presentation deadline)
- **V1** and **V2** were trained with `train.max_steps = 1500` (instead of 3000) to fit the deadline; both used the same budget, so they remain comparable. The dev WER was still falling at step 1400, so longer training should help.
- **V2** skipped mask tuning (`mask_tuning.skip = true`, so mask_time_prob = 0.075, as the spec allows) and ran **without TTS**. The TTS coverage check disabled synthesis because 1 of 2,202 sentences contained `ḿ`, which MMS-TTS cannot write. This was fixed afterwards: such sentences are now skipped individually (2,201 usable). Today's V2 is therefore *acoustic augmentation only* (tempo, noise, reverb, gain; 2,936 copies, 3.8 h).
- Results (CV 27.0 test, 1,096 utterances): V1 WER 58.46 / CER 20.40 / TER 19.29; V2 WER 54.92 / CER 19.54 / TER 19.18. V2 − V1 WER = −3.54 points, 95% CI [−4.54, −2.47], **significant**. TER difference is not significant.

- **V3** (2026-10-21): CPT shortened to 500 steps (0.49 GPU h on 6.06 h of unlabelled audio, warm-up 50, batch 2 × 16 accumulation), B1 = 800 steps on OpenSLR 86 (3,567 utterances), B2 = 1,500 steps on CV train. **V4** (2026-10-22): same CPT encoder and B1, B2 = 1,500 steps on the V2 training set (CV + 2,936 augmented copies, no TTS, mask 0.075).
- Final test results (CV 27.0 test, 1,096 utterances, 101 speakers): V1 WER 58.46 / TER 19.29; V2 54.92 / 19.18; V3 48.35 / 16.57; **V4 46.85 / 16.37**. Every WER difference between variants is significant. TER: V3 and V4 are significantly better than V1 and V2; V2 vs V1 and V4 vs V3 are not significant.
- Factorial effects on WER (speaker-bootstrap 95% CI): augmentation −2.48 [−3.26, −1.74]; domain adaptation −9.08 [−10.01, −8.16]; interaction +2.05 [+0.62, +3.44], which is significant and sub-additive (augmentation helps less once domain adaptation is used). On TER only domain adaptation is significant (−2.77 [−3.22, −2.36]). See `results/factorial_effects_ci.csv`.

## Main dataset choice (added 2026-10-20)
- `data.primary` = `auto` | `common_voice` | `slr86`. `auto` uses Common Voice when it is attached and its clips are present. Otherwise it falls back to OpenSLR 86, with a loud warning and the choice recorded in `data_summary.json`.
- OpenSLR 86 is downloaded automatically from the official mirrors (`data.slr86.download: true`). When it is the main corpus, it gets a seeded, speaker-disjoint split per gender (10% dev, 10% test by utterances, at least one speaker kept for training), and training sentences that also occur in dev/test are removed. Its training audio also goes into the CPT pool, and its training sentences form the TTS text pool.
- OpenSLR 86 transcripts carry annotation tags (`[breath]`, `[snap]`, `[external]`, `[hesitation]`, `[abrupt]`; 1,306 in total). Tags are removed. `[abrupt]` recordings (16) are dropped because their audio is cut off.
- If clips are not in `<cv>/clips/`, they are looked up by file name anywhere under the input (Kaggle sometimes unpacks uploads into other folders). If more than 20% are missing, the run stops with a listing of the dataset contents.

## Training
8. **Metrics in Trainer logs are fractions** (as in the original notebook); **all result files use percentages**. `jiwer` is called directly instead of `evaluate.load("wer")`, so that evaluation works offline. The numbers are identical because `evaluate`'s WER wraps jiwer.
9. `preprocess_logits_for_metrics` reduces logits to argmax ids on the GPU, which keeps memory low during dev evaluation.
10. Incomplete runs (time budget) are **not** written to `models/<V>`. Only a finished run (`max_steps` reached) is, with `training_complete.json`.
11. Checkpoints are deleted after a variant completes (`train.keep_checkpoints: false`), and `models/B1` is deleted once both V3 and V4 are complete, to stay under ~18 GB.
12. **B1 is trained once** (`models/B1`) and reused by V3 and V4 (identical by construction, as the spec allows). B1 uses `mask_time_prob = 0.05` and the Section 3 hyperparameters, with `max_steps = 2000`.
13. The Kaggle stack pins `transformers==4.57.0`, `huggingface-hub==0.36.2`, `datasets==4.1.1` and `accelerate==1.10.1`. The code was tested with these versions, and transformers 4.57 needs huggingface-hub < 1.0. The web app uses **Gradio 5.50**, because Gradio 6 requires huggingface-hub ≥ 1.16.

## Augmentation
14. **Transforms are implemented in numpy/scipy/librosa** (tempo: phase-vocoder `time_stretch`; speed: resampling; noise: MUSAN or generated coloured noise at U(10, 30) dB SNR; reverb: a seeded bank of 64 pyroomacoustics shoebox RIRs with RT60 U(0.2, 0.8) s, image-source order ≤ 17; gain: U(−6, +6) dB, clip-safe). `audiomentations` was not used because its randomness relies on global RNGs, while the spec requires each copy to be reproducible from `(seed, clip, copy)`. Every transform takes an explicit `numpy.random.Generator`.
15. **Pitch check:** the assertion is applied to the *median over 20 clips* of the per-clip relative change in median F0 (pyin). Per-clip values are kept in `pitch_check.json`.
16. The fixed transform order inside a chain is speed → tempo → reverb → noise → gain.

## Continued pretraining
17. The **CNN feature encoder is frozen during CPT** (`cpt.freeze_feature_encoder: true`) to fit 15 s crops into 16 GB. Gradient checkpointing is on. The effective batch is 4 crops × 8 accumulation = 32 crops.
18. CPT's linear schedule is defined over `cpt.max_steps = 12000`. The run also ends at `cpt.max_total_hours = 20` (counted as complete), so CPT takes exactly two sessions whatever the GPU speed.
19. Loss is normalised per masked frame and per micro-batch (`loss / num_masked / grad_accum`). This differs from the HF example, which rescales already-accumulated gradients.
20. **XLS-R checkpoint verified:** `Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")` loads with **no missing keys** (the quantiser and `project_q`/`project_hid` are present). A 2-step CPU run on real CV clips gave finite losses and a codevector perplexity of about 80–90.

## Evaluation
21. Bootstrap CIs are percentile intervals over 1,000 speaker-block resamples. For the bar chart, per-system CIs are computed the same way (`system_cis.json`).
22. `ter_matched_words`: units are aligned position-wise inside word pairs whose tone-stripped forms are equal. A syllabic nasal counts when the **reference** marks it, and an unmarked hypothesis nasal is read as M.
