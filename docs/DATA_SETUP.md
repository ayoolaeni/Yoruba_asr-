# Data setup

All datasets are uploaded to Kaggle as **private** datasets. The notebook finds them anywhere under `/kaggle/input` by their **marker files**, so the dataset names you choose do not matter (except where noted below). Archives (`.tar.gz`, `.zip`) are extracted automatically to `/tmp/yoruba_asr/extracted/`.

| Corpus | Role | Licence | Marker the code looks for |
|---|---|---|---|
| Common Voice Scripted Speech 27.0, Yoruba | **required**: train/dev/test; `other.tsv` is used as unlabelled audio for CPT only | CC0. Its terms **forbid re-hosting**, so keep it private and never commit it | a folder with `train.tsv`, `dev.tsv`, `test.tsv` and `clips/` |
| Google Yoruba corpus (OpenSLR 86) | OOD labelled (stage B1) + unlabelled (CPT) | CC BY-SA 4.0 | a folder with `line_index.tsv` whose ids start with `yo` |
| ÌròyìnSpeech (Ògúnrẹ̀mí et al., 2024) | OOD labelled (B1) + unlabelled (CPT) | text CC BY 4.0; check the audio terms when you download | folder name contains `iroyin`, plus a manifest (see below) |
| BibleTTS Yoruba | optional, unlabelled CPT audio only (disabled by default; enable it with `data.bibletts.enabled: true`) | CC BY-SA | folder name contains `bibletts` |
| MUSAN subset | optional noise source for augmentation | CC BY 4.0 | folder named `musan` |
| OpenSLR 28 RIRs | optional real room impulse responses (otherwise rooms are simulated with pyroomacoustics) | Apache 2.0 | folder name contains `rir` |

## Common Voice 27.0 Yoruba
Download it from the **Mozilla Data Collective** (free account). Expected layout after extraction:
```
cv-corpus-27.0-2026-09-11/yo/
  train.tsv dev.tsv test.tsv validated.tsv invalidated.tsv other.tsv
  validated_sentences.tsv clip_durations.tsv clips/*.mp3
```
Reference counts for release 27.0: train 1,468 · dev 983 · test 1,096 · other 1,152 · invalidated 269. The audit checks these against what it finds; your full archive matches them exactly.

## OpenSLR 86
Download `yo_ng_female.zip` and `yo_ng_male.zip` from https://www.openslr.org/86/ and upload both zips in one private dataset. Each archive contains `line_index.tsv` (`<file_id><TAB><transcript>`) and `<file_id>.wav`. The loader also accepts a space instead of a tab. The speaker is the id prefix (e.g. `yof_00295`).

## ÌròyìnSpeech: manifest format
The release layout is not fixed in the project specification, so the loader needs a **manifest** table:
```csv
path,sentence,speaker
audio/news_0001.wav,Ìjọba ti kéde ètò tuntun fún àwọn àgbẹ̀.,spk01
```
- `path` is relative to the manifest file (or absolute); `speaker` is optional.
- Column names are matched case-insensitively. Accepted audio columns: `path/audio/audio_path/file/filename/wav/audio_filepath`. Accepted text columns: `sentence/text/transcript/transcription/normalized_text`.
- If the release already ships a CSV/TSV/JSONL with such columns, it is detected automatically. Otherwise, write one and set `EXTRA_OVERRIDES = {"data.iroyin.manifest": "/kaggle/input/<your-dataset>/manifest.csv"}`.

## Tone-quality gate for OOD corpora
If more than 20% of a corpus's transcripts carry **no tone marks at all**, the corpus is marked `needs_manual_check` in `results/corpus_audit.json` and **excluded**. Listen to ~50 utterances; if the transcripts are acceptable, include it with `EXTRA_OVERRIDES = {"data.slr86.force_include": True}` (or `data.iroyin.force_include`).

## What the pipeline filters (logged in the audit)
- Utterances whose transcript contains **digits** are dropped (no trusted Yoruba number expansion).
- Clips shorter than 0.5 s or longer than 15 s after silence trimming, and corrupt or unreadable files.
- Any OOD/TTS sentence whose tone-stripped normalised form equals a CV dev/test sentence (leakage control).
