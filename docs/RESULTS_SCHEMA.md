# Output formats

All error rates in result files are **percentages** (0–100). Trainer logs (`train_log.csv`, `eval_wer`) use **fractions**, as in the original notebook.

## `results/<V>/metrics.json` (V1–V4)
```json
{"variant": "V1", "n_utts": 1090, "wer": 0.0, "cer": 0.0, "wer_no_tone": 0.0, "ter": 0.0,
 "ter_matched_words": 0.0, "tone_confusion": [[0,0,0],[0,0,0],[0,0,0]],
 "train_clips": 0, "train_hours": 0.0, "gpu_hours": 0.0, "mask_time_prob": 0.05,
 "checkpoint": "models/V1", "vocab_sha256": "...", "stage_b1": "not used|complete|skipped"}
```
`tone_confusion` rows are the reference tone and columns the hypothesis tone, both in the order H, M, L.

## `results/<V>/predictions.csv`
`path, speaker, gender, age, duration, reference, hypothesis, reference_notone, hypothesis_notone`

## Other files
| File | Columns / content |
|---|---|
| `summary.csv` | variant, n_utts, wer, cer, wer_no_tone, ter, ter_matched_words, train_clips, train_hours, gpu_hours, mask_time_prob, checkpoint, vocab_sha256 |
| `significance.csv` | system_a, system_b, comparison, metric (wer/ter), value_a, value_b, delta (= A − B), delta_mean, ci_low, ci_high, significant, n_speakers, n_utts, resamples |
| `system_cis.json` | `{variant: {metric: {point, ci_low, ci_high}}}` |
| `factorial_effects.csv` | metric, V1..V4, augmentation_main_effect, adaptation_main_effect, interaction |
| `ablation.csv` | run, description, transforms, use_tts, mask_time_prob, max_steps, train_clips, dev_wer, dev_ter, test_wer, test_ter, test_cer, test_wer_no_tone, status |
| `corpus_audit.json/.md` | per corpus/split: clips_found, clips_kept, hours, speakers, sample_rates, duration mean/min/max, filtered (by reason), tone-mark/sub-dot shares, character inventory; CV reference-count check |
| `leakage_report.json` | forbidden sentence count, removed items per corpus, dev/test audio found in pools (must be 0), speaker overlap |
| `vocab_info.json` | vocab path, **sha256**, size, characters, unknown characters in CV dev/test |
| `data_summary.json` | clip counts and hours of every prepared pool |
| `V2/pitch_check.json` | per-clip median F0 before/after tempo, median relative difference, passed |
| `V2/tts_coverage.json` | missing characters (NFC/NFD), tone/sub-dot/macron representability, `enabled_effective`, synthesised/skipped counts |
| `V2/mask_tuning.json` | candidates `{p: {dev_wer, status}}`, chosen |
| `V3/cpt_log.csv` | step, loss, contrastive_loss, diversity_loss, codevector_perplexity, gumbel_temperature, grad_norm, lr, elapsed_h |
| `V3/cpt_status.json` | completed, stop_reason, completed_steps, total_hours |
| `<stage>/run_info.json` | sessions: wall_seconds, gpu_name, peak_gpu_memory_gb, gpu_hours, steps, steps_per_sec; totals |
| `<V>/error_analysis/` | error_analysis.json, breakdown_{gender,age,length_bucket,subdot_words}.csv, tone_confusion.png, examples.md |
| `figures/` | variants_wer_cer_ter.png, factorial_interaction.png, ablation.png, cpt_loss.png, dev_wer_curves.png (300 dpi, greyscale) |
| `thesis_tables.{md,csv,tex}` | audit, main results, significance, effects, ablation |
