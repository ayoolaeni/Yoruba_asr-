# ---- Run the stage --------------------------------------------------------------------------------
import time
SESSION_PLAN = """
| Run  | STAGE                  | Attach as input                               | GPU h (T4, estimate) |
|------|------------------------|-----------------------------------------------|----------------------|
| 0    | smoke                  | Common Voice                                  | < 0.3                |
| 1    | prepare                | Common Voice (+ SLR86, ÌròyìnSpeech, MUSAN)   | ~0.5                 |
| 2    | V1                     | data + output of run 1                        | 4-5                  |
| 3    | augment+tune_mask      | data + previous output                        | 3-4                  |
| 4    | V2                     | data + previous output                        | 4-5                  |
| 5-6  | cpt (twice, resumes)   | data + previous output                        | ~20                  |
| 7    | V3                     | data + previous output                        | 5-6                  |
| 8    | V4                     | data + previous output                        | 4-5                  |
| 9    | evaluate               | previous output                               | ~1                   |
| 10+  | ablation (one per run) | data + previous output                        | ~1.5 each            |
| last | analyse                | previous output                               | < 0.5                |
(estimates; the actual GPU time of every stage is logged in results/**/run_info.json)
"""
print(SESSION_PLAN)

overrides = {**BASE_OVERRIDES, **EXTRA_OVERRIDES}
if "cpt" in STAGE:
    overrides["cpt.time_budget_hours"] = float(CPT_MAX_HOURS)
if STAGE == "smoke" and ON_KAGGLE:
    overrides["tts.model"] = "facebook/mms-tts-yor"   # exercise the real TTS + coverage check

_t0 = time.time()
RESULT = pipeline.run_stage(STAGE, overrides, smoke=(STAGE == "smoke"), resume=RESUME, only=ABLATION_ONLY,
                            dry_run=DRY_RUN)
print(f"\nStage '{STAGE}' finished in {(time.time() - _t0) / 3600:.2f} h")
