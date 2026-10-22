# ---- Locate data ----------------------------------------------------------------------------------
import json
from yoruba_asr.config import discover_paths

BASE_OVERRIDES = {"paths.input_root": INPUT_ROOT, "paths.work_root": WORK_ROOT, "paths.tmp_dir": TMP_DIR,
                  "data.primary": PRIMARY_DATASET, "data.slr86.download": bool(DOWNLOAD_OPENSLR86) and ON_KAGGLE}
_cfg = pipeline.stage_config("prepare", BASE_OVERRIDES)
FOUND = discover_paths(_cfg)   # downloads OpenSLR 86 here if it is not attached and DOWNLOAD_OPENSLR86=True
print(json.dumps(FOUND.summary(), indent=2, ensure_ascii=False))
_needs_data = any(s in STAGE for s in ("smoke", "prepare", "augment", "all_light", "ablation", "cpt", "V3", "V4"))
if FOUND.common_voice is None and not FOUND.slr86 and _needs_data and not DRY_RUN:
    raise FileNotFoundError(
        "No speech data found. Either attach Common Voice 27.0 Yoruba (private dataset, the .tar.gz), or turn "
        "Internet ON and keep DOWNLOAD_OPENSLR86 = True so OpenSLR 86 is downloaded automatically.")
if FOUND.slr86:
    print("OpenSLR 86 (Google Yoruba corpus):", [str(p) for p in FOUND.slr86])
print("Main dataset setting PRIMARY_DATASET =", PRIMARY_DATASET)
if FOUND.common_voice is not None:
    print("\nCommon Voice:", FOUND.common_voice)
    from yoruba_asr.data.common_voice import index_audio_files
    _n_clips = len(index_audio_files([INPUT_ROOT, f"{TMP_DIR}/extracted"], exts=(".mp3",)))
    print(f"Common Voice MP3 clips found in the attached data: {_n_clips} (the full 27.0 Yoruba release has 4,970)")
    if _n_clips < 1000 and STAGE not in ("smoke",) and not DRY_RUN:
        print("*** WARNING: most audio clips are missing. Re-upload the original .tar.gz archive "
              "(see the error message below and docs/KAGGLE_GUIDE.md). ***")
print("SLR86 folders:", len(FOUND.slr86), "| ÌròyìnSpeech:", FOUND.iroyin or FOUND.iroyin_manifest,
      "| MUSAN:", FOUND.musan, "| previous outputs:", len(FOUND.previous_outputs))
