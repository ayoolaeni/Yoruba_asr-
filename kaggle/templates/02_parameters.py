# ============================ PARAMETERS (the only cell you edit) ============================
STAGE   = "prepare"   # one of: "smoke", "prepare", "V1", "augment", "tune_mask", "V2",
                      #         "cpt", "V3", "V4", "evaluate", "ablation", "analyse", "all_light"
                      # stages can be chained with "+", e.g. "augment+tune_mask"
RESUME  = True        # resume from checkpoints found in /kaggle/working or attached previous outputs
ABLATION_ONLY = None  # e.g. "no_tts" to run one ablation per session
PUSH_TO_HUB = False   # needs Kaggle Secret HF_TOKEN
HUB_REPO_PREFIX = "your-hf-username/yoruba-xlsr"   # used only if PUSH_TO_HUB
CPT_MAX_HOURS = 11.0
EXTRA_OVERRIDES = {}  # e.g. {"train.max_steps": 3000}
DRY_RUN = False       # True: print the plan for STAGE without training
PRIMARY_DATASET = "auto"   # "auto": Common Voice if attached and complete, otherwise OpenSLR 86
                           # "slr86": train/test on OpenSLR 86 only (no upload needed)
                           # "common_voice": always use Common Voice
DOWNLOAD_OPENSLR86 = True  # download OpenSLR 86 automatically (~0.9 GB, needs Internet On)
