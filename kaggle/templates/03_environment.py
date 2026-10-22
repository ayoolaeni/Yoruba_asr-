# ---- Environment check -------------------------------------------------------------------
import os, sys, shutil, platform, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # use ONE GPU only (before importing torch)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ON_KAGGLE = os.path.exists("/kaggle/input")
if ON_KAGGLE:
    CODE_DIR, WORK_ROOT, INPUT_ROOT, TMP_DIR = ("/kaggle/working/yoruba_asr_code", "/kaggle/working",
                                                "/kaggle/input", "/tmp/yoruba_asr")
else:  # local acceptance test (python kaggle/build_notebook.py --test / nbconvert)
    _root = os.environ.get("YASR_LOCAL_TEST_ROOT", os.path.abspath("_nb_local"))
    CODE_DIR, WORK_ROOT, TMP_DIR = (f"{_root}/code", f"{_root}/working", f"{_root}/tmp")
    INPUT_ROOT = os.environ.get("YASR_INPUT_ROOT", f"{_root}/input")
CODE_DIR = CODE_DIR.replace("\\", "/")
for d in (CODE_DIR, WORK_ROOT, TMP_DIR):
    os.makedirs(d, exist_ok=True)

import torch
print("Python      :", platform.python_version())
print("torch       :", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU         :", p.name, f"{p.total_memory / 1024**3:.1f} GB")
else:
    print("GPU         : none (CPU only). For real training set Accelerator = GPU T4/P100.")
import importlib.metadata as _md  # do not import transformers yet: the next cell may pin its version
try:
    print("transformers:", _md.version("transformers"), "(before pinning)")
except _md.PackageNotFoundError:
    print("transformers: not installed yet")
for label, path in (("disk /kaggle/working", WORK_ROOT), ("disk /tmp", TMP_DIR)):
    print(f"{label:12s}: {shutil.disk_usage(path).free / 1024**3:.1f} GB free")
print("\nReminder: Kaggle's weekly GPU quota is about 30 h (see Settings → Quota). Each session is limited "
      "to 12 h; this notebook stops training at ~11 h and saves a checkpoint.")
if ON_KAGGLE and not torch.cuda.is_available() and STAGE not in ("smoke", "prepare", "evaluate", "analyse"):
    print("\n*** WARNING: no GPU detected. Turn on Accelerator = GPU in the session options. ***")
