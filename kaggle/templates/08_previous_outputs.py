# ---- Previous outputs: chain sessions (prepare -> V1 -> ...) ----------------------------------------
from yoruba_asr.config import copy_previous_outputs

if FOUND.previous_outputs and not DRY_RUN:
    print("Copying outputs of previous notebook versions into", WORK_ROOT, "(skips files already present) ...")
    for line in copy_previous_outputs(FOUND, __import__("pathlib").Path(WORK_ROOT)):
        print("  copied", line)
else:
    print("No previous output attached." if not FOUND.previous_outputs else "dry run: nothing copied")
print("Work dir contents:", sorted(os.listdir(WORK_ROOT)))
