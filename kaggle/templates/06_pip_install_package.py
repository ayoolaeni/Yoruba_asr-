# ---- Install the package (editable) and make it importable in this kernel --------------------
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", CODE_DIR], check=True)
sys.path.insert(0, f"{CODE_DIR}/src")
import importlib
for _m in [m for m in list(sys.modules) if m.startswith("yoruba_asr")]:
    del sys.modules[_m]
import yoruba_asr
from yoruba_asr import pipeline
from yoruba_asr.config import CONFIGS_DIR
print("yoruba_asr", yoruba_asr.__version__, "| configs:", CONFIGS_DIR)
