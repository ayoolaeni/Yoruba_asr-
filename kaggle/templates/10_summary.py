# ---- Summary: key metrics, files written, figures, optional Hub upload ------------------------------
from pathlib import Path
import pandas as pd

_work = Path(WORK_ROOT) / ("smoke" if STAGE == "smoke" else ("light" if STAGE == "all_light" else ""))
_res = _work / "results"
if (_res / "summary.csv").exists():
    print("== Test-set results so far (%) ==")
    print(pd.read_csv(_res / "summary.csv").to_string(index=False))
for v in ("V1", "V2", "V3", "V4"):
    st = _res / v / "run_status.json"
    if st.exists() and v in STAGE:
        print(f"\n{v} status:", json.loads(st.read_text(encoding="utf-8")))
_gpu = 0.0
for ri in _res.rglob("*run_info.json"):
    try:
        _gpu += json.loads(ri.read_text(encoding="utf-8")).get("total_gpu_hours", 0.0)
    except Exception:
        pass
print(f"\nLogged GPU time across all stages so far: {_gpu:.2f} h")

_recent = sorted((p for p in _res.rglob("*") if p.is_file() and p.stat().st_mtime >= _t0 - 5),
                 key=lambda p: p.stat().st_mtime)
print(f"\nFiles written in this run ({len(_recent)}):")
for p in _recent[-60:]:
    print("  ", p.relative_to(_work))

try:
    from IPython.display import Image, display
    for png in sorted((_res / "figures").glob("*.png")):
        print(png.name); display(Image(filename=str(png), width=600))
except Exception as exc:
    print("figures not shown:", exc)

if PUSH_TO_HUB and STAGE in ("V1", "V2", "V3", "V4") and not DRY_RUN:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi
    _token = UserSecretsClient().get_secret("HF_TOKEN")
    _api = HfApi(token=_token)
    _repo = f"{HUB_REPO_PREFIX}-{STAGE.lower()}"
    _api.create_repo(_repo, private=True, exist_ok=True)
    _api.upload_folder(folder_path=str(_work / "models" / STAGE), repo_id=_repo)
    print("pushed", STAGE, "to https://huggingface.co/" + _repo, "(private)")
    del _token
