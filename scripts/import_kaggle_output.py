"""Copy trained models and results from a downloaded Kaggle notebook output into this repository.

Usage:
    python scripts/import_kaggle_output.py path/to/kaggle_output.zip
    python scripts/import_kaggle_output.py path/to/extracted_output_folder

Copies models/V1..V4 -> yoruba-asr/models/ and results/ -> yoruba-asr/results/, so the web app
(python app/app.py) finds them. Only finished models (with training_complete.json) are copied.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VARIANTS = ("V1", "V2", "V3", "V4")


def _find_root(folder: Path) -> Path:
    """The folder that contains models/ (Kaggle zips may add an extra top-level folder)."""
    for cand in [folder, *[p for p in folder.rglob("models") if p.is_dir()]]:
        root = cand if (cand / "models").is_dir() else cand.parent
        if (root / "models").is_dir():
            return root
    raise SystemExit(f"No 'models' folder found in {folder}")


def main(src: str) -> None:
    src_path = Path(src)
    tmp = None
    if src_path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="kaggle_out_"))
        with zipfile.ZipFile(src_path) as zf:
            zf.extractall(tmp)
        src_path = tmp
    root = _find_root(src_path)
    copied = []
    for v in VARIANTS:
        m = root / "models" / v
        if (m / "config.json").exists():
            if not (m / "training_complete.json").exists():
                print(f"skip {v}: training not finished (no training_complete.json)")
                continue
            dst = REPO / "models" / v
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(m, dst)
            copied.append(v)
    if (root / "results").is_dir():
        shutil.copytree(root / "results", REPO / "results", dirs_exist_ok=True)
        print("results/ copied")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print("models copied:", copied or "none")
    print("Start the app with:  python app/app.py   then open http://127.0.0.1:7870")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
