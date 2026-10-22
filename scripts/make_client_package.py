"""Assemble the self-contained client package (Docker app + trained models) and zip it.

Usage:  python scripts/make_client_package.py [output_dir]
Default output: ../Yoruba_ASR_Client_Package/ and ../Yoruba_ASR_Client_Package.zip (next to the repo).

The package contains only what is needed to run the web app with Docker: the app, the inference code,
the four trained models, the results shown in the "Results" tab, start/stop scripts and HOW_TO_RUN.txt.
It does NOT contain Common Voice audio or test transcripts.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP_FILES = ["Dockerfile", "docker-compose.yml", ".dockerignore", "START_WINDOWS.bat", "STOP_WINDOWS.bat",
             "start_mac_linux.sh", "stop_mac_linux.sh", "HOW_TO_RUN.txt"]
MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "vocab.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "model_card.md", "training_complete.json"]
RESULT_FILES = ["summary.csv", "significance.csv", "factorial_effects.csv", "factorial_effects_ci.csv",
                "system_cis.json", "thesis_tables.md"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in TOP_FILES:
        shutil.copy2(REPO / f, out / f)
    shutil.copytree(REPO / "app", out / "app", ignore=IGNORE)
    shutil.copytree(REPO / "src" / "yoruba_asr", out / "src" / "yoruba_asr", ignore=IGNORE)

    found = []
    for v in ("V1", "V2", "V3", "V4"):
        src = REPO / "models" / v
        if not (src / "model.safetensors").exists():
            print(f"WARNING: models/{v} not found - skipped")
            continue
        (out / "models" / v).mkdir(parents=True)
        for f in MODEL_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, out / "models" / v / f)
        found.append(v)

    (out / "results").mkdir()
    for f in RESULT_FILES:
        if (REPO / "results" / f).exists():
            shutil.copy2(REPO / "results" / f, out / "results" / f)
    if (REPO / "results" / "figures").exists():
        shutil.copytree(REPO / "results" / "figures", out / "results" / "figures")
    print(f"package folder: {out}  (models: {', '.join(found)})")
    return out


def make_zip(folder: Path) -> Path:
    zpath = folder.with_suffix(".zip")
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                # model weights do not compress; store them to keep zipping fast
                method = zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
                zf.write(p, Path(folder.name) / p.relative_to(folder), compress_type=method)
    print(f"zip: {zpath}  ({zpath.stat().st_size / 1024**3:.2f} GB)")
    return zpath


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "Yoruba_ASR_Client_Package"
    make_zip(build(target))
