"""Generate the self-contained Kaggle notebook ``kaggle/yoruba_asr_kaggle.ipynb``.

The notebook embeds the whole ``yoruba_asr`` package (``%%writefile`` cells), the configs and
``pyproject.toml``, so the user only attaches data datasets.

Usage:
    python kaggle/build_notebook.py                 # (re)build the notebook
    python kaggle/build_notebook.py --test          # build + execute it locally with STAGE="smoke"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = Path(__file__).resolve().parent / "templates"
OUT = Path(__file__).resolve().parent / "yoruba_asr_kaggle.ipynb"


def _t(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8").rstrip() + "\n"


def embedded_files() -> list[Path]:
    files = [ROOT / "pyproject.toml", ROOT / "requirements-kaggle.txt"]
    files += sorted((ROOT / "configs").glob("*.yaml"))
    files += sorted(p for p in (ROOT / "src" / "yoruba_asr").rglob("*.py") if "__pycache__" not in p.parts)
    return files


def build(out: Path = OUT, stage: str | None = None) -> Path:
    files = embedded_files()
    rel = [f.relative_to(ROOT).as_posix() for f in files]
    dirs = sorted({str(Path(r).parent.as_posix()) for r in rel if Path(r).parent.as_posix() != "."})
    reqs = (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8").strip()

    cells = [new_markdown_cell(_t("01_title.md"))]
    params = _t("02_parameters.py")
    if stage:
        params = params.replace('STAGE   = "prepare"', f'STAGE   = "{stage}"')
    cells.append(new_code_cell(params, metadata={"tags": ["parameters"]}))
    cells.append(new_markdown_cell("## Environment check"))
    cells.append(new_code_cell(_t("03_environment.py")))
    cells.append(new_markdown_cell("## Install pinned packages"))
    cells.append(new_code_cell(_t("04_install.py").replace("{{REQUIREMENTS}}", reqs)))
    cells.append(new_markdown_cell(
        "## Package source\nThe cells below write the `yoruba_asr` package (data pipeline, augmentation, "
        "domain adaptation, training, evaluation, inference), its configs and `pyproject.toml`. "
        "Collapse them, there is nothing to edit here."))
    cells.append(new_code_cell(_t("05_mkdirs.py").replace("{{DIRS}}", repr(dirs))))
    cells.append(new_code_cell(
        '%%writefile {CODE_DIR}/README.md\n# yoruba_asr\n\nPackage source embedded by the Kaggle notebook. '
        'See the project repository for documentation.\n'))
    for f, r in zip(files, rel):
        body = f.read_text(encoding="utf-8")
        cells.append(new_code_cell(f"%%writefile {{CODE_DIR}}/{r}\n{body}"))
    cells.append(new_code_cell(_t("06_pip_install_package.py")))
    cells.append(new_markdown_cell("## Locate data"))
    cells.append(new_code_cell(_t("07_locate_data.py")))
    cells.append(new_markdown_cell("## Previous outputs (chaining sessions)"))
    cells.append(new_code_cell(_t("08_previous_outputs.py")))
    cells.append(new_markdown_cell("## Run the stage"))
    cells.append(new_code_cell(_t("09_run_stage.py")))
    cells.append(new_markdown_cell("## Summary"))
    cells.append(new_code_cell(_t("10_summary.py")))
    cells.append(new_code_cell(_t("11_next_step.py")))

    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True},
    })
    nbformat.validate(nb)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out))
    return out


def run_local_test(stage: str = "smoke") -> int:
    """Acceptance test: execute the notebook locally on the tiny fixture (no Kaggle paths)."""
    import tempfile

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_tiny_fixture import make_fixture

    root = Path(tempfile.mkdtemp(prefix="yasr_nb_"))
    fixture = make_fixture(root / "input")
    nb_path = build(root / "nb_test.ipynb", stage=stage)
    env = {**os.environ, "YASR_LOCAL_TEST_ROOT": str(root).replace("\\", "/"),
           "YASR_INPUT_ROOT": str(fixture).replace("\\", "/"), "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_path),
           "--output", str(root / "nb_executed.ipynb"), "--ExecutePreprocessor.timeout=1800"]
    print("executing:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env, cwd=str(root))
    print("executed notebook:", root / "nb_executed.ipynb", "| exit code", rc)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="build and execute locally with STAGE='smoke'")
    ap.add_argument("--stage", default="smoke")
    a = ap.parse_args()
    print("written:", build())
    if a.test:
        sys.exit(run_local_test(a.stage))
